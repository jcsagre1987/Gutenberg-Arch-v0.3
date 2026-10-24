#!/usr/bin/env python3
# Copyright 2026 Juan Carlo Sagre
# Gutenberg Arch — 2026 Series
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Gutenberg Arch v0.3 — SSD backbone plus delta-rule associative memory.

v0.3 is an experimental branch built by inheriting the working v0.2 implementation:
- same chunked SSD-inspired selective SSM and sequentially carried state
- same input-dependent delta/B/C projections and stable negative A decay
- same causal depthwise convolution, residuals, pre-norms and gated FFN
- same tokenizer, vocabulary contract, and tied token-embedding / LM-head weights
- exact transfer of every compatible v0.2 checkpoint tensor

Targeted change based on the AR/MQAR diagnostics:
- add a causal, differentiable key-value fast-weight memory at the LM hidden output
- use one shared address projection for writes and reads, so the same token has the
  same address representation in a pair-write and query-read position
- use a gated delta-rule write: M <- decay*M + eta*k*(v - k^T M)
  rather than relying solely on the v0.2 SSM state to learn the whole association
- read before updating at each time step, preventing the current token from leaking
  into its own memory read

This is a research architecture, not a claim that AR/MQAR is solved. The memory is
learned and causal; it does not hard-code synthetic token IDs or use labels at inference.
CPU-first. No CUDA/custom kernels or third-party neural-network dependencies are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import Unified_Gutenberg_Arch_v0_2_SSD as v2

ARCHITECTURE_NAME = "Unified Gutenberg Arch v0.3 Associative SSD"
PARENT_ARCHITECTURE_NAME = v2.ARCHITECTURE_NAME
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "Models"
DEFAULT_BASE_V2 = DEFAULT_MODELS_DIR / "unified_gutenberg_ssd_base_v0_2.pt"
DEFAULT_BASE_V3 = DEFAULT_MODELS_DIR / "unified_gutenberg_ssd_base_v0_3.pt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


class DeltaRuleAssociativeMemory(nn.Module):
    """Causal key-value memory with a shared address projection and delta-rule writes.

    At time t, the routing address comes from the previous input token. This supports
    adjacent key/value sequences: when the current token is a value, its preceding key
    addresses the write; when the current token is an answer marker, its preceding key
    addresses the read. This is a learned local-transition memory, not a hard-coded
    detector for any special token.

    State per example and head has shape (key_dim, value_dim). Reads are computed from
    the state *before* the current time step's write. The write gate is learned from
    previous-token embedding, current-token embedding, and the v0.2 backbone hidden.
    """

    def __init__(self, d_model: int, requested_heads: int) -> None:
        super().__init__()
        if d_model < 1 or requested_heads < 1:
            raise ValueError("d_model and requested_heads must be positive")
        # Ensure the memory-head count divides d_model, even for unusual user configs.
        self.n_heads = math.gcd(d_model, requested_heads)
        self.d_model = d_model
        self.head_dim = d_model // self.n_heads
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(3 * d_model, self.n_heads, bias=True)
        self.read_gate = nn.Linear(d_model, self.n_heads, bias=True)
        self.memory_to_hidden = nn.Linear(d_model, d_model, bias=False)
        self.write_rate_logits = nn.Parameter(torch.zeros(self.n_heads))  # sigmoid -> 0.5
        retention = 0.995
        retention_logit = math.log(retention / (1.0 - retention))
        self.decay_logits = nn.Parameter(torch.full((self.n_heads,), retention_logit))

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.memory_to_hidden.weight)
        nn.init.xavier_uniform_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -1.5)
        nn.init.xavier_uniform_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, -0.5)

    def project_address(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Shared normalized address function used for both writes and reads."""
        projected = self.key_proj(token_embeddings)
        projected = projected.reshape(*projected.shape[:-1], self.n_heads, self.head_dim)
        return F.normalize(projected, p=2.0, dim=-1, eps=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        backbone_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if token_embeddings.shape != backbone_hidden.shape:
            raise ValueError("token_embeddings and backbone_hidden must have matching shapes")
        if token_embeddings.shape[:2] != input_ids.shape:
            raise ValueError("token embedding shape does not match input_ids")

        batch, length, _ = backbone_hidden.shape
        # At t, the previous token supplies the address. Position zero has no predecessor.
        previous_embeddings = torch.zeros_like(token_embeddings)
        if length > 1:
            previous_embeddings[:, 1:, :] = token_embeddings[:, :-1, :]

        addresses = self.project_address(previous_embeddings)  # (B,T,H,K)
        values = self.value_proj(self.value_norm(backbone_hidden))
        values = values.reshape(batch, length, self.n_heads, self.head_dim)
        write_features = torch.cat(
            (previous_embeddings, token_embeddings, backbone_hidden), dim=-1
        )
        write_gate = torch.sigmoid(self.write_gate(write_features))  # (B,T,H)
        read_gate = torch.sigmoid(self.read_gate(backbone_hidden))  # (B,T,H)
        decay = torch.sigmoid(self.decay_logits).clamp(min=0.90, max=0.9999).view(1, self.n_heads, 1, 1)
        write_rate = torch.sigmoid(self.write_rate_logits).view(1, self.n_heads, 1)

        # Float32 memory arithmetic makes the recurrent write/read stable on CPU.
        state = torch.zeros(
            batch, self.n_heads, self.head_dim, self.head_dim,
            dtype=torch.float32, device=backbone_hidden.device,
        )
        reads: list[torch.Tensor] = []
        for t in range(length):
            address = addresses[:, t].float()  # shared write address / read query
            # Read BEFORE the current write; current token cannot read its own value.
            retrieved = torch.einsum("bhk,bhkv->bhv", address, state)
            retrieved = retrieved * read_gate[:, t, :, None].float()
            reads.append(retrieved)

            # Decay the old association, predict current value at this address, and
            # write only the residual. Repeated keys therefore update rather than
            # unconditionally accumulating duplicate key-value vectors.
            state = state * decay
            predicted = torch.einsum("bhk,bhkv->bhv", address, state)
            target_value = values[:, t].float()
            error = target_value - predicted
            eta = (write_rate * write_gate[:, t, :, None].float()).unsqueeze(-1)
            state = state + eta * address.unsqueeze(-1) * error.unsqueeze(-2)

        read_tensor = torch.stack(reads, dim=1).reshape(batch, length, self.d_model)
        return self.memory_to_hidden(read_tensor.to(dtype=backbone_hidden.dtype))


class UnifiedGutenbergSSDv3(v2.UnifiedGutenbergSSD):
    """v0.2 backbone plus a learnable delta-rule associative-memory path.

    All v0.2 module names and parameter shapes are retained. A v0.2 checkpoint can be
    loaded into this class with only the new associative-memory parameters initialized.
    Set ``disable_associative_memory=True`` to run the exact v0.2 path for an ablation.
    """

    def __init__(self, config: Optional[v2.UnifiedGutenbergSSDConfig] = None) -> None:
        super().__init__(config)
        self.assoc_memory = DeltaRuleAssociativeMemory(self.config.d_model, self.config.n_heads)
        # Small nonzero initialization gives memory parameters gradients from step one,
        # while keeping the transferred v0.2 backbone dominant at initialization.
        self.memory_gain = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        disable_associative_memory: bool = False,
    ):
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have non-empty shape (batch, sequence).")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor.")
        if input_ids.device.type != "cpu":
            raise ValueError("This prototype currently supports CPU only.")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Sequence length exceeds max_seq_len={self.config.max_seq_len}.")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape.")

        token_embeddings = self.embedding_dropout(self.token_embedding(input_ids))
        hidden = token_embeddings
        for block in self.blocks:
            hidden = block(hidden)

        if not disable_associative_memory:
            # Use the same token embedding table for key/query addresses; values are
            # context-aware backbone states. No target labels are passed to memory.
            memory_context = self.assoc_memory(
                input_ids, self.token_embedding(input_ids), hidden
            )
            hidden = hidden + self.memory_gain.to(dtype=hidden.dtype) * memory_context

        logits = self.lm_head(self.final_norm(hidden))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long(),
            ignore_index=v2.PAD_ID,
        )
        return logits, loss


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v2_checkpoint_as_v3(path: str | Path) -> tuple[UnifiedGutenbergSSDv3, dict[str, Any]]:
    """Load every compatible v0.2 tensor; initialize only v0.3 memory tensors."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != PARENT_ARCHITECTURE_NAME:
        raise ValueError(
            f"Expected v0.2 checkpoint architecture {PARENT_ARCHITECTURE_NAME!r}; "
            f"got {payload.get('architecture')!r}"
        )
    config = v2.UnifiedGutenbergSSDConfig(**payload["config"])
    model = UnifiedGutenbergSSDv3(config).cpu()
    source_state = payload["model_state_dict"]
    result = model.load_state_dict(source_state, strict=False)
    expected_missing = {name for name in result.missing_keys if name.startswith("assoc_memory.")}
    expected_missing.add("memory_gain")
    if config.tie_embeddings:
        expected_missing.add("lm_head.weight")
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "Unexpected v0.2 -> v0.3 checkpoint mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    # Assert exact transfer of all source tensors included in v0.2's checkpoint.
    loaded_state = model.state_dict()
    for name, source_tensor in source_state.items():
        if name not in loaded_state:
            raise RuntimeError(f"Source tensor missing from v0.3: {name}")
        if not torch.equal(source_tensor, loaded_state[name]):
            raise RuntimeError(f"Source tensor was not copied exactly: {name}")
    if config.tie_embeddings and model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError("Tied embeddings were not preserved during v0.2 transfer")
    metadata = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "source_architecture": payload.get("architecture"),
        "source_parameter_count": payload.get("parameter_count"),
        "v3_parameter_count": model.num_parameters(),
        "copied_source_tensor_count": len(source_state),
        "new_memory_parameters": sum(p.numel() for n, p in model.named_parameters() if n.startswith("assoc_memory.")) + model.memory_gain.numel(),
        "exact_v2_tensor_transfer": True,
    }
    return model, metadata


def save_v3_checkpoint(path: str | Path, model: UnifiedGutenbergSSDv3, metadata: Optional[dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if model.config.tie_embeddings:
        state.pop("lm_head.weight", None)
    payload = {
        "architecture": ARCHITECTURE_NAME,
        "parent_architecture": PARENT_ARCHITECTURE_NAME,
        "config": asdict(model.config),
        "model_state_dict": state,
        "metadata": metadata or {},
        "parameter_count": model.num_parameters(),
        "tied_embeddings": model.config.tie_embeddings,
        "checkpoint_role": "experimental_v0_3_base_not_validated_for_AR_MQAR",
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)
    return path


def load_v3_checkpoint(path: str | Path) -> UnifiedGutenbergSSDv3:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(f"Unexpected architecture: {payload.get('architecture')!r}")
    config = v2.UnifiedGutenbergSSDConfig(**payload["config"])
    model = UnifiedGutenbergSSDv3(config).cpu()
    state = payload["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    expected_missing = {"lm_head.weight"} if config.tie_embeddings else set()
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"v0.3 checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    if config.tie_embeddings and model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError("Tied embeddings were not restored in v0.3 checkpoint")
    return model


def _finite_gradients(model: nn.Module) -> tuple[list[str], list[str]]:
    missing = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    nonfinite = [name for name, param in model.named_parameters() if param.grad is not None and not torch.isfinite(param.grad).all().item()]
    return missing, nonfinite


def run_self_tests(threads: int = 2) -> dict[str, Any]:
    """Run lightweight CPU correctness tests without altering any source checkpoint."""
    torch.manual_seed(90303)
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    report: dict[str, Any] = {
        "architecture": ARCHITECTURE_NAME,
        "parent_architecture": PARENT_ARCHITECTURE_NAME,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": "CPU",
        "threads": torch.get_num_threads(),
    }

    # Test 1: original v2 chunked SSM implementation still matches its sequential reference.
    op = v2.ChunkedSelectiveSSM(d_inner=16, n_heads=4, d_state=5, chunk_size=4)
    x = torch.randn(2, 11, 16, requires_grad=True)
    delta = (torch.rand(2, 11, 4) * 0.5 + 0.03).requires_grad_()
    write = torch.randn(2, 11, 4, 5, requires_grad=True)
    read = torch.randn(2, 11, 4, 5, requires_grad=True)
    chunked, state = op(x, delta, write, read, return_state=True)
    reference, reference_state = op.forward_sequential_reference(x, delta, write, read)
    error = float((chunked - reference).abs().max().item())
    state_error = float((state - reference_state).abs().max().item())
    if error > 2e-5 or state_error > 2e-5:
        raise AssertionError(f"v2 SSM recurrence changed: output_error={error}, state_error={state_error}")
    report["test_v2_chunked_recurrence_unchanged"] = "PASS"
    report["chunked_reference_max_abs_error"] = error
    report["final_state_max_abs_error"] = state_error

    # Test 2: v3 exact v2-backbone parity with the new branch gated off, then full gradients.
    config = v2.UnifiedGutenbergSSDConfig(
        d_model=32, n_layers=2, d_state=8, n_heads=4, expand=2,
        d_conv=4, ffn_mult=2, chunk_size=8, max_seq_len=40, dropout=0.0,
        use_lrn=False, tie_embeddings=True,
    )
    base_model = v2.UnifiedGutenbergSSD(config).cpu().eval()
    model = UnifiedGutenbergSSDv3(config).cpu().eval()
    transfer = model.load_state_dict(base_model.state_dict(), strict=False)
    expected_missing = {n for n in transfer.missing_keys if n.startswith("assoc_memory.")}
    expected_missing.add("memory_gain")
    if set(transfer.missing_keys) != expected_missing or transfer.unexpected_keys:
        raise AssertionError(f"Unexpected v2->v3 copy mismatch: {transfer}")
    model.memory_gain.data.zero_()
    ids = torch.randint(0, 256, (2, 28), dtype=torch.long)
    with torch.inference_mode():
        logits_v2 = base_model(ids)
        logits_v3_disabled = model(ids, disable_associative_memory=True)
    copy_error = float((logits_v2 - logits_v3_disabled).abs().max().item())
    if copy_error != 0.0:
        raise AssertionError(f"v3 disabled-memory path does not match v2: {copy_error}")
    report["test_v2_backbone_exact_parity_when_memory_disabled"] = "PASS"
    report["v2_vs_v3_disabled_memory_max_abs_error"] = copy_error

    model.memory_gain.data.fill_(0.05)
    model.train()
    train_ids = torch.randint(0, 160, (2, 28), dtype=torch.long)
    targets = torch.randint(0, 160, (2, 28), dtype=torch.long)
    logits, loss = model(train_ids, targets)
    if logits.shape != (2, 28, config.vocab_size) or not torch.isfinite(loss).item():
        raise AssertionError("Invalid v3 forward output or loss")
    loss.backward()
    missing, nonfinite = _finite_gradients(model)
    if missing or nonfinite:
        raise AssertionError(f"v3 backward invalid: missing={missing}, nonfinite={nonfinite}")
    memory_grad_groups = [
        "assoc_memory.key_proj.weight", "assoc_memory.value_proj.weight",
        "assoc_memory.write_gate.weight", "assoc_memory.read_gate.weight",
        "assoc_memory.memory_to_hidden.weight", "memory_gain",
    ]
    for name in memory_grad_groups:
        parameter = dict(model.named_parameters())[name]
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise AssertionError(f"Expected finite gradient for {name}")
    report["test_v3_forward_backward_finite_gradients"] = "PASS"
    report["smoke_parameter_count"] = model.num_parameters()
    report["smoke_loss"] = float(loss.detach().item())
    report["new_memory_gradient_groups_connected"] = memory_grad_groups

    # Test 3: strict causal-prefix invariance with the memory branch enabled.
    model.eval()
    altered = ids[:1].clone()
    altered[:, 17:] = torch.randint(0, 256, altered[:, 17:].shape)
    with torch.inference_mode():
        prefix_a = model(ids[:1])[:, :17]
        prefix_b = model(altered)[:, :17]
    causal_error = float((prefix_a - prefix_b).abs().max().item())
    if causal_error > 2e-5:
        raise AssertionError(f"v3 causal-prefix test failed: {causal_error}")
    report["test_v3_causal_prefix"] = "PASS"
    report["causality_max_abs_error"] = causal_error

    # Test 4: checkpoint round-trip and tied embeddings using a temporary file.
    with tempfile.TemporaryDirectory(prefix="unified_gutenberg_v3_test_") as tmp:
        checkpoint = Path(tmp) / "v3_test.pt"
        save_v3_checkpoint(checkpoint, model, {"self_test": True})
        reloaded = load_v3_checkpoint(checkpoint).eval()
        with torch.inference_mode():
            expected = model(ids)
            actual = reloaded(ids)
        checkpoint_error = float((expected - actual).abs().max().item())
        if checkpoint_error != 0.0:
            raise AssertionError(f"v3 checkpoint round-trip mismatch: {checkpoint_error}")
        if reloaded.lm_head.weight is not reloaded.token_embedding.weight:
            raise AssertionError("v3 tied embeddings were not restored")
        report["checkpoint_test_size_bytes"] = checkpoint.stat().st_size
    report["test_v3_checkpoint_round_trip_tied_embeddings"] = "PASS"
    report["checkpoint_max_abs_error"] = checkpoint_error
    report["overall"] = "PASS"
    return report


def initialize_v3_from_v2(source: Path, destination: Path, seed: int, report_dir: Path) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must differ; v2 checkpoint is read-only")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing v3 checkpoint: {destination}")
    source_hash_before = sha256_file(source)
    torch.manual_seed(seed)
    model, transfer = load_v2_checkpoint_as_v3(source)
    save_metadata = {
        **transfer,
        "v3_seed_for_new_memory_parameters": seed,
        "status": "INITIALIZED_NOT_TRAINED_NOT_VALIDATED_FOR_AR_MQAR",
        "design_change": "v0.2 backbone retained; add causal gated delta-rule associative memory",
        "source_checkpoint_read_only": True,
    }
    save_v3_checkpoint(destination, model, save_metadata)
    reloaded = load_v3_checkpoint(destination).eval()
    model.eval()
    probe = torch.tensor([[v2.BOS_ID, 5, 16, 96, 6, 16, 7]], dtype=torch.long)
    with torch.inference_mode():
        expected = model(probe)
        actual = reloaded(probe)
    reload_error = float((expected - actual).abs().max().item())
    if reload_error != 0.0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"v3 checkpoint reload verification failed: {reload_error}")
    source_hash_after = sha256_file(source)
    if source_hash_after != source_hash_before:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Source v0.2 checkpoint SHA-256 changed unexpectedly")
    report = {
        "status": "PASS_V3_INITIALIZED_FROM_V2_NOT_TRAINED",
        "source_checkpoint": str(source),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "source_checkpoint_unchanged": True,
        "destination_checkpoint": str(destination),
        "destination_sha256": sha256_file(destination),
        "destination_size_bytes": destination.stat().st_size,
        "source_tensor_count_copied": transfer["copied_source_tensor_count"],
        "v2_parameter_count": transfer["source_parameter_count"],
        "v3_parameter_count": reloaded.num_parameters(),
        "exact_v2_tensor_transfer": True,
        "checkpoint_round_trip_max_abs_error": reload_error,
        "training_performed": False,
        "ar_mqar_capability_claim": "NOT_TESTED",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"unified_gutenberg_v3_initialization_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run CPU correctness tests without writing project checkpoints")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--print-default-params", action="store_true")
    parser.add_argument("--init-from-v2", type=Path, help="Initialize a new v0.3 checkpoint from a read-only v0.2 checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_BASE_V3, help="Destination for --init-from-v2; refuses to overwrite")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=30326, help="Seed for new v0.3 memory parameters during transfer")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if not args.self_test and not args.print_default_params and args.init_from_v2 is None:
        parser.error("Use --self-test, --print-default-params, or --init-from-v2")

    if args.print_default_params:
        model = UnifiedGutenbergSSDv3(v2.UnifiedGutenbergSSDConfig())
        print(json.dumps({
            "architecture": ARCHITECTURE_NAME,
            "parent_architecture": PARENT_ARCHITECTURE_NAME,
            "config": asdict(model.config),
            "v2_parameter_count_for_same_config": v2.UnifiedGutenbergSSD(model.config).num_parameters(),
            "v3_parameter_count": model.num_parameters(),
            "new_parameter_count": model.num_parameters() - v2.UnifiedGutenbergSSD(model.config).num_parameters(),
        }, indent=2))
    if args.self_test:
        result = run_self_tests(args.threads)
        print(json.dumps(result, indent=2))
    if args.init_from_v2 is not None:
        initialize_v3_from_v2(args.init_from_v2, args.output, args.seed, args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
