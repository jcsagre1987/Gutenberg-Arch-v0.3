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

"""Unified Gutenberg Arch v0.2 — CPU-first, chunked selective SSM / SSD-inspired LM.

This is a research prototype inspired by Mamba-2's chunkwise State Space Duality (SSD),
not a paper-exact or optimized reproduction of the official Mamba-2 implementation.

Key design:
* input-dependent delta, B and C projections (selective SSM)
* causal depthwise convolution implemented with explicit left padding
* blockwise intra-chunk matrix operations with recurrent state passing between chunks
* Legacy-inspired stable negative decay, residual paths, pre-norms and tied embeddings
* optional LocalResponseNorm (off by default; supplied LRN source is Local Response Norm)
* standalone CPU operation; no einops, CUDA extension, Transformers, or custom kernel required

The architecture cannot be guaranteed to pass AR/MQAR. Use the provided self-tests for
implementation correctness, then a held-out recall benchmark for capability validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHITECTURE_NAME = "Unified Gutenberg Arch v0.2 SSD-Inspired"
PAD_ID, BOS_ID, EOS_ID = 258, 256, 257
DEFAULT_VOCAB_SIZE = 259


@dataclass
class UnifiedGutenbergSSDConfig:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    n_heads: int = 4
    expand: int = 2
    d_conv: int = 4
    ffn_mult: int = 2
    chunk_size: int = 32
    max_seq_len: int = 128
    dropout: float = 0.0
    use_lrn: bool = False
    lrn_size: int = 5
    tie_embeddings: bool = True

    def validate(self) -> None:
        if self.vocab_size < 259:
            raise ValueError("vocab_size must be at least 259 for Gutenberg byte tokens.")
        for name in ("d_model", "n_layers", "d_state", "n_heads", "expand", "d_conv", "ffn_mult", "chunk_size", "max_seq_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1.")
        d_inner = self.d_model * self.expand
        if self.n_heads > d_inner or d_inner % self.n_heads != 0:
            raise ValueError("n_heads must divide d_model * expand.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1).")
        if self.lrn_size < 1 or self.lrn_size % 2 == 0:
            raise ValueError("lrn_size must be a positive odd integer.")


class GutenbergByteTokenizer:
    vocab_size = DEFAULT_VOCAB_SIZE
    pad_id, bos_id, eos_id = PAD_ID, BOS_ID, EOS_ID

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        values = [int(t) for t in token_ids if 0 <= int(t) <= 255]
        return bytes(values).decode("utf-8", errors="replace")


class ChunkedSelectiveSSM(nn.Module):
    """Selective SSM with SSD-inspired chunked matrix evaluation.

    For each head, A_h is a stable negative scalar. Per-token delta_t, B_t and C_t are
    input-dependent. Within each chunk, all pairwise causal interactions are evaluated
    using a lower-triangular decay matrix and batched matrix multiplication. Only the
    final state is carried sequentially from one chunk to the next.

    Tensors:
      x: (batch, length, inner); reshape to (batch, length, heads, head_dim)
      delta: (batch, length, heads); B,C: (batch, length, heads, state)
      state: (batch, heads, head_dim, state)

    This formulation uses O(chunk_size^2 * batch * heads) temporary interaction memory.
    It is deliberately explicit and correctness-testable, not yet optimized for long contexts.
    """
    def __init__(self, d_inner: int, n_heads: int, d_state: int, chunk_size: int) -> None:
        super().__init__()
        if d_inner % n_heads:
            raise ValueError("d_inner must be divisible by n_heads")
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = d_inner // n_heads
        self.d_state = d_state
        self.chunk_size = chunk_size
        # A_h = -exp(A_log_h) guarantees strictly negative continuous-time decay.
        self.A_log = nn.Parameter(torch.arange(1, n_heads + 1, dtype=torch.float32).log())

    def _chunk(
        self,
        x: torch.Tensor,       # (B,T,H,P)
        delta: torch.Tensor,   # (B,T,H)
        Bv: torch.Tensor,      # (B,T,H,N)
        Cv: torch.Tensor,      # (B,T,H,N)
        state: torch.Tensor,  # (B,H,P,N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, length, heads, head_dim = x.shape
        A = -torch.exp(self.A_log.float()).view(1, 1, heads)
        log_a = delta.float() * A                 # (B,T,H), <= 0
        log_cum = torch.cumsum(log_a, dim=1)      # log(prod_{j<=t} a_j)
        cum = torch.exp(log_cum)                  # stable decay, (B,T,H)

        # Pairwise transition prod_{j=s+1..t} a_j = exp(cumlog_t-cumlog_s).
        # Mask before exponentiation so upper-triangular entries cannot overflow.
        log_by_head = log_cum.transpose(1, 2)    # (B,H,T)
        log_ratio = log_by_head.unsqueeze(-1) - log_by_head.unsqueeze(-2)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        log_ratio = log_ratio.masked_fill(~causal.view(1, 1, length, length), float("-inf"))
        transition = torch.exp(log_ratio)         # (B,H,T,T)

        # Dynamic write/read vectors: C_t dot B_s. Apply scalar per-head decay.
        read_write = torch.einsum("bthn,bshn->bhts", Cv.float(), Bv.float())  # (B,H,T,T)
        interaction = transition * read_write
        x_delta = x.float() * delta.float().unsqueeze(-1)  # (B,T,H,P)
        local = torch.matmul(interaction, x_delta.permute(0, 2, 1, 3))
        local = local.permute(0, 2, 1, 3)                  # (B,T,H,P)

        # Contribution of state passed from the previous chunk.
        carried = torch.einsum("bthn,bhpn,bth->bthp", Cv.float(), state.float(), cum)
        y = local + carried

        # Last state = decay_to_end * incoming_state + all writes in this chunk.
        last_log = log_by_head[:, :, -1:]                # (B,H,1)
        last_ratio = torch.exp(last_log - log_by_head)   # (B,H,T)
        written_state = torch.einsum(
            "bhs,bhsp,bhsn->bhpn",
            last_ratio,
            x_delta.permute(0, 2, 1, 3),
            Bv.float().permute(0, 2, 1, 3),
        )
        final_decay = torch.exp(last_log.squeeze(-1)).view(bsz, heads, 1, 1)
        next_state = final_decay * state.float() + written_state
        return y.to(dtype=x.dtype), next_state.to(dtype=state.dtype)

    def forward(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        Bv: torch.Tensor,
        Cv: torch.Tensor,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_inner:
            raise ValueError(f"x must have shape (B,L,{self.d_inner})")
        bsz, length, _ = x.shape
        xh = x.reshape(bsz, length, self.n_heads, self.head_dim)
        state = x.new_zeros(bsz, self.n_heads, self.head_dim, self.d_state)
        outputs: list[torch.Tensor] = []
        for start in range(0, length, self.chunk_size):
            end = min(length, start + self.chunk_size)
            y_chunk, state = self._chunk(
                xh[:, start:end], delta[:, start:end], Bv[:, start:end], Cv[:, start:end], state
            )
            outputs.append(y_chunk)
        y = torch.cat(outputs, dim=1).reshape(bsz, length, self.d_inner)
        return (y, state) if return_state else y

    def forward_sequential_reference(
        self, x: torch.Tensor, delta: torch.Tensor, Bv: torch.Tensor, Cv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slow reference recurrence, used only to validate the chunked math."""
        bsz, length, _ = x.shape
        xh = x.reshape(bsz, length, self.n_heads, self.head_dim).float()
        state = x.new_zeros(bsz, self.n_heads, self.head_dim, self.d_state).float()
        A = -torch.exp(self.A_log.float()).view(1, self.n_heads)
        outputs: list[torch.Tensor] = []
        for t in range(length):
            dt = delta[:, t].float()
            a = torch.exp(dt * A)
            xt = xh[:, t] * dt.unsqueeze(-1)
            state = a[:, :, None, None] * state + xt.unsqueeze(-1) * Bv[:, t].float()[:, :, None, :]
            yt = torch.einsum("bhn,bhpn->bhp", Cv[:, t].float(), state)
            outputs.append(yt)
        return torch.stack(outputs, dim=1).reshape(bsz, length, self.d_inner).to(x.dtype), state


class SelectiveSSMBlock(nn.Module):
    def __init__(self, config: UnifiedGutenbergSSDConfig) -> None:
        super().__init__()
        d_inner = config.d_model * config.expand
        self.norm_ssm = nn.LayerNorm(config.d_model)
        self.in_proj = nn.Linear(config.d_model, 2 * d_inner, bias=False)
        self.local_conv = nn.Conv1d(d_inner, d_inner, kernel_size=config.d_conv, groups=d_inner, bias=True)
        self.param_proj = nn.Linear(d_inner, config.n_heads + 2 * config.n_heads * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.n_heads, config.n_heads, bias=True)
        nn.init.constant_(self.dt_proj.bias, -2.0)
        self.ssm = ChunkedSelectiveSSM(d_inner, config.n_heads, config.d_state, config.chunk_size)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        if config.use_lrn:
            self.lrn = nn.LocalResponseNorm(config.lrn_size, alpha=1e-4, beta=0.75, k=1.0)
        else:
            self.lrn = None
        self.norm_ffn = nn.LayerNorm(config.d_model)
        ffn_hidden = config.d_model * config.ffn_mult
        self.ffn_in = nn.Linear(config.d_model, 2 * ffn_hidden, bias=False)
        self.ffn_out = nn.Linear(ffn_hidden, config.d_model, bias=False)
        residual_std = 0.02 / math.sqrt(2.0 * config.n_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=residual_std)
        nn.init.normal_(self.ffn_out.weight, mean=0.0, std=residual_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.norm_ssm(inputs)
        drive, gate = self.in_proj(hidden).chunk(2, dim=-1)
        # Exact left padding makes the depthwise convolution causal without right-side waste.
        drive = F.pad(drive.transpose(1, 2), (self.local_conv.kernel_size[0] - 1, 0))
        drive = F.silu(self.local_conv(drive).transpose(1, 2))
        selective = self.param_proj(drive)
        dt_raw, Bv, Cv = torch.split(
            selective, [self.ssm.n_heads, self.ssm.n_heads * self.ssm.d_state, self.ssm.n_heads * self.ssm.d_state], dim=-1
        )
        Bv = Bv.reshape(inputs.shape[0], inputs.shape[1], self.ssm.n_heads, self.ssm.d_state)
        Cv = Cv.reshape(inputs.shape[0], inputs.shape[1], self.ssm.n_heads, self.ssm.d_state)
        delta = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=1.0)
        hidden = self.ssm(drive, delta, Bv, Cv)
        hidden = hidden + drive * self.D
        hidden = hidden * F.silu(gate)
        if self.lrn is not None:
            hidden = self.lrn(hidden.transpose(1, 2)).transpose(1, 2).contiguous()
        inputs = residual + self.dropout(self.out_proj(hidden))

        residual = inputs
        hidden = self.norm_ffn(inputs)
        value, gate = self.ffn_in(hidden).chunk(2, dim=-1)
        hidden = self.ffn_out(F.silu(value) * gate)
        return residual + self.dropout(hidden)


class UnifiedGutenbergSSD(nn.Module):
    def __init__(self, config: Optional[UnifiedGutenbergSSDConfig] = None) -> None:
        super().__init__()
        self.config = config or UnifiedGutenbergSSDConfig()
        self.config.validate()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have non-empty shape (batch, sequence).")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor.")
        if input_ids.device.type != "cpu":
            raise ValueError("This prototype currently supports CPU only.")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Sequence length exceeds max_seq_len={self.config.max_seq_len}.")
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        if targets is None:
            return logits
        if targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape.")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1).long(), ignore_index=PAD_ID)
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: str | Path, model: UnifiedGutenbergSSD, metadata: Optional[dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if model.config.tie_embeddings:
        state.pop("lm_head.weight", None)
    payload = {
        "architecture": ARCHITECTURE_NAME,
        "config": asdict(model.config),
        "model_state_dict": state,
        "metadata": metadata or {},
        "parameter_count": model.num_parameters(),
        "tied_embeddings": model.config.tie_embeddings,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)
    return path


def load_checkpoint(path: str | Path) -> UnifiedGutenbergSSD:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(f"Unexpected architecture: {payload.get('architecture')!r}")
    config = UnifiedGutenbergSSDConfig(**payload["config"])
    model = UnifiedGutenbergSSD(config)
    state = payload["model_state_dict"]
    if config.tie_embeddings:
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != {"lm_head.weight"}:
            raise RuntimeError(f"Checkpoint key mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
        if model.lm_head.weight is not model.token_embedding.weight:
            raise RuntimeError("Tied embeddings were not restored.")
    else:
        model.load_state_dict(state, strict=True)
    return model


def run_self_tests(threads: int = 2) -> dict[str, Any]:
    torch.manual_seed(90210)
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    results: dict[str, Any] = {
        "architecture": ARCHITECTURE_NAME,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": "CPU",
        "threads": torch.get_num_threads(),
    }

    # TEST 1 — chunked SSD recurrence must match explicit sequential reference.
    start = time.perf_counter()
    op = ChunkedSelectiveSSM(d_inner=16, n_heads=4, d_state=5, chunk_size=4)
    x = torch.randn(2, 11, 16, requires_grad=True)
    delta = (torch.rand(2, 11, 4) * 0.5 + 0.03).requires_grad_()
    Bv = torch.randn(2, 11, 4, 5, requires_grad=True)
    Cv = torch.randn(2, 11, 4, 5, requires_grad=True)
    chunked, final_state = op(x, delta, Bv, Cv, return_state=True)
    reference, reference_state = op.forward_sequential_reference(x, delta, Bv, Cv)
    max_err = float((chunked - reference).abs().max().detach().item())
    state_err = float((final_state - reference_state).abs().max().detach().item())
    if max_err > 2e-5 or state_err > 2e-5:
        raise AssertionError(f"SSD/reference mismatch: output={max_err}, state={state_err}")
    loss = chunked.square().mean()
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in op.parameters()):
        raise AssertionError("Selective SSM parameter gradients invalid.")
    results["test_1_chunked_vs_sequential_reference"] = "PASS"
    results["chunked_reference_max_abs_error"] = max_err
    results["final_state_max_abs_error"] = state_err
    results["test_1_seconds"] = round(time.perf_counter() - start, 4)

    # TEST 2 — complete LM forward/backward, finite gradients, and strict causality.
    start = time.perf_counter()
    cfg = UnifiedGutenbergSSDConfig(d_model=32, n_layers=2, d_state=8, n_heads=4,
                                    expand=2, d_conv=4, ffn_mult=2, chunk_size=8,
                                    max_seq_len=32, dropout=0.0, use_lrn=False)
    model = UnifiedGutenbergSSD(cfg).cpu()
    ids = torch.randint(0, 256, (2, 24), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 24), dtype=torch.long)
    logits, loss = model(ids, targets)
    if logits.shape != (2, 24, cfg.vocab_size) or not torch.isfinite(loss).item():
        raise AssertionError("Invalid model output or loss.")
    loss.backward()
    missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()]
    if missing or bad:
        raise AssertionError(f"Bad gradients: missing={missing}, nonfinite={bad}")
    model.eval()
    altered = ids[:1].clone()
    altered[:, 12:] = torch.randint(0, 256, altered[:, 12:].shape)
    with torch.no_grad():
        original_prefix = model(ids[:1])[:, :12]
        altered_prefix = model(altered)[:, :12]
    causal_err = float((original_prefix - altered_prefix).abs().max().item())
    if causal_err > 2e-5:
        raise AssertionError(f"Causality check failed: {causal_err}")
    results["test_2_forward_backward_causality"] = "PASS"
    results["parameter_count_smoke_config"] = model.num_parameters()
    results["initial_loss"] = float(loss.detach().item())
    results["causality_max_abs_error"] = causal_err
    results["test_2_seconds"] = round(time.perf_counter() - start, 4)

    # TEST 3 — checkpoint round-trip and tied embeddings.
    start = time.perf_counter()
    with torch.no_grad():
        expected = model(ids)
    with __import__("tempfile").TemporaryDirectory(prefix="unified_gutenberg_ssd_test_") as tmp:
        ckpt = Path(tmp) / "model.pt"
        save_checkpoint(ckpt, model, {"status": "SELF_TEST_ONLY"})
        reloaded = load_checkpoint(ckpt).eval()
        with torch.no_grad():
            actual = reloaded(ids)
        reload_err = float((expected - actual).abs().max().item())
        if reload_err != 0.0:
            raise AssertionError(f"Checkpoint round trip mismatch: {reload_err}")
        if reloaded.lm_head.weight is not reloaded.token_embedding.weight:
            raise AssertionError("Tied embedding alias was not restored.")
        results["checkpoint_size_bytes"] = ckpt.stat().st_size
        results["checkpoint_sha256_test"] = sha256_file(ckpt)
    results["test_3_checkpoint_round_trip"] = "PASS"
    results["checkpoint_max_abs_error"] = reload_err
    results["test_3_seconds"] = round(time.perf_counter() - start, 4)
    results["overall"] = "PASS"
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--print-default-params", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if not args.self_test and not args.print_default_params:
        parser.error("Use --self-test or --print-default-params")
    if args.print_default_params:
        model = UnifiedGutenbergSSD(UnifiedGutenbergSSDConfig())
        print(json.dumps({"config": asdict(model.config), "parameter_count": model.num_parameters()}, indent=2))
    if args.self_test:
        print(json.dumps(run_self_tests(args.threads), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
