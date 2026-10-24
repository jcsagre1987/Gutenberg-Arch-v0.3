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

"""Final release-gate test for Unified Gutenberg Arch v0.3 (CPU-only).

The script reproduces the validated v0.3 memory-enabled training recipe on 16-pair
synthetic AR/MQAR tasks, evaluates held-out and mapping-swapped examples, and creates a
separate release-candidate checkpoint ONLY if the task thresholds, source-hash check,
RAM ceiling, save/reload parity, and byte-level text inference smoke test all pass.

Important scope statement: the model is trained here on synthetic associative recall,
not natural-language next-token prediction. The text test verifies byte-tokenizer and
CPU inference plumbing; it does NOT establish language understanding or generation quality.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import Unified_Gutenberg_Arch_v0_3_Associative_SSD as v3
from Unified_Gutenberg_Arch_v0_2_SSD import GutenbergByteTokenizer
from train_unified_gutenberg_ssd_ar_mqar import (
    KEY_TOKENS,
    VALUE_TOKENS,
    make_batch,
    make_fixed_set,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "Models"
DEFAULT_V3_BASE = DEFAULT_MODELS_DIR / "unified_gutenberg_ssd_base_v0_3.pt"
EXPECTED_V3_BASE_SHA256 = "6668389cc323f3f913054c3573c9ce1b381a241eadb27ed5b1047e5ebe5eb1c1"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
TEXT_SMOKE = "Gutenberg Arch v0.3 CPU release smoke test."
VERIFY_PREFIX = "VERIFY_RESULT_JSON="


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def peak_rss_bytes() -> int | None:
    """Return process peak resident/working-set bytes using platform-native APIs."""
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("Psapi.dll")
            kernel32 = ctypes.WinDLL("Kernel32.dll")
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            ok = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.PeakWorkingSetSize)
        except Exception:
            return None
        return None
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if platform.system() == "Darwin" else value * 1024
    except Exception:
        return None


def setup_cpu(threads: int) -> None:
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def forward_logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(ids, disable_associative_memory=False)


def candidate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    value_ids = torch.tensor(VALUE_TOKENS, dtype=torch.long, device=logits.device)
    target_indices = targets - int(VALUE_TOKENS[0])
    if target_indices.numel() and (
        int(target_indices.min().item()) < 0 or int(target_indices.max().item()) >= len(VALUE_TOKENS)
    ):
        raise ValueError("Training target is outside the legal synthetic value-token set.")
    candidate_logits = logits.index_select(-1, value_ids)
    return F.cross_entropy(
        candidate_logits.reshape(-1, len(VALUE_TOKENS)), target_indices.reshape(-1)
    )


def evaluate(model: torch.nn.Module, examples: list, batch_size: int, pairs: int) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    value_ids = torch.tensor(VALUE_TOKENS, dtype=torch.long)
    total = full_correct = candidate_correct = 0
    full_loss_sum = candidate_loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            positions = chunk[0][1]
            if any(item[1] != positions for item in chunk):
                raise RuntimeError("Evaluation batch has inconsistent answer positions.")
            inputs = torch.tensor([item[0] for item in chunk], dtype=torch.long)
            targets = torch.tensor([item[2] for item in chunk], dtype=torch.long)
            logits = forward_logits(model, inputs)[:, positions, :]
            if logits.shape[:2] != targets.shape:
                raise RuntimeError(f"Logit/target shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
            full_loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            ).item())
            candidate_logits = logits.index_select(-1, value_ids)
            candidate_targets = targets - int(VALUE_TOKENS[0])
            candidate_loss_sum += float(F.cross_entropy(
                candidate_logits.reshape(-1, len(VALUE_TOKENS)), candidate_targets.reshape(-1), reduction="sum"
            ).item())
            full_correct += int((logits.argmax(-1) == targets).sum().item())
            predictions = value_ids[candidate_logits.argmax(-1)]
            candidate_correct += int((predictions == targets).sum().item())
            total += int(targets.numel())
    model.train(was_training)
    return {
        "examples": len(examples),
        "queries_scored": total,
        "pairs_per_context": pairs,
        "full_vocab_top1_accuracy": full_correct / max(total, 1),
        "candidate_top1_accuracy": candidate_correct / max(total, 1),
        "full_vocab_answer_cross_entropy": full_loss_sum / max(total, 1),
        "candidate_answer_cross_entropy": candidate_loss_sum / max(total, 1),
        "candidate_uniform_chance": 1.0 / len(VALUE_TOKENS),
    }


def make_counterfactual_set(examples: list, num_pairs: int) -> list[tuple[list[int], list[int], int, int, int]]:
    variants = []
    for ids, answer_positions, targets in examples:
        original = list(ids)
        swapped = list(ids)
        if not answer_positions or not targets:
            raise ValueError("Counterfactual requires at least one query.")
        query_pos = int(answer_positions[0])
        query_key = int(original[query_pos - 1])
        key_positions = [2 + 3 * i for i in range(num_pairs)]
        value_positions = [3 + 3 * i for i in range(num_pairs)]
        mapping = {int(original[kp]): int(original[vp]) for kp, vp in zip(key_positions, value_positions)}
        if query_key not in mapping:
            raise RuntimeError(f"Query key {query_key} is missing from the context mapping.")
        other_key = next(key for key in mapping if key != query_key)
        mapping[query_key], mapping[other_key] = mapping[other_key], mapping[query_key]
        for kp, vp in zip(key_positions, value_positions):
            swapped[vp] = mapping[int(swapped[kp])]
        new_target = int(mapping[query_key])
        if query_pos + 1 < len(swapped):
            swapped[query_pos + 1] = new_target
        variants.append((original, swapped, query_pos, int(targets[0]), new_target))
    return variants


def evaluate_counterfactual(model: torch.nn.Module, variants: list, batch_size: int) -> dict[str, Any]:
    value_ids = torch.tensor(VALUE_TOKENS, dtype=torch.long)
    original_correct = swapped_correct = joint_correct = 0
    total = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(variants), batch_size):
                chunk = variants[start:start + batch_size]
                positions = {item[2] for item in chunk}
                if len(positions) != 1:
                    raise RuntimeError("Counterfactual batch has inconsistent query positions.")
                pos = next(iter(positions))
                original_ids = torch.tensor([item[0] for item in chunk], dtype=torch.long)
                swapped_ids = torch.tensor([item[1] for item in chunk], dtype=torch.long)
                old_targets = torch.tensor([item[3] for item in chunk], dtype=torch.long)
                new_targets = torch.tensor([item[4] for item in chunk], dtype=torch.long)
                original_logits = forward_logits(model, original_ids)[:, pos, :].index_select(-1, value_ids)
                swapped_logits = forward_logits(model, swapped_ids)[:, pos, :].index_select(-1, value_ids)
                original_predictions = value_ids[original_logits.argmax(-1)]
                swapped_predictions = value_ids[swapped_logits.argmax(-1)]
                original_correct += int((original_predictions == old_targets).sum().item())
                swapped_correct += int((swapped_predictions == new_targets).sum().item())
                joint_correct += int(((original_predictions == old_targets) & (swapped_predictions == new_targets)).sum().item())
                total += len(chunk)
    finally:
        model.train(was_training)
    return {
        "examples": total,
        "original_first_query_accuracy": original_correct / max(total, 1),
        "swapped_first_query_accuracy": swapped_correct / max(total, 1),
        "joint_correct_to_swapped_answer_rate": joint_correct / max(total, 1),
    }


def position_baselines(examples: list, pairs: int) -> dict[str, float]:
    total = first_correct = last_correct = 0
    for ids, _positions, targets in examples:
        first_value = int(ids[3])
        last_value = int(ids[3 + 3 * (pairs - 1)])
        for target in targets:
            first_correct += int(first_value == int(target))
            last_correct += int(last_value == int(target))
            total += 1
    return {
        "first_context_value_accuracy": first_correct / max(total, 1),
        "last_context_value_accuracy": last_correct / max(total, 1),
    }


def create_heldout(args: argparse.Namespace) -> tuple[dict[str, list], dict[str, list]]:
    datasets = {
        "AR": make_fixed_set(args.pairs, 1, args.eval_examples, args.seed + 2101),
        "MQAR": make_fixed_set(args.pairs, args.mqar_queries, args.eval_examples, args.seed + 2201),
    }
    counterfactuals = {task: make_counterfactual_set(items, args.pairs) for task, items in datasets.items()}
    return datasets, counterfactuals


def train_and_score(args: argparse.Namespace, base_path: Path) -> tuple[v3.UnifiedGutenbergSSDv3, dict[str, Any], dict[str, list], dict[str, list]]:
    train_rng = random.Random(args.seed + 9001)
    batches = []
    for step in range(1, args.steps + 1):
        task = "AR" if step % 2 == 1 else "MQAR"
        query_count = 1 if task == "AR" else args.mqar_queries
        batch = make_batch(args.batch_size, args.pairs, query_count, train_rng, args.model_max_seq_len)
        batches.append((task, *batch))

    heldout, counterfactuals = create_heldout(args)
    model = v3.load_v3_checkpoint(base_path).cpu()
    model.train()
    # Match the validated "current" optimizer recipe from the prior comparison.
    torch.manual_seed(args.seed + 100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    learning_curve = []
    rolling_losses: list[float] = []
    start = time.perf_counter()
    for step, (task, inputs, positions, targets) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = forward_logits(model, inputs)[:, positions, :]
        loss = candidate_loss(logits, targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at training step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
            raise RuntimeError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        rolling_losses.append(float(loss.detach().item()))
        if step % args.eval_every == 0 or step == args.steps:
            ar_metrics = evaluate(model, heldout["AR"], args.batch_size, args.pairs)
            mqar_metrics = evaluate(model, heldout["MQAR"], args.batch_size, args.pairs)
            ar_cf = evaluate_counterfactual(model, counterfactuals["AR"], args.batch_size)
            mqar_cf = evaluate_counterfactual(model, counterfactuals["MQAR"], args.batch_size)
            record = {
                "step": step,
                "recent_mean_candidate_loss": sum(rolling_losses[-args.eval_every:]) / min(len(rolling_losses), args.eval_every),
                "AR": ar_metrics,
                "MQAR": mqar_metrics,
                "AR_counterfactual": ar_cf,
                "MQAR_counterfactual": mqar_cf,
                "elapsed_seconds": time.perf_counter() - start,
            }
            learning_curve.append(record)
            print(
                f"[STEP {step}/{args.steps}] train_loss={record['recent_mean_candidate_loss']:.4f} "
                f"AR={ar_metrics['candidate_top1_accuracy']:.3f} "
                f"MQAR={mqar_metrics['candidate_top1_accuracy']:.3f} "
                f"AR_CF={ar_cf['joint_correct_to_swapped_answer_rate']:.3f} "
                f"MQAR_CF={mqar_cf['joint_correct_to_swapped_answer_rate']:.3f}",
                flush=True,
            )
    # Final metrics are computed once more, independently of the evaluation cadence.
    final = {
        "AR": evaluate(model, heldout["AR"], args.batch_size, args.pairs),
        "MQAR": evaluate(model, heldout["MQAR"], args.batch_size, args.pairs),
        "AR_counterfactual": evaluate_counterfactual(model, counterfactuals["AR"], args.batch_size),
        "MQAR_counterfactual": evaluate_counterfactual(model, counterfactuals["MQAR"], args.batch_size),
    }
    final["position_baselines"] = {
        task: position_baselines(heldout[task], args.pairs) for task in ("AR", "MQAR")
    }
    final["learning_curve"] = learning_curve
    final["steps_completed"] = args.steps
    final["elapsed_seconds"] = time.perf_counter() - start
    return model.eval(), final, heldout, counterfactuals


def score_gate(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, bool]:
    return {
        "AR_candidate_accuracy": metrics["AR"]["candidate_top1_accuracy"] >= args.min_candidate_accuracy,
        "AR_full_vocab_accuracy": metrics["AR"]["full_vocab_top1_accuracy"] >= args.min_full_accuracy,
        "AR_counterfactual_joint": metrics["AR_counterfactual"]["joint_correct_to_swapped_answer_rate"] >= args.min_counterfactual_accuracy,
        "MQAR_candidate_accuracy": metrics["MQAR"]["candidate_top1_accuracy"] >= args.min_candidate_accuracy,
        "MQAR_full_vocab_accuracy": metrics["MQAR"]["full_vocab_top1_accuracy"] >= args.min_full_accuracy,
        "MQAR_counterfactual_joint": metrics["MQAR_counterfactual"]["joint_correct_to_swapped_answer_rate"] >= args.min_counterfactual_accuracy,
    }


def save_release_checkpoint(path: Path, model: v3.UnifiedGutenbergSSDv3, metadata: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing release checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if model.config.tie_embeddings:
        state.pop("lm_head.weight", None)
    payload = {
        "architecture": v3.ARCHITECTURE_NAME,
        "parent_architecture": v3.PARENT_ARCHITECTURE_NAME,
        "config": asdict(model.config),
        "model_state_dict": state,
        "metadata": metadata,
        "parameter_count": model.num_parameters(),
        "tied_embeddings": model.config.tie_embeddings,
        "checkpoint_role": "experimental_open_weight_release_candidate_validated_on_synthetic_AR_MQAR_only",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_verify_process(args: argparse.Namespace) -> int:
    checkpoint = args.verify_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    setup_cpu(args.threads)
    process_start_peak = peak_rss_bytes()
    file_hash = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference_logits_sha256 = payload.get("metadata", {}).get("reference_probe_logits_sha256")
    model = v3.load_v3_checkpoint(checkpoint).cpu().eval()
    if model.config.tie_embeddings and model.lm_head.weight is not model.token_embedding.weight:
        raise RuntimeError("Tied embeddings did not survive fresh-process reload.")
    heldout, counterfactuals = create_heldout(args)
    probe_ids = torch.tensor([heldout["AR"][0][0]], dtype=torch.long)
    with torch.inference_mode():
        probe_logits = model(probe_ids)
    reloaded_probe_logits_sha256 = tensor_sha256(probe_logits)
    if not reference_logits_sha256:
        raise RuntimeError("Checkpoint metadata does not contain the pre-save probe-logit fingerprint.")
    if reloaded_probe_logits_sha256 != reference_logits_sha256:
        raise RuntimeError("Fresh-process logits differ from pre-save logits; checkpoint round-trip is not exact.")
    metrics = {
        "AR": evaluate(model, heldout["AR"], args.batch_size, args.pairs),
        "MQAR": evaluate(model, heldout["MQAR"], args.batch_size, args.pairs),
        "AR_counterfactual": evaluate_counterfactual(model, counterfactuals["AR"], args.batch_size),
        "MQAR_counterfactual": evaluate_counterfactual(model, counterfactuals["MQAR"], args.batch_size),
    }
    tokenizer = GutenbergByteTokenizer()
    encoded = tokenizer.encode(TEXT_SMOKE)
    if tokenizer.decode(encoded) != TEXT_SMOKE:
        raise RuntimeError("UTF-8 byte tokenizer encode/decode round trip failed.")
    if len(encoded) > model.config.max_seq_len:
        raise RuntimeError(
            f"Text smoke sequence has {len(encoded)} tokens, exceeding max_seq_len={model.config.max_seq_len}."
        )
    input_ids = torch.tensor([encoded], dtype=torch.long)
    with torch.inference_mode():
        text_logits = model(input_ids)
    if tuple(text_logits.shape) != (1, len(encoded), model.config.vocab_size):
        raise RuntimeError(f"Unexpected text-smoke logits shape: {tuple(text_logits.shape)}")
    if not torch.isfinite(text_logits).all().item():
        raise RuntimeError("Non-finite logits in byte-level text inference smoke test.")
    peak = peak_rss_bytes()
    if peak is None:
        raise RuntimeError("Could not measure process peak RSS/working set; release memory gate cannot be verified.")
    result = {
        "status": "PASS_FRESH_PROCESS_ARTIFACT_VERIFICATION",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "architecture": v3.ARCHITECTURE_NAME,
        "parameter_count": model.num_parameters(),
        "max_seq_len": model.config.max_seq_len,
        "tied_embeddings_restored": (not model.config.tie_embeddings) or (model.lm_head.weight is model.token_embedding.weight),
        "pre_save_logits_sha256": reference_logits_sha256,
        "fresh_process_logits_sha256": reloaded_probe_logits_sha256,
        "save_reload_exact_logits_match": True,
        "heldout_and_counterfactual_metrics": metrics,
        "byte_text_smoke": {
            "status": "PASS",
            "text": TEXT_SMOKE,
            "utf8_round_trip": True,
            "input_tokens": len(encoded),
            "logits_shape": list(text_logits.shape),
            "finite_logits": True,
            "scope_note": "This verifies byte-tokenizer and inference plumbing only; checkpoint has not been trained on natural-language text.",
        },
        "peak_rss_or_working_set_bytes": peak,
        "memory_measurement_at_start_bytes": process_start_peak,
    }
    print(VERIFY_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_V3_BASE)
    parser.add_argument("--expected-base-sha256", default=EXPECTED_V3_BASE_SHA256,
                        help="Expected immutable v0.3 base SHA-256; override only when intentionally validating another base.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="Folder for the release-candidate checkpoint (default: Models beside this script).")
    parser.add_argument("--output-checkpoint", type=Path, default=None,
                        help="Optional explicit destination for the trained release candidate; existing files are never overwritten.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--mqar-queries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42030)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-candidate-accuracy", type=float, default=0.85)
    parser.add_argument("--min-full-accuracy", type=float, default=0.85)
    parser.add_argument("--min-counterfactual-accuracy", type=float, default=0.80)
    parser.add_argument("--max-ram-gb", type=float, default=4.0)
    parser.add_argument("--verify-checkpoint", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true", help="Check portable imports, architecture correctness, tokenizer round-trip, and synthetic batch generation without a checkpoint or training run.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.threads < 1 or args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or args.eval_examples < 16:
        raise SystemExit("threads/steps/eval-every/batch-size must be positive and eval-examples must be >= 16.")
    if not 1 <= args.mqar_queries <= args.pairs <= min(len(KEY_TOKENS), len(VALUE_TOKENS)):
        raise SystemExit("Require 1 <= mqar-queries <= pairs <= 64.")
    for key in ("learning_rate", "max_ram_gb"):
        if getattr(args, key) <= 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be > 0.")
    for key in ("weight_decay", "grad_clip"):
        if getattr(args, key) < 0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0.")
    if args.self_test:
        setup_cpu(args.threads)
        architecture_report = v3.run_self_tests(args.threads)
        tokenizer = GutenbergByteTokenizer()
        sample_text = "Gutenberg Arch portable smoke test — CPU-first."
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        if decoded != sample_text:
            raise RuntimeError(f"Byte tokenizer round-trip mismatch: {decoded!r}")
        fixed = make_fixed_set(4, 2, 4, args.seed)
        batch_ids, answer_positions, batch_targets = make_batch(2, 4, 2, random.Random(args.seed + 1), 128)
        if batch_ids.ndim != 2 or batch_targets.shape != (2, 2) or answer_positions.numel() != 2:
            raise RuntimeError("Synthetic AR/MQAR helper smoke test returned unexpected shapes.")
        if len(fixed) != 4:
            raise RuntimeError("Synthetic held-out set smoke test returned unexpected size.")
        print(json.dumps({
            "status": "PASS_PORTABLE_IMPORT_AND_SMOKE_TEST",
            "architecture_self_tests": architecture_report.get("overall", "UNKNOWN"),
            "tokenizer_round_trip": "PASS",
            "synthetic_batch_generation": "PASS",
            "synthetic_fixed_set_generation": "PASS",
            "checkpoint_required": False,
            "training_performed": False,
            "checkpoint_created": False,
            "models_directory": str(DEFAULT_MODELS_DIR.resolve()),
            "checkpoint_note": "Expected behavior: --self-test does not train or save a checkpoint. A full release-gate run writes a candidate into the Models folder only after all validation gates pass.",
            "scope_note": "This smoke test validates imports and code plumbing only; it does not validate general-purpose language capability.",
        }, indent=2), flush=True)
        return 0

    if args.verify_checkpoint is not None:
        return run_verify_process(args)

    base_path = args.base_checkpoint.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()

    # Resolve all default output locations relative to the script, not the shell's CWD.
    # Create the immediate Models folder during a real gate run so the expected location
    # exists even if validation stops early because the required base checkpoint is absent.
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_checkpoint.expanduser().resolve() if args.output_checkpoint else (
        models_dir / f"unified_gutenberg_arch_v0_3_release_candidate_{timestamp}.pt"
    )
    report_path = report_dir / f"unified_gutenberg_v0_3_final_release_gate_{timestamp}.json"
    print(f"[MODEL DIR] {models_dir}", flush=True)
    print(f"[BASE CHECKPOINT] {base_path}", flush=True)
    print(f"[OUTPUT CHECKPOINT DESTINATION] {output_path}", flush=True)
    if not base_path.is_file():
        raise SystemExit(
            f"v0.3 base checkpoint not found: {base_path}\n"
            f"Place the correct base checkpoint in {models_dir} or pass --base-checkpoint with its full path.\n"
            "The portable smoke tests do not create model checkpoints; no training was run."
        )
    if output_path == base_path:
        raise SystemExit("Output checkpoint must differ from the read-only base checkpoint.")
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output checkpoint: {output_path}")
    setup_cpu(args.threads)
    base_hash_before = sha256_file(base_path)
    if args.expected_base_sha256 and base_hash_before.lower() != args.expected_base_sha256.strip().lower():
        raise SystemExit(
            "Base checkpoint SHA-256 does not match the pinned v0.3 base. "
            f"Expected={args.expected_base_sha256}; actual={base_hash_before}. "
            "Do not proceed until you confirm the source checkpoint."
        )
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    if base_payload.get("architecture") != v3.ARCHITECTURE_NAME:
        raise SystemExit(f"Base checkpoint architecture mismatch: {base_payload.get('architecture')!r}")
    sequence_len = 1 + 3 * args.pairs + 4 * args.mqar_queries - 1
    args.model_max_seq_len = int(base_payload["config"]["max_seq_len"])
    if sequence_len > args.model_max_seq_len:
        raise SystemExit(
            f"Requested task needs sequence length about {sequence_len}, exceeding max_seq_len={args.model_max_seq_len}."
        )
    if peak_rss_bytes() is None:
        raise SystemExit("Cannot measure peak process memory on this platform; refusing an unverifiable RAM-gated run.")

    report: dict[str, Any] = {
        "experiment": "Unified Gutenberg Arch v0.3 final release gate",
        "status": "RUNNING",
        "scope": "CPU-only synthetic AR/MQAR training plus fresh-process byte-text inference smoke test",
        "natural_language_capability_claim": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_base_checkpoint": str(base_path),
        "source_base_sha256_before": base_hash_before,
        "expected_base_sha256": args.expected_base_sha256,
        "output_checkpoint": str(output_path),
        "settings": {
            "steps": args.steps, "eval_every": args.eval_every, "batch_size": args.batch_size,
            "eval_examples_per_task": args.eval_examples, "pairs": args.pairs,
            "mqar_queries": args.mqar_queries, "seed": args.seed,
            "optimizer": "AdamW", "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "gradient_clip_norm": args.grad_clip,
            "objective": "candidate-only cross-entropy over 64 legal synthetic value tokens",
            "memory_limit_bytes": int(args.max_ram_gb * (1024 ** 3)),
            "thresholds": {
                "candidate_accuracy_each_task": args.min_candidate_accuracy,
                "full_vocab_accuracy_each_task": args.min_full_accuracy,
                "counterfactual_joint_accuracy_each_task": args.min_counterfactual_accuracy,
            },
        },
        "base_checkpoint_immutable": True,
        "checkpoint_written": False,
    }
    atomic_write_json(report_path, report)
    checkpoint_created = False
    start = time.perf_counter()
    print(f"[BASE] {base_path}", flush=True)
    print(f"[OUTPUT] {output_path}", flush=True)
    print(f"[INFO] steps={args.steps}, pairs={args.pairs}, MQAR queries={args.mqar_queries}, threads={torch.get_num_threads()}", flush=True)
    try:
        model, metrics, _heldout, _counterfactuals = train_and_score(args, base_path)
        report["training_and_heldout_metrics"] = metrics
        report["task_gate"] = score_gate(metrics, args)
        report["task_gate_passed"] = all(report["task_gate"].values())
        peak_before_save = peak_rss_bytes()
        report["training_process_peak_rss_or_working_set_bytes"] = peak_before_save
        memory_pass = peak_before_save is not None and peak_before_save <= int(args.max_ram_gb * (1024 ** 3))
        report["training_process_memory_gate_passed"] = memory_pass
        base_hash_after_training = sha256_file(base_path)
        report["source_base_sha256_after_training"] = base_hash_after_training
        report["base_checkpoint_unchanged"] = base_hash_after_training == base_hash_before

        if not report["task_gate_passed"]:
            raise RuntimeError(f"Held-out AR/MQAR task gate failed: {report['task_gate']}")
        if not memory_pass:
            raise RuntimeError(f"Training process exceeded the {args.max_ram_gb:.2f} GiB peak RAM ceiling or memory measurement failed.")
        if not report["base_checkpoint_unchanged"]:
            raise RuntimeError("Source base checkpoint SHA-256 changed during training.")

        probe_ids = torch.tensor([_heldout["AR"][0][0]], dtype=torch.long)
        with torch.inference_mode():
            reference_probe_logits = model(probe_ids)
        reference_probe_logits_sha256 = tensor_sha256(reference_probe_logits)
        metadata = {
            "reference_probe_logits_sha256": reference_probe_logits_sha256,
            "release_candidate_status": "experimental_open_weight_candidate",
            "training_status": "trained_and_heldout_validated_on_synthetic_AR_MQAR_only",
            "natural_language_pretraining_performed": False,
            "source_base_checkpoint": str(base_path),
            "source_base_sha256": base_hash_before,
            "source_base_modified": False,
            "training": report["settings"],
            "heldout_metrics": {k: metrics[k] for k in ("AR", "MQAR", "AR_counterfactual", "MQAR_counterfactual", "position_baselines")},
            "task_gate": report["task_gate"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_release_checkpoint(output_path, model, metadata)
        checkpoint_created = True
        report["checkpoint_written"] = True
        report["checkpoint_size_bytes"] = output_path.stat().st_size
        report["checkpoint_sha256"] = sha256_file(output_path)
        atomic_write_json(report_path, report)

        # This is intentionally a separate Python process, not just a second load in RAM.
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--verify-checkpoint", str(output_path),
            "--threads", str(args.threads), "--batch-size", str(args.batch_size),
            "--eval-examples", str(args.eval_examples), "--pairs", str(args.pairs),
            "--mqar-queries", str(args.mqar_queries), "--seed", str(args.seed),
        ]
        child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        report["fresh_process_return_code"] = child.returncode
        if child.returncode != 0:
            report["fresh_process_stdout"] = child.stdout[-12000:]
            report["fresh_process_stderr"] = child.stderr[-12000:]
            raise RuntimeError(f"Fresh-process checkpoint verification failed (exit={child.returncode}).")
        marker_lines = [line for line in child.stdout.splitlines() if line.startswith(VERIFY_PREFIX)]
        if not marker_lines:
            report["fresh_process_stdout"] = child.stdout[-12000:]
            report["fresh_process_stderr"] = child.stderr[-12000:]
            raise RuntimeError("Fresh verification process returned no structured result.")
        fresh = json.loads(marker_lines[-1][len(VERIFY_PREFIX):])
        report["fresh_process_verification"] = fresh
        if fresh.get("status") != "PASS_FRESH_PROCESS_ARTIFACT_VERIFICATION":
            raise RuntimeError("Fresh-process report status was not PASS.")
        if fresh.get("checkpoint_sha256") != report["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint SHA-256 changed between parent and fresh-process verification.")
        if fresh.get("save_reload_exact_logits_match") is not True:
            raise RuntimeError("Exact pre-save versus fresh-process inference comparison failed.")
        if fresh["peak_rss_or_working_set_bytes"] > int(args.max_ram_gb * (1024 ** 3)):
            raise RuntimeError("Fresh-process inference exceeded the configured RAM ceiling.")
        fresh_metrics = fresh["heldout_and_counterfactual_metrics"]
        fresh_gate_metrics = {
            "AR": fresh_metrics["AR"], "MQAR": fresh_metrics["MQAR"],
            "AR_counterfactual": fresh_metrics["AR_counterfactual"],
            "MQAR_counterfactual": fresh_metrics["MQAR_counterfactual"],
        }
        fresh_gate = score_gate(fresh_gate_metrics, args)
        report["fresh_process_task_gate"] = fresh_gate
        if not all(fresh_gate.values()):
            raise RuntimeError(f"Fresh-process held-out task gate failed: {fresh_gate}")
        if fresh["byte_text_smoke"]["status"] != "PASS":
            raise RuntimeError("Byte-level text inference smoke test failed.")

        base_hash_final = sha256_file(base_path)
        report["source_base_sha256_final"] = base_hash_final
        report["base_checkpoint_unchanged"] = base_hash_final == base_hash_before
        if not report["base_checkpoint_unchanged"]:
            raise RuntimeError("Source base checkpoint SHA-256 changed during release validation.")

        report["status"] = "PASS_EXPERIMENTAL_OPEN_WEIGHT_RELEASE_CANDIDATE"
        report["release_decision"] = {
            "pass": True,
            "artifact_is_trained": True,
            "fresh_process_reload_passed": True,
            "exact_save_reload_logits_match": True,
            "heldout_AR_MQAR_passed": True,
            "counterfactual_passed": True,
            "byte_text_inference_smoke_passed": True,
            "base_checkpoint_unchanged": True,
            "ram_limit_gib": args.max_ram_gb,
            "warning": "This validates synthetic associative recall and byte-level inference plumbing only; it does not demonstrate natural-language training or general-purpose language capability.",
        }
        report["elapsed_seconds_total"] = time.perf_counter() - start
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(report_path, report)
        print("\n[FINAL RELEASE GATE] PASS_EXPERIMENTAL_OPEN_WEIGHT_RELEASE_CANDIDATE", flush=True)
        print(f"[CHECKPOINT] {output_path}", flush=True)
        print(f"[SHA256] {report['checkpoint_sha256']}", flush=True)
        print(f"[REPORT] {report_path}", flush=True)
        print(json.dumps({
            "status": report["status"], "checkpoint": str(output_path),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "base_checkpoint_unchanged": report["base_checkpoint_unchanged"],
            "task_gate": report["task_gate"],
            "fresh_process_task_gate": report["fresh_process_task_gate"],
            "training_peak_ram_bytes": report["training_process_peak_rss_or_working_set_bytes"],
            "fresh_process_peak_ram_bytes": fresh["peak_rss_or_working_set_bytes"],
            "report": str(report_path),
            "scope_warning": report["release_decision"]["warning"],
        }, indent=2), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "FAIL_RELEASE_GATE"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds_total"] = time.perf_counter() - start
        if checkpoint_created:
            try:
                output_path.unlink(missing_ok=True)
                report["failed_checkpoint_removed"] = True
                report["checkpoint_written"] = False
            except Exception as remove_exc:
                report["failed_checkpoint_removed"] = False
                report["checkpoint_removal_error"] = f"{type(remove_exc).__name__}: {remove_exc}"
        try:
            report["base_checkpoint_unchanged"] = sha256_file(base_path) == base_hash_before
        except Exception:
            report["base_checkpoint_unchanged"] = False
        atomic_write_json(report_path, report)
        print(f"[FINAL RELEASE GATE] FAIL_RELEASE_GATE: {report['error']}", file=sys.stderr, flush=True)
        print(f"[REPORT] {report_path}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
