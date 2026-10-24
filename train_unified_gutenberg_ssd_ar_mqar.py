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

"""Train and validate Unified Gutenberg v0.2 SSD-inspired on synthetic AR/MQAR.

Safety contract:
- Loads the random base checkpoint and never writes to it.
- Trains one disposable in-memory model on a mixture of AR and MQAR examples.
- Uses separate fixed validation, test, and higher-load test sets.
- Creates a promoted, read-only AR/MQAR checkpoint ONLY if both AR and MQAR meet
  the configured held-out accuracy thresholds and the base hash is unchanged.
- If the gate fails, saves a JSON report but does not create the promoted checkpoint.

This is a synthetic recall capability test, not natural-language pretraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from Unified_Gutenberg_Arch_v0_2_SSD import (
    ARCHITECTURE_NAME,
    BOS_ID,
    DEFAULT_VOCAB_SIZE,
    UnifiedGutenbergSSD,
    UnifiedGutenbergSSDConfig,
    load_checkpoint,
    save_checkpoint,
)

# Synthetic marker tokens; they are valid vocabulary IDs and are not used as keys/values.
PAIR_MARKER = 5
QUERY_MARKER = 6
ANSWER_MARKER = 7
KEY_TOKENS = tuple(range(16, 80))       # 64 possible key tokens
VALUE_TOKENS = tuple(range(96, 160))    # 64 possible value tokens
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "Models"
DEFAULT_BASE = DEFAULT_MODELS_DIR / "unified_gutenberg_ssd_base_v0_2.pt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
PROMOTED_FILENAME = "unified_gutenberg_ssd_ar_mqar_passed_v0_2.pt"

# One example is (model inputs, answer-marker positions, expected next value tokens).
RecallExample = tuple[list[int], list[int], list[int]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp, path)


def make_example(num_pairs: int, num_queries: int, rng: random.Random) -> RecallExample:
    if not 1 <= num_pairs <= min(len(KEY_TOKENS), len(VALUE_TOKENS)):
        raise ValueError("num_pairs exceeds the synthetic key/value pools.")
    if not 1 <= num_queries <= num_pairs:
        raise ValueError("num_queries must be in [1, num_pairs].")

    keys = rng.sample(list(KEY_TOKENS), num_pairs)
    values = rng.sample(list(VALUE_TOKENS), num_pairs)
    mapping = dict(zip(keys, values))
    pairs = list(mapping.items())
    rng.shuffle(pairs)

    query_keys = rng.sample(keys, num_queries)
    rng.shuffle(query_keys)

    tokens = [BOS_ID]
    for key, value in pairs:
        tokens.extend((PAIR_MARKER, key, value))

    answer_positions: list[int] = []
    answer_values: list[int] = []
    for key in query_keys:
        tokens.extend((QUERY_MARKER, key, ANSWER_MARKER))
        answer_positions.append(len(tokens) - 1)
        value = mapping[key]
        answer_values.append(value)
        # Include prior answers in subsequent MQAR queries' context.
        tokens.append(value)

    # Final label is predicted from the final ANSWER_MARKER, not included in input.
    return tokens[:-1], answer_positions, answer_values


def make_fixed_set(num_pairs: int, num_queries: int, count: int, seed: int) -> list[RecallExample]:
    rng = random.Random(seed)
    return [make_example(num_pairs, num_queries, rng) for _ in range(count)]


def make_batch(
    count: int,
    num_pairs: int,
    num_queries: int,
    rng: random.Random,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [make_example(num_pairs, num_queries, rng) for _ in range(count)]
    positions = examples[0][1]
    if any(example[1] != positions for example in examples):
        raise RuntimeError("Training batch has inconsistent answer marker positions.")
    max_len = max(len(example[0]) for example in examples)
    if max_len > max_seq_len:
        raise ValueError(
            f"AR/MQAR prompt length {max_len} exceeds max_seq_len={max_seq_len}; "
            "reduce pair/query counts or use a compatible checkpoint."
        )
    if any(len(example[0]) != max_len for example in examples):
        raise RuntimeError("Fixed task configuration unexpectedly produced varying lengths.")
    input_ids = torch.tensor([example[0] for example in examples], dtype=torch.long)
    target_ids = torch.tensor([example[2] for example in examples], dtype=torch.long)
    position_ids = torch.tensor(positions, dtype=torch.long)
    return input_ids, position_ids, target_ids


def evaluate(
    model: UnifiedGutenbergSSD,
    examples: Sequence[RecallExample],
    batch_size: int,
    num_pairs: int,
    num_queries: int,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    value_ids = torch.tensor(VALUE_TOKENS, dtype=torch.long)
    total = 0
    correct_full = 0
    correct_restricted = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            positions = chunk[0][1]
            if any(example[1] != positions for example in chunk):
                raise RuntimeError("Evaluation batch has inconsistent answer positions.")
            inputs = torch.tensor([e[0] for e in chunk], dtype=torch.long)
            targets = torch.tensor([e[2] for e in chunk], dtype=torch.long)
            pos = torch.tensor(positions, dtype=torch.long)
            logits = model(inputs)[:, pos, :]
            if logits.shape[:2] != targets.shape:
                raise RuntimeError("Logit/target shape mismatch in AR/MQAR evaluator.")
            loss_sum += float(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            ).item())
            pred_full = logits.argmax(dim=-1)
            pred_restricted = value_ids[logits.index_select(-1, value_ids).argmax(dim=-1)]
            correct_full += int((pred_full == targets).sum().item())
            correct_restricted += int((pred_restricted == targets).sum().item())
            total += targets.numel()
    model.train(was_training)
    return {
        "examples": len(examples),
        "queries_scored": total,
        "pairs_per_context": num_pairs,
        "queries_per_context": num_queries,
        "answer_cross_entropy_full_vocab": loss_sum / max(total, 1),
        "exact_match_top1_full_vocab": correct_full / max(total, 1),
        "exact_match_top1_value_candidates_only": correct_restricted / max(total, 1),
        "chance_value_candidates_only": 1.0 / len(VALUE_TOKENS),
        "chance_full_vocab_uniform": 1.0 / DEFAULT_VOCAB_SIZE,
    }


def task_passes(metrics: dict[str, Any], min_full_accuracy: float, min_candidate_accuracy: float) -> bool:
    return (
        metrics["exact_match_top1_full_vocab"] >= min_full_accuracy
        and metrics["exact_match_top1_value_candidates_only"] >= min_candidate_accuracy
    )


def report_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    print(
        f"[{prefix}] full_acc={metrics['exact_match_top1_full_vocab']:.3f} "
        f"candidate_acc={metrics['exact_match_top1_value_candidates_only']:.3f} "
        f"answer_loss={metrics['answer_cross_entropy_full_vocab']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000,
                        help="Maximum mixed AR/MQAR optimizer updates (default: 1000).")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=64,
                        help="Fixed validation examples per task.")
    parser.add_argument("--test-examples", type=int, default=256,
                        help="Separate held-out test examples per task.")
    parser.add_argument("--generalization-examples", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--mqar-queries", type=int, default=3)
    parser.add_argument("--generalization-pairs", type=int, default=12)
    parser.add_argument("--generalization-mqar-queries", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42026)
    parser.add_argument("--min-full-accuracy", type=float, default=0.80)
    parser.add_argument("--min-candidate-accuracy", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("threads", "steps", "eval_every", "batch_size", "eval_examples", "test_examples", "generalization_examples"):
        value = getattr(args, name)
        min_value = 0 if name == "steps" else 1
        if value < min_value:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= {min_value}.")
    for name in ("min_full_accuracy", "min_candidate_accuracy"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.pairs < 2 or args.pairs > min(len(KEY_TOKENS), len(VALUE_TOKENS)):
        raise SystemExit("--pairs must be between 2 and 64.")
    if not 1 <= args.mqar_queries <= args.pairs:
        raise SystemExit("--mqar-queries must be between 1 and --pairs.")
    if args.generalization_pairs < args.pairs or args.generalization_pairs > min(len(KEY_TOKENS), len(VALUE_TOKENS)):
        raise SystemExit("--generalization-pairs must be between --pairs and 64.")
    if not 1 <= args.generalization_mqar_queries <= args.generalization_pairs:
        raise SystemExit("--generalization-mqar-queries must be between 1 and --generalization-pairs.")

    base_path = args.base_checkpoint.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if not base_path.is_file():
        raise SystemExit(f"Base checkpoint not found: {base_path}")
    promoted_path = models_dir / PROMOTED_FILENAME
    if promoted_path.exists():
        raise SystemExit(
            f"Refusing to overwrite an existing promoted checkpoint: {promoted_path}\n"
            "Back it up/rename it before rerunning."
        )
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1 for training; this script is not baseline-only.")

    try:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"unified_gutenberg_ar_mqar_{timestamp}.json"
    base_hash_before = sha256_file(base_path)
    model = load_checkpoint(base_path)
    model.config.validate()
    if model.config.vocab_size != DEFAULT_VOCAB_SIZE:
        raise SystemExit(f"Expected vocab_size={DEFAULT_VOCAB_SIZE}, found {model.config.vocab_size}.")
    if model.config.max_seq_len < 3 * args.generalization_pairs + 4 * args.generalization_mqar_queries + 1:
        # Conservative length check; make_example does the exact input-length check below.
        print("[INFO] Generalization prompt length will be checked exactly against the model context.")

    parameter_count = model.num_parameters()
    run_started = time.perf_counter()
    train_rng = random.Random(args.seed + 1000)
    val_sets = {
        "AR": make_fixed_set(args.pairs, 1, args.eval_examples, args.seed + 1101),
        "MQAR": make_fixed_set(args.pairs, args.mqar_queries, args.eval_examples, args.seed + 1201),
    }
    test_sets = {
        "AR": make_fixed_set(args.pairs, 1, args.test_examples, args.seed + 2101),
        "MQAR": make_fixed_set(args.pairs, args.mqar_queries, args.test_examples, args.seed + 2201),
    }
    generalization_sets = {
        "AR": make_fixed_set(args.generalization_pairs, 1, args.generalization_examples, args.seed + 3101),
        "MQAR": make_fixed_set(
            args.generalization_pairs, args.generalization_mqar_queries,
            args.generalization_examples, args.seed + 3201
        ),
    }
    # Fail before training if requested sequence shapes do not fit the checkpoint.
    for name, examples in {**{f"val_{k}": v for k, v in val_sets.items()},
                           **{f"test_{k}": v for k, v in test_sets.items()},
                           **{f"higher_{k}": v for k, v in generalization_sets.items()}}.items():
        longest = max(len(ex[0]) for ex in examples)
        if longest > model.config.max_seq_len:
            raise SystemExit(
                f"{name} sequence length {longest} exceeds max_seq_len={model.config.max_seq_len}. "
                "Lower pair/query counts or use a base with a larger context."
            )

    report: dict[str, Any] = {
        "architecture": ARCHITECTURE_NAME,
        "report_type": "unified_gutenberg_ar_mqar_train_eval_gate",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "device": "CPU",
        "threads": args.threads,
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256_before": base_hash_before,
        "base_checkpoint_is_source_only": True,
        "parameter_count": parameter_count,
        "training": {
            "method": "joint alternating AR/MQAR synthetic training; no natural language data is read",
            "steps_requested": args.steps,
            "steps_completed": 0,
            "batch_size": args.batch_size,
            "pairs_per_context": args.pairs,
            "mqar_queries_per_context": args.mqar_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "pass_criteria": {
            "test_full_vocab_top1_minimum_for_both_tasks": args.min_full_accuracy,
            "test_value_candidate_top1_minimum_for_both_tasks": args.min_candidate_accuracy,
            "checkpoint_created_only_if_both_tasks_pass": True,
        },
        "baseline_zero_shot_test": {},
        "learning_curve": [],
        "validation_sets": {task: {"examples": len(s)} for task, s in val_sets.items()},
        "test_sets": {task: {"examples": len(s)} for task, s in test_sets.items()},
        "higher_load_sets": {task: {"examples": len(s)} for task, s in generalization_sets.items()},
        "promoted_checkpoint": None,
    }
    atomic_write_json(report_path, report)

    try:
        print(f"[BASE] {base_path}")
        print(f"[INFO] parameters={parameter_count:,}; max_seq_len={model.config.max_seq_len}; no natural-language dataset will be read")
        print(f"[INFO] joint AR/MQAR training updates={args.steps}; batch={args.batch_size}; threads={args.threads}")
        for task, pairs, queries in (("AR", args.pairs, 1), ("MQAR", args.pairs, args.mqar_queries)):
            baseline = evaluate(model, test_sets[task], args.batch_size, pairs, queries)
            report["baseline_zero_shot_test"][task] = baseline
            report_metrics(f"BASELINE {task}", baseline)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        model.train()
        recent_losses: list[float] = []
        total_tokens_seen = 0
        for step in range(1, args.steps + 1):
            task = "AR" if step % 2 == 1 else "MQAR"
            num_queries = 1 if task == "AR" else args.mqar_queries
            inputs, positions, targets = make_batch(
                args.batch_size, args.pairs, num_queries, train_rng, model.config.max_seq_len
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)[:, positions, :]
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss on step {step} ({task}).")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise RuntimeError(f"Non-finite gradient norm on step {step} ({task}).")
            optimizer.step()
            recent_losses.append(float(loss.detach().item()))
            total_tokens_seen += int(inputs.numel())

            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                val_metrics = {
                    "AR": evaluate(model, val_sets["AR"], args.batch_size, args.pairs, 1),
                    "MQAR": evaluate(model, val_sets["MQAR"], args.batch_size, args.pairs, args.mqar_queries),
                }
                model.train()
                record = {
                    "step": step,
                    "mean_train_answer_loss_since_last_eval": sum(recent_losses) / max(len(recent_losses), 1),
                    "last_train_answer_loss": float(loss.detach().item()),
                    "last_grad_norm": float(torch.as_tensor(grad_norm).item()),
                    "tokens_seen_in_inputs": total_tokens_seen,
                    "validation": val_metrics,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
                report["learning_curve"].append(record)
                report["training"]["steps_completed"] = step
                report["training"]["tokens_seen_in_inputs"] = total_tokens_seen
                recent_losses.clear()
                print(
                    f"[STEP {step:4d}/{args.steps}] train_loss={record['mean_train_answer_loss_since_last_eval']:.4f} "
                    f"AR_val={val_metrics['AR']['exact_match_top1_full_vocab']:.3f} "
                    f"MQAR_val={val_metrics['MQAR']['exact_match_top1_full_vocab']:.3f}"
                )
                atomic_write_json(report_path, report)

        model.eval()
        final_test = {
            "AR": evaluate(model, test_sets["AR"], args.batch_size, args.pairs, 1),
            "MQAR": evaluate(model, test_sets["MQAR"], args.batch_size, args.pairs, args.mqar_queries),
        }
        high_load = {
            "AR": evaluate(model, generalization_sets["AR"], args.batch_size,
                           args.generalization_pairs, 1),
            "MQAR": evaluate(model, generalization_sets["MQAR"], args.batch_size,
                             args.generalization_pairs, args.generalization_mqar_queries),
        }
        report["final_held_out_test"] = final_test
        report["higher_load_generalization_test"] = high_load
        for task in ("AR", "MQAR"):
            report_metrics(f"FINAL TEST {task}", final_test[task])
            report_metrics(f"HIGHER LOAD {task}", high_load[task])

        base_hash_after = sha256_file(base_path)
        report["base_checkpoint_sha256_after"] = base_hash_after
        report["base_checkpoint_unchanged"] = base_hash_after == base_hash_before
        passed_tasks = {
            task: task_passes(metrics, args.min_full_accuracy, args.min_candidate_accuracy)
            for task, metrics in final_test.items()
        }
        report["task_gate"] = passed_tasks
        all_pass = all(passed_tasks.values()) and report["base_checkpoint_unchanged"]

        if all_pass:
            if promoted_path.exists():
                raise FileExistsError(f"Promoted checkpoint already exists; refusing to overwrite: {promoted_path}")
            metadata = {
                "artifact_type": "synthetic_ar_mqar_validated_checkpoint",
                "training_status": "synthetic_ar_mqar_trained; no natural_language_training_yet",
                "source_base_checkpoint": str(base_path),
                "source_base_sha256": base_hash_before,
                "source_base_modified": False,
                "training_steps": args.steps,
                "tokens_seen_in_inputs": total_tokens_seen,
                "training_seed": args.seed,
                "optimizer": "AdamW",
                "learning_rate": args.learning_rate,
                "task_training": {"AR": "alternating synthetic key-value recall", "MQAR": "alternating multi-query key-value recall"},
                "held_out_test_metrics": final_test,
                "higher_load_metrics": high_load,
                "pass_thresholds": report["pass_criteria"],
                "checkpoint_role": "read_only_source_for_future_language_training; language training must save elsewhere",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            models_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(promoted_path, model, metadata=metadata)
            # Verify saved file reconstructs same model outputs before marking it read-only.
            reloaded = load_checkpoint(promoted_path).eval()
            probe_ids = torch.tensor([test_sets["AR"][0][0]], dtype=torch.long)
            with torch.inference_mode():
                original_logits = model.eval()(probe_ids)
                reload_logits = reloaded(probe_ids)
            roundtrip_error = float((original_logits - reload_logits).abs().max().item())
            if not torch.equal(original_logits, reload_logits):
                promoted_path.unlink(missing_ok=True)
                raise RuntimeError(f"Promoted checkpoint round-trip failed (max_abs_error={roundtrip_error}).")
            # On Windows, this sets the read-only file attribute. Future training must write a new file.
            os.chmod(promoted_path, stat.S_IREAD)
            report["promoted_checkpoint"] = {
                "path": str(promoted_path),
                "sha256": sha256_file(promoted_path),
                "size_bytes": promoted_path.stat().st_size,
                "read_only_attribute_set": True,
                "round_trip_max_abs_error": roundtrip_error,
                "metadata_training_status": metadata["training_status"],
            }
            report["status"] = "PASS_CHECKPOINT_CREATED_READ_ONLY"
            print(f"[PASS] AR and MQAR met held-out gates. Read-only checkpoint: {promoted_path}")
        else:
            report["status"] = "FAIL_NO_PROMOTED_CHECKPOINT"
            report["checkpoint_not_created_reason"] = {
                "task_gate": passed_tasks,
                "base_checkpoint_unchanged": report["base_checkpoint_unchanged"],
                "message": "At least one held-out task missed the pass threshold, or the base hash changed. No promoted checkpoint was created.",
            }
            print("[FAIL] Gate not met. No promoted AR/MQAR checkpoint was created.")

        report["elapsed_seconds_total"] = time.perf_counter() - run_started
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(report_path, report)
        print(f"[REPORT] {report_path}")
        print(json.dumps({
            "status": report["status"],
            "base_checkpoint_unchanged": report["base_checkpoint_unchanged"],
            "test_gate": report["task_gate"],
            "promoted_checkpoint": report["promoted_checkpoint"],
            "report": str(report_path),
        }, indent=2))
        return 0 if all_pass else 2
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED_NO_PROMOTED_CHECKPOINT"
        report["error"] = "KeyboardInterrupt: run stopped before passing the held-out AR/MQAR gate."
        report["elapsed_seconds_total"] = time.perf_counter() - run_started
        try:
            atomic_write_json(report_path, report)
        except Exception:
            pass
        print("[INTERRUPTED] No promoted checkpoint was created.", file=sys.stderr)
        print(f"[REPORT] {report_path}", file=sys.stderr)
        return 130
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["elapsed_seconds_total"] = time.perf_counter() - run_started
        try:
            atomic_write_json(report_path, report)
        except Exception:
            pass
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"[REPORT] {report_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
