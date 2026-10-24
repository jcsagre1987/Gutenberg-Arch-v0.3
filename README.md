# Gutenberg Arch v0.3 — Portable CPU Release

Copyright 2026 Juan Carlo Sagre. Licensed under Apache-2.0; see `LICENSE`.

## Preserved implementation

- v0.2 SSD-inspired selective SSM backbone and byte tokenizer.
- v0.3 causal delta-rule associative-memory extension.
- Existing v0.2-to-v0.3 tensor transfer, tied embeddings, self-tests, synthetic AR/MQAR evaluation, checkpoint hash checks, RAM gate, fresh-process verification, and candidate-only save logic.
- Portable defaults anchored to the project script folder, not a developer-specific Windows path.

The v0.3 architecture intentionally imports `Unified_Gutenberg_Arch_v0_2_SSD.py`. Keep the v0.2 source beside v0.3. The v0.3 release gate also imports `train_unified_gutenberg_ssd_ar_mqar.py` for its synthetic data generators. Do not rename these modules unless you also update the imports.

## Requirements and initial checks

- Python 3.10 or newer recommended.
- CPU build of PyTorch.

Open PowerShell and change to the extracted project folder. All commands below assume that directory is the current directory:

```powershell
python -m pip install torch
python Unified_Gutenberg_Arch_v0_2_SSD.py --self-test --threads 2
python Unified_Gutenberg_Arch_v0_3_Associative_SSD.py --self-test --threads 2
python test_unified_gutenberg_v0_3_final_release_gate.py --self-test --threads 2
```

### Important: self-tests do not create a persistent checkpoint

The `--self-test` options perform correctness/plumbing tests. Checkpoint round-trip tests use temporary files that are deleted afterwards. Therefore, a successful `PASS` result with no new `.pt` file is expected.

The project folder is the folder containing these scripts. Default output paths are relative to this script folder (not to whatever directory PowerShell happened to start in):

```text
Models\
reports\
```

## Which command should I use to create a checkpoint?

There are three different operations below. Choose the one matching your goal. A **base/initialized checkpoint** is not the same as a **trained release candidate**.

### A. Create an initial v0.2 checkpoint

`Unified_Gutenberg_Arch_v0_2_SSD.py` does not currently expose a command-line option to save a persistent project checkpoint. Its self-test only creates a temporary test checkpoint. To create a new, randomly initialized v0.2 base checkpoint using the v0.2 module's existing `UnifiedGutenbergSSD` and `save_checkpoint` APIs, run this from the project folder:

```powershell
New-Item -ItemType Directory -Force Models | Out-Null
python -c "from pathlib import Path; import torch; from Unified_Gutenberg_Arch_v0_2_SSD import UnifiedGutenbergSSD, UnifiedGutenbergSSDConfig, save_checkpoint; p=Path('Models/unified_gutenberg_ssd_base_v0_2.pt'); assert not p.exists(), f'Refusing to overwrite existing checkpoint: {p}'; torch.manual_seed(2026); model=UnifiedGutenbergSSD(UnifiedGutenbergSSDConfig()); save_checkpoint(p, model, {'status':'INITIALIZED_RANDOM_UNTRAINED_BASE'}); print('Created:', p.resolve())"
```

This creates **untrained random initial weights**. It is a starting checkpoint, not a trained model and not a successful AR/MQAR result. The command refuses to overwrite an existing file at that destination.

To attempt to train/test a v0.2 checkpoint on synthetic AR/MQAR, run the bundled training helper after the base exists:

```powershell
python train_unified_gutenberg_ssd_ar_mqar.py --base-checkpoint "Models\unified_gutenberg_ssd_base_v0_2.pt" --models-dir "Models" --report-dir "reports"
```

The helper saves a separate `Models\unified_gutenberg_ssd_ar_mqar_passed_v0_2.pt` promoted checkpoint only if its configured held-out gates pass and the source base remains unchanged. It refuses to overwrite an existing promoted checkpoint. If the gates fail, read the JSON report; a promoted checkpoint is not created.

### B. Initialize a v0.3 base checkpoint from a v0.2 checkpoint

Use `Unified_Gutenberg_Arch_v0_3_Associative_SSD.py` when you already have a v0.2 checkpoint and want to create a v0.3 checkpoint with the associative-memory branch added:

```powershell
python Unified_Gutenberg_Arch_v0_3_Associative_SSD.py --init-from-v2 "Models\unified_gutenberg_ssd_base_v0_2.pt" --output "Models\unified_gutenberg_ssd_base_v0_3.pt"
```

The script checks that compatible v0.2 tensors transfer exactly, preserves tied embeddings, reloads the saved v0.3 checkpoint, checks inference parity, and confirms the source v0.2 checkpoint was not changed. It writes an initialization report under `reports` by default and refuses to overwrite an existing destination.

**This is initialization, not training.** The new v0.3 associative-memory parameters are initialized from the configured seed, but the script does not train the memory or prove AR/MQAR capability. A locally created base will have its own SHA-256 and usually will not match the release gate's default pinned base hash.

### C. Train a v0.3 release candidate using the final release gate

Use `test_unified_gutenberg_v0_3_final_release_gate.py` when you already have the v0.3 base checkpoint you intend to validate. The source-only ZIP does not include checkpoint weights. By default, the script expects:

```text
Models\unified_gutenberg_ssd_base_v0_3.pt
```

The known default base SHA-256 pin is:

```text
6668389cc323f3f913054c3573c9ce1b381a241eadb27ed5b1047e5ebe5eb1c1
```

For that exact base, run:

```powershell
python test_unified_gutenberg_v0_3_final_release_gate.py
```

The default run trains for 2,000 optimizer steps, then evaluates held-out AR/MQAR accuracy, full-vocabulary predictions, counterfactual mapping swaps, RAM use, checkpoint round-trip behavior, and fresh-process inference. The release-candidate checkpoint is saved only if the required gates pass; if fresh-process verification fails after it was written, the script removes the candidate. The base checkpoint is kept read-only. A JSON report is written under `reports`.

#### If you deliberately use a different v0.3 base checkpoint

A newly initialized v0.3 checkpoint from step B will usually have a different hash from the pinned base. **Do not simply disable or change the hash pin without checking which file you intend to validate.** From the project folder, run these commands together in the same PowerShell session:

```powershell
$basePath = "Models\unified_gutenberg_ssd_base_v0_3.pt"
$baseHash = (Get-FileHash $basePath -Algorithm SHA256).Hash

python test_unified_gutenberg_v0_3_final_release_gate.py --base-checkpoint $basePath --expected-base-sha256 $baseHash
```

The second command calculates the file's SHA-256, and the final command tells the release gate to verify that selected file against the calculated hash. You may inspect the calculated value by entering `$baseHash` separately before starting the release gate.

**This does not make the checkpoint equivalent to the original pinned base or prove that it is trained.** It intentionally validates the custom file you selected, so only use this override when you have confirmed that this is the base you want to test. Keep the JSON report and the base hash with the experiment records.

You may choose a custom output folder or candidate filename:

```powershell
python test_unified_gutenberg_v0_3_final_release_gate.py --models-dir "D:\GutenbergModels"
python test_unified_gutenberg_v0_3_final_release_gate.py --output-checkpoint "D:\GutenbergModels\candidate.pt"
```

The script prints `[MODEL DIR]`, `[BASE CHECKPOINT]`, `[OUTPUT CHECKPOINT DESTINATION]`, `[CHECKPOINT]`, and `[REPORT]` so you can confirm exactly where it reads and writes files. Existing output files are not overwritten.

## Checkpoint creation hierarchy: v0.2 base → v0.3 release candidate

This is the recommended checkpoint lineage for reproducing the successful experimental run that produced `unified_gutenberg_arch_v0_3_release_candidate_20261025_035353.pt`.

```text
1. v0.2 initialized base
   Models\unified_gutenberg_ssd_base_v0_2.pt
   (random initial weights; not trained)
           |
           | v0.3 inherits the v0.2 backbone and transfers compatible tensors
           v
2. v0.3 initialized base
   Models\unified_gutenberg_ssd_base_v0_3.pt
   (v0.2 tensors copied; v0.3 associative-memory weights initialized; not trained)
           |
           | final release-gate script trains this v0.3 model on synthetic AR/MQAR
           | and evaluates held-out, counterfactual, RAM, and reload checks
           v
3. v0.3 experimental release candidate
   Models\unified_gutenberg_arch_v0_3_release_candidate_<timestamp>.pt
   (saved only after all required release gates pass)
```

### Why each checkpoint must be created first

1. **Create the v0.2 base first.** The v0.2 model definition supplies the backbone architecture and configuration. The v0.3 implementation subclasses that model and needs a compatible v0.2 checkpoint to initialize from. A `--self-test` only uses a temporary checkpoint and does not create this persistent base.

2. **Create the v0.3 base from the v0.2 base.** The release gate loads a checkpoint whose architecture is v0.3. The `--init-from-v2` command transfers compatible v0.2 tensors (63 tensors in the logged run), adds newly initialized associative-memory parameters, and verifies save/reload parity. This step changes the checkpoint format and architecture payload to v0.3; it does **not** train the model.

3. **Run the v0.3 release gate from that v0.3 base.** The release gate uses the v0.3 base as its starting weights, trains a separate in-memory model, and writes a *different* release-candidate checkpoint only after the required evaluation gates pass. The base checkpoint is protected and should remain unchanged. If validation fails, a candidate is not retained (and a candidate written before a later fresh-process failure is removed).

4. **The optional v0.2 AR/MQAR trainer is a separate branch, not a required checkpoint in this successful path.** `train_unified_gutenberg_ssd_ar_mqar.py` attempts to train and promote a v0.2 checkpoint only if its own gates pass. In the recorded run, its gate failed, so no promoted v0.2 checkpoint was created. The successful v0.3 experiment continued from the initialized v0.2 base instead. Keep the failed report for audit purposes; do not describe that v0.2 checkpoint as trained successfully.

### Recorded successful checkpoint lineage (example)

The following values identify the successful run recorded on October 25, 2026. They are an audit trail for that run, not constants expected from every initialization:

| Artifact | Recorded SHA-256 / outcome |
|---|---|
| v0.2 initialized base | `91f17be338f3b359d90ba7bd46fd50cb0f9e166eec3210d58b33eb4d199694c5` |
| v0.3 initialized base | `8d5614d27689ac529f4d21582fbfeac84b5e09634be169e424697b81042300db` |
| Final v0.3 release candidate | `e7dcf14e495ad5afa1d51027ca76e547c9d0292a313705d40434f1c9c0b50640` |
| Candidate filename | `unified_gutenberg_arch_v0_3_release_candidate_20261025_035353.pt` |
| Release-gate status | `PASS_EXPERIMENTAL_OPEN_WEIGHT_RELEASE_CANDIDATE` |

The candidate was saved under `Models\` and its JSON report under `reports\`. Keep the candidate and its matching report together. Different initializations or runs may produce different hashes and timestamped filenames; always record the exact values printed by the run you are using.

## Checkpoint workflow at a glance

| Script / command | What it does | Persistent checkpoint? |
|---|---|---|
| `Unified_Gutenberg_Arch_v0_2_SSD.py --self-test` | Runs v0.2 correctness tests; temporary checkpoint is removed | **No** |
| v0.2 API command in section A | Creates a random, untrained v0.2 base | **Yes** |
| `train_unified_gutenberg_ssd_ar_mqar.py` | Trains v0.2 on synthetic recall; saves promoted artifact only if gates pass | **Only if gates pass** |
| v0.3 `--init-from-v2 ... --output ...` | Transfers v0.2 tensors and initializes v0.3 associative-memory parameters | **Yes, initialized/not trained** |
| `test_unified_gutenberg_v0_3_final_release_gate.py --self-test` | Checks imports, correctness and synthetic-data plumbing | **No** |
| Full `test_unified_gutenberg_v0_3_final_release_gate.py` | Trains from an existing v0.3 base and evaluates release gates | **Only if all release gates pass** |

## Scope and interpretation

A passing release gate validates the tested synthetic associative-recall tasks and byte-level inference plumbing only. It does not establish natural-language pretraining or general-purpose language capability. Retain the checkpoint SHA-256 and associated JSON report as a pair for reproducibility.
