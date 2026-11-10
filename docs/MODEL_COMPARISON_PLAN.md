# Final milestone: model comparison and controlled augmentation

Prepared 2026-11-10 for a roughly one-week work window. **This is a proposed
extension protocol and team handoff, not evidence of new training.** The user's
current professor feedback is to compare additional models and add augmentation
and explainability. It supersedes the earlier one-baseline-only work scope for
this extension; the completed baseline remains preserved.

## Recommendation and assignments

| Owner | Work | Question answered |
| --- | --- | --- |
| Teammate 1 | [R(2+1)D-18 classifier](handoffs/R2PLUS1D18.md), no augmentation | How does factorizing spatial and temporal convolutions change the outcome? |
| Teammate 2 | [R3D-18 classifier](handoffs/R3D18.md), no augmentation | How does a standard residual 3D video classifier compare under our existing protocol? |
| Bijay | [U-Net++ augmentation ablation and explanations](AUGMENTATION_EXPLAINABILITY_PLAN.md) | Does training-only augmentation change performance, and which inputs influence the RD score? |

These choices reuse the existing video tensors and an established torchvision
model API. R3D uses joint 3D convolutions; R(2+1)D factorizes them into spatial
and temporal operations with an intervening nonlinearity. This gives a specific
architecture question, rather than just a longer list of model names.
[Primary sources and implementation details](research/video-model-comparison-sources.md).

Assignment updated at the user's request: teammate 1 receives the more involved
implementation, R(2+1)D-18. Its factorized stem and intermediate feature maps need
additional care during grayscale adaptation and memory checks. This is an
implementation judgment, not a measured claim about which model trains slower
or converges less readily.

**Both are feasibility recommendations, not predicted winners.** Both are larger
than our 3.60M-parameter baseline; factorization does not guarantee faster training
or lower memory. Real-input timing and memory probes come before full training.
Avoid adding transformer training or a segmentation task to this week's minimum
deliverable. Our task has clip labels; pixel-level masks have not been established.

## What already exists

The completed run is `unetpp_rd_002_amp`. Its name follows the released ERDES
classifier, but its implementation is a **3D encoder plus global pooling and a
classification head**, not a nested segmentation decoder. It returns one RD logit
per clip. See [verified results](UNETPP_RESULTS.md) and
[source audit](research/erdes-unetpp-protocol.md).

| Reference fact | Existing evidence |
| --- | --- |
| Official task | `non_rd_vs_rd`; 0 = normal/PVD (non-RD), 1 = RD |
| Training / validation | 3,873 clips (362 RD) / 431 clips (40 RD) |
| Official test | 1,077 clips (100 RD), already evaluated |
| Selected reference epoch | 48, by highest validation accuracy; earliest ties |
| Reference validation | Accuracy 98.84%, sensitivity 92.50% |
| Reference official test | TP 86, TN 964, FP 13, FN 14; accuracy 97.49%, sensitivity 86.00% |

Compare new validation results with **reference validation**, and new test results
with **reference test**. Published ERDES results are author-reported, separate
from our measurements. In particular, torchvision R3D-18 is **not** the ERDES
paper's deeper `[4,4,4,4]` MONAI ResNet; neither proposed model is an exact
reproduction of that published ResNet row.

## Read before implementing

Every teammate needs this repository at the same starting revision, including
`src/`, `configs/`, `scripts/`, `tests/`, and notebooks. Sending just a handoff
without its linked files is insufficient. Read in this order:

1. `AGENTS.md`, then this document and the assigned handoff.
2. `docs/UNETPP_RESULTS.md` and `configs/unetpp_rd_002_training.json` for the
   completed result and inherited protocol.
3. `src/rd_ultrasound_ml/unetpp_baseline.py`, `unetpp_training.py`,
   `unetpp_data.py`, `unetpp_kaggle.py`; inspect the matching tests before reuse.
4. `docs/KAGGLE_GUIDE.md`, `scripts/build_kaggle_notebooks.py`, and
   `scripts/prepare_kaggle_job.py` when building the remote jobs.

Course material checked: `docs/course-guideline/AI_Coding_Agents_Practical_Reading_Guide.pdf`.
It stresses context, checkable changes, evidence, reproducibility and leakage.
`AGENTS.md` names `docs/course-guidance/`, which is absent; the existing directory
is `docs/course-guideline/`. No final grading rubric or written version of the
new professor feedback was found. Use the user's feedback above without
inventing grading requirements or a confirmed presentation date.

## Shared experiment protocol

Use [the existing training config](../configs/unetpp_rd_002_training.json) as the
source of inherited settings. Write a **new config and output directory** for
each experiment. Freeze each resolved config before its full run.

| Item | Required first comparison |
| --- | --- |
| Dataset identity | Same pinned release/revision and original train/validation membership |
| Cached input | Float32 `[1,96,128,128]`; arithmetic grayscale, square zero padding, full-volume trilinear interpolation, divide by 255 |
| Model initialization | From scratch; torchvision `weights=None`; one input channel and one output logit |
| Normalization of input | Existing `[0,1]` values; no extra RGB/Kinetics normalization |
| Training | 50 epochs, seed 42, all 3,873 training clips each epoch |
| Optimizer and loss | Adam, learning rate `1.5e-5`, betas `(0.9,0.999)`, eps `1e-8`, zero weight decay; unweighted `BCEWithLogitsLoss` |
| Scheduler | Inherit existing ReduceLROnPlateau on validation loss, factor 0.1, patience 10, including its remaining resolved defaults |
| Sampling and augmentation | No class resampling, weighting or augmentation in either peer's first run |
| Precision | Existing FP16 autocast, FP32 master weights and GradScaler, after finite-gradient probe |
| Effective batch | 48; start by probing physical batch 4 / accumulation 12 |
| Checkpoint selection | Highest validation accuracy at threshold 0.5; earliest ties, matching reference |
| Evaluation | Unaugmented validation; one sigmoid probability per clip; threshold 0.5 fixed |

This is an **architecture comparison under the inherited training recipe**, not
each architecture's best achievable performance. Inheriting a small learning
rate can disadvantage a new model. Report that limitation. Do not silently change
the optimizer, learning rate, frame count or pretrained status to obtain a better
result. If an optimization study becomes necessary, record it as a separate
validation-only extension with its own trial budget and config.

### Resource gate, BatchNorm and recovery

1. Verify synthetic input/output shape and finite backward gradients. Then probe
   real **training** clips from both classes on the intended GPU, with warmup,
   synchronized step timing and peak allocated/reserved CUDA memory.
2. Try physical batch 4 first. If either model cannot fit, coordinate a common
   smaller physical batch, preferably 2 / accumulation 24 (or 1 / 48), before
   either final run starts. A smaller physical batch remains a disclosed
   difference from the original baseline. Save the tested settings.
3. Torchvision models use BatchNorm, whereas the existing U-Net++ uses
   InstanceNorm. Gradient accumulation preserves the optimizer's effective batch
   size but **does not** make BatchNorm observe 48 clips at once. Report physical
   batch and normalization; retain the architecture's default BatchNorm for the
   first comparison. Do not claim a normalization-controlled ablation.
4. Complete one full train/validation epoch, then project remaining time using
   its measured wall time plus cache verification, checkpoint/export and restart
   overhead. Record the available GPU budget; no free-quota assumption.
5. Use resumable bounded jobs and save normally before their time limits. Resume
   only that experiment's model/optimizer/scheduler/scaler/RNG state. Probe weights
   are disposable; final training starts from the recorded seed and fresh weights.

If 50 epochs will not fit, retain the partial result and label its exact epoch
budget. A practical fallback is a separately named 2D ResNet-18 with fixed sparse
frames and mean-pooled frame features, but that changes the temporal protocol and
requires a new documented plan. Do not silently reduce either 3D model to 16
frames or present a partial run as comparable full-budget training.

## Reuse the cache without reusing baseline identity

The existing GPU notebook and submission scripts are **U-Net++ specific**. Changing
only an import or `experiment_id` is insufficient:

- `load_prepared_shards()` in `unetpp_kaggle.py` compares each manifest's
  `config_digest` to the **entire** supplied config, including model/code identity.
  Passing a new model config directly will reject valid existing caches.
- Implement a narrow adapter that validates old shards against their original
  preparation config, retaining all tensor/source hashes, partition membership,
  shapes, labels, order and preprocessing checks. Record the original preparation
  config digest and returned cache fingerprint alongside the new training config
  digest and source hashes. Bind both identities to the new run's resume checks.
  Training augmentation belongs to the new run, separate from deterministic
  cached preprocessing. Do not rewrite manifests or weaken validation guards.
- `run_training(model_factory, ...)` can already accept a different classifier,
  but its signature alone does not solve cache loading, model-specific exports,
  runtime recording, or notebook embedding.
- Existing notebook builders can overwrite reference configs/notebooks. Create
  model-specific builders/bundles and directories; preserve the completed source
  snapshots, result archives and frozen configs. Do not resume `unetpp_rd_002_amp`.
- `unetpp_evaluation.py` contains reference-specific freeze checks. Build a new
  evaluation entry point with explicit model identity rather than relabeling an
  old freeze or bypassing its checks.

Use private, version-pinned Kaggle preparation outputs if each peer's account can
access them. Bijay must supply the two exact source/version identifiers and grant
access through the team's normal process. Private files are not accessible merely
because their names appear in this repo. Otherwise follow `docs/DATA_ACCESS.md`
to prepare identical inputs independently and retain their provenance. Never
exchange API tokens or commit the roughly 27 GB cache/checkpoints.

Use uv for local Python environments; keep project dependencies unchanged unless
needed. GPU notebooks may use the documented managed runtime, recording Python,
torch, **torchvision**, CUDA/cuDNN and GPU versions. Confirm torch/torchvision
compatibility there; neither model has been tested in that runtime by this plan.

## Holdout policy and reporting

The official test set is **already exposed at project level**, including baseline
scores. Develop all additions on training/validation only. Freeze the complete
experiment list, configs, checkpoints and threshold before new test inference.
Evaluate each frozen new model once and retain every result, not just the winner.
Label it **reused official test benchmark**, not a fresh untouched holdout. No
test-driven threshold tuning, augmentation changes or selective reruns. Preserve
the existing reference test predictions; do not rerun the reference for a nicer row.

Official splits are clip-level and may share patients; neither these comparisons
nor clip-level confidence intervals establish unseen-patient generalization.
One seed limits robustness claims. A new patient-separated or external evaluation
needs verified grouping/access and a separately documented protocol.

For each partition, save TP/TN/FP/FN, sensitivity, specificity, precision, F1,
accuracy, balanced accuracy, average precision and ROC-AUC. AP is not automatically
the same as trapezoidal PR-AUC. Keep metric definitions from `binary_metrics()`.
Report parameter count, physical/effective batch, epochs, selected epoch, training
time, peak memory, and inference timing with hardware/batch/precision specified.
Accuracy alone is inadequate for this imbalanced task.

Use a four-row table: existing U-Net++, R3D-18, R(2+1)D-18, U-Net++ + augmentation.
Keep validation and reused-test tables separate. Fill unmeasured cells with
`pending`, not estimates. The augmentation comparison is baseline versus
augmented U-Net++; it does not establish that augmentation helps every model.

## Required return package from each teammate

Commit small code/config/notebook changes on `feature/r3d18-rd` or
`feature/r2plus1d18-rd`; keep shared-interface changes coordinated to limit merge
conflicts. Return a concise model report and a private artifact location with:

- Resolved config, code commit/source hashes, runtime, exact cache source versions,
  fingerprint and unchanged split manifests.
- Probe and epoch timing, actual epochs completed, checkpoint-selection rule and
  selected epoch; per-epoch train/validation history and validation predictions.
- Prediction rows containing `clip_id`, dataset label, probability, predicted
  class and partition; enough to independently recompute all reported metrics.
- Selected checkpoint SHA256, resumable state and artifact hashes in ignored
  storage. A successful reload must reproduce saved validation probabilities
  within a recorded numerical tolerance in the same environment.
- Tests/checks actually run, failures, deviations and incomplete work. Synthetic
  tests are code evidence, not ultrasound-model performance.

## One-week sequence

| Day | Teammates 1 and 2 | Bijay |
| --- | --- | --- |
| 1 | Read handoffs; implement adapters; shape/gradient/cache checks and GPU probe | Lock protocol; check cache access; inspect training-only augmentation examples |
| 2 | Complete first full epoch, measure total budget, start bounded full runs | Start U-Net++ + augmentation from scratch; implement baseline Grad-CAM |
| 3–4 | Continue/resume training; keep all logs and predictions | Continue ablation; check explanations and occlusion/randomization controls |
| 5 | Finish runs; audit validation; freeze all final choices together | Assemble validation comparison and fixed-clip explanation panels |
| 6 | Run frozen benchmark evaluation; independently verify rows/metrics | Export comparison tables, matched curves, confusion matrices and overlays |
| 7 | Review methods/deviations and rehearse | Final slides: question, experiment matrix, results, failure cases, limitations |

Minimum defensible progress is completed comparisons with auditable evidence,
one controlled augmentation ablation and checked explanation examples. Additional
seeds are useful only after these fit the measured budget. A lower-scoring model
still contributes a valid comparison; report what happened.
