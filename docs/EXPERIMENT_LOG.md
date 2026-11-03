# Experiment Log

The first 256-clip Colab GPU pilot is complete. Record subsequent runs using the
template below. Validation findings are development results; the test set remains reserved.

Current preparation: [baseline_001 proposal](BASELINE_PLAN.md) and
`configs/baseline_001.json` record the candidate GPU pilot. Input preparation
passed on eight inspection clips locally and in Colab. The GPU pilot is implemented
in `notebooks/baseline_training.ipynb`, using `configs/baseline_001_training.json`.
Its next sections audit the saved predictions and implement the controlled
`baseline_002_weighted_pilot` comparison using `configs/baseline_002_weighted.json`.
The weighted comparison has completed in the user's saved Colab execution.
Both custom-CNN runs are engineering pilots, not ERDES paper-benchmark reproductions.
The next target is one released UNet++ classifier baseline; see
[the reproduction plan](UNETPP_BASELINE_PLAN.md).

## 2026-10-28 — baseline_001_pilot: completed Colab run

- Run ID: `20261028T165642454326Z`; evidence: the user's saved executed
  `notebooks/baseline_training.ipynb` and reported table. Archived the completed
  notebook under ignored `outputs/baseline_training_verification/` before adding
  follow-up cells. The remote ZIP itself is not available locally for independent inspection.
- Question: can the small, unweighted 3D CNN establish a useful first reference
  on the preserved official clip-level partitions?
- Data: 256 training clips (232 non-RD, 24 RD), all 431 validation clips
  (391 non-RD, 40 RD); no final test use. Exact manifest hashes are in the exported
  `manifests.json`/`summary.json`; follow-up section 10 verifies them in the live session.
- Input: float32 `[B,1,16,128,128]`, fixed uniform frame sampling, grayscale,
  square padding, area resize, divide by 255, no augmentation. The numerical
  roundoff correction is recorded in `preprocessing_implementation.json`.
- Model: compact strided 3D CNN, 8/16/32 channels, global mean and one logit,
  17,585 parameters, from scratch.
- Training: seed 42, Adam 0.001, batch 8, five epochs, unweighted BCEWithLogitsLoss;
  fresh initialization after the diagnostic. Selection: highest validation AP,
  earliest on ties; threshold fixed at 0.5.
- Environment: Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0,
  numpy 2.1.3, pandas 2.2.3, OpenCV 5.0.0, scikit-learn 1.6.1.
- Preflight passed: four-clip diagnostic reached loss 0.3242589 and training
  accuracy 1.0 after 25 steps; all 687 video hashes and both prepared arrays passed.
  Reported preprocessing duration: 180.1 seconds.

| Epoch | Training loss | Validation loss | Validation AP | RD sensitivity at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.443322 | 0.354354 | 0.065550 | 0 |
| 2 | 0.336223 | 0.322154 | 0.065871 | 0 |
| 3 | 0.322931 | 0.315691 | 0.066106 | 0 |
| 4 | 0.321433 | 0.313046 | 0.077541 | 0 |
| 5 | 0.316059 | 0.311541 | 0.090683 | 0 |

Selected checkpoint: epoch 5, validation AP 0.09068340359421731. At threshold 0.5:
TP=0, TN=391, FP=0, FN=40, sensitivity=0, specificity=1, precision undefined,
F1=0, balanced accuracy=0.5, accuracy=0.9071925754060325, ROC-AUC approximately
0.449105 (rounded in the displayed table). The constant non-RD reference has
the same thresholded predictions, AP 0.09280742459396751 and ROC-AUC 0.5.

Interpretation: this run establishes an executed engineering baseline with no
demonstrated validation improvement over the constant reference. It missed all
40 RD validation clips at 0.5. Falling loss does not establish useful discrimination.
The outputs alone do not establish that class imbalance is the sole cause.
Metrics/checkpoint selection were checked against the code and reference arithmetic;
independent recomputation from the remote prediction CSV is the next notebook step.

Limitations: one small training subset, one seed, five epochs, clip-level splitting
with possible same-patient overlap, no clinical validation or unseen-patient claim.
The export cell reports the original results ZIP was created; download/persistent
storage of that ZIP is not independently confirmed. No commit/tag was created.

## 2026-10-28 — baseline_002_weighted_pilot: completed engineering comparison

Hypothesis: training-only positive-class weighting may improve RD discrimination
and avoid all-non-RD predictions. Change only the training loss to
`BCEWithLogitsLoss(pos_weight=232/24)`, with the ratio recomputed from the saved
training manifest. Keep all original inputs, rows/order, seed, fresh initialization,
model, optimizer, learning rate, batch size, five epochs, selection rule, and
threshold 0.5 fixed. Validation loss stays unweighted for comparison. Training
loss scales differ between experiments. No resampling, threshold tuning, or test use.

Run sections 10–12 in the same connected T4 session after preserving the original
ZIP. Section 10 checks predictions/metrics/labels and saves classwise score
summaries. Section 11 creates a separate run and trains a fresh model. Section 12
exports the new run and a comparison table. Treat both all-RD and all-non-RD
outcomes as failures to demonstrate useful thresholded discrimination; compare
AP and ROC-AUC alongside sensitivity/specificity. Weighting does not guarantee
improvement or calibrated probabilities.

The saved notebook now confirms the prediction audit passed and the weighted
run `20261028T201752658036Z_weighted` completed five epochs and exported a ZIP.
Its parent is the repeated baseline run `20261028T193629481701Z`; the repeated
unweighted losses/AP match the first run reported above (timings differ).

| Selected validation result | Unweighted pilot | Weighted pilot | Constant non-RD |
| --- | ---: | ---: | ---: |
| Selected epoch | 5 | 5 | — |
| Average precision | 0.0906834036 | 0.1453199778 | 0.0928074246 |
| ROC-AUC (rounded display) | 0.449105 | 0.611061 | 0.500000 |
| Sensitivity at 0.5 | 0 | 0 | 0 |
| Specificity at 0.5 | 1 | 1 | 1 |
| Balanced accuracy at 0.5 | 0.5 | 0.5 | 0.5 |
| Accuracy at 0.5 | 0.907193 | 0.907193 | 0.907193 |
| Unweighted validation BCE | 0.311541 | 0.642468 | not reported |

Both selected models have TP=0, TN=391, FP=0, FN=40 and undefined positive
precision. Weighting raised AP by 0.0546365742 in this single validation run,
but did not produce any RD detection at the fixed threshold. This is preliminary
ranking improvement, not a robust or clinical finding. Validation BCE worsened;
weighted scores have not been established as calibrated probabilities. The
baseline audit found very similar mean scores for non-RD (0.075287) and RD
(0.074952). Raw prediction files/remote ZIPs have not been independently inspected
locally; evidence here is the executed notebook and its successful audit output.

The user's clarification on 2026-10-30 makes one feasible **paper-aligned reference
reproduction** the priority before a new contribution. Preserve these results
as engineering evidence. The target uses the released UNet++ classifier, 96-frame
trilinear inputs, full official training/validation sets and the reference
50-epoch protocol, with paper/code disagreements disclosed. Its FP32 CUDA
preflight subsequently passed (2026-11-01, below); full reference training and
test comparison remain outstanding. See [the source audit](research/erdes-unetpp-protocol.md).

## 2026-11-01 — UNet++ T4 resource check complete; full training prepared

The user's saved `unetpp_resource_check.ipynb` execution passed on a Tesla T4,
PyTorch 2.11.0+cu130 and Python 3.13.16. Batch 4 used peak allocated memory
6,543,866,368 bytes; median FP32 train/validation steps were 1.22137/0.32998
seconds. All tested batches (1, 2, 4) had finite nonzero gradients. No research
weights were retained and no test data were used. The projected 50-epoch
compute time is 16.92 hours, excluding full-data IO, checkpointing and restarts.

Saved remote evidence:
`MyDrive/rd-ultrasound-ml/outputs/unetpp_resource_check/20261101T193236373065Z/`.
This is a hardware check, not a model-performance result.

Next implementation: `unetpp_training.ipynb`, configuration
`unetpp_rd_001_training.json`, and reusable training/cache modules. The new
`unetpp_rd_001_amp` protocol explicitly adds mixed precision after a real-input
finite-gradient check, physical batch 4 with accumulation 12, full official
train/validation partitions and a fixed 50-epoch target. Adam and checkpoint by
validation accuracy follow the released code. The first call runs one epoch
for actual IO timing; continuation preserves the same run. Training/recovery
artifacts persist directly to Drive; the 27.08 GB input cache stays on runtime
disk. Config/code/runtime/input mismatches reject incompatible resume.

CPU tests cover short-window normalization and interrupted training with exact
synthetic resumed/uninterrupted parameter agreement. The GPU AMP probe, complete
4,304-clip acquisition, full 50-epoch run and official test comparison have not
yet been executed locally. No new model-performance results are claimed.
See [the full run guide](UNETPP_BASELINE_PLAN.md).

## 2026-11-01 — Kaggle workflow prepared after Colab quota exhaustion

User evidence: Colab preparation completed all 3,873 training and 431 validation
clips, printing the expected constant non-RD reference. GPU access was then
exhausted and the runtime disconnected before training section 6. No full-data
UNet++ model result is recorded.

Prepared Kaggle CPU-shard and GPU-training notebooks under model/task names,
`unetpp_rd_002_amp` config, an account setup wizard and official CLI wrappers.
The two CPU outputs preserve float32 inputs while each fits within Kaggle's
20 GB saved-output limit. The GPU job reads both, probes precision and runs a
bounded training interval; later jobs import a preceding successfully saved
training output. Kaggle's save boundary differs from mounted-Drive persistence.
Source/label/hash/runtime guards remain; test data remain unopened.

Local synthetic evidence exercises both CPU preparation cells, attached tensor
verification, partial training/export ZIP hashes and saved-output continuation.
The official CLI version/help and metadata request construction are verified
without submitting a job. Remote account eligibility, source acquisition and
GPU training still need the user's authenticated Kaggle session. See
[Kaggle instructions](KAGGLE_GUIDE.md).

## 2026-11-02 — Diagnose missing Kaggle preparation manifests

The first CUDA-enabled training attempt reported T4/PyTorch 2.11.0+cu128, then
failed before training because zero preparation manifests were attached. Direct
status/output inspection found both CPU preparation jobs in ERROR, each stopping
on `erdes_metadata.csv` with `URLError` after five retries. No clip preparation
completion is established. CPU metadata requested Internet; the logs do not
identify the underlying network reason.

After user phone verification, both CPU jobs were resubmitted as version 2 with
GPU disabled. One was observed RUNNING. Download progress and completion remain
unconfirmed: log retrieval returned HTTP 429, so polling was stopped. No further
GPU job was launched. The user was asked to inspect the current CPU log in Kaggle.

Changed `scripts/kaggle_cli.py` to check dependency completion before GPU
submission, with regression tests for ERROR and successful continuation inputs.
The tests reproduced the missing check before the fix and passed afterward.
This prevents allocating a GPU for known failed/incomplete notebook dependencies;
it does not change training, preprocessing, splits or source integrity checks.
See [the Kaggle recovery note](KAGGLE_GUIDE.md).

## 2026-11-02 — Kaggle CPU inputs saved; first GPU epoch submitted

The user supplied a successful CPU preparation 00 version 2 log: all 2,152
selected clips completed, runtime 5,691.4 seconds, 13.54 GB saved output and
Accelerator None. No download errors appeared in that successful run. Direct
Kaggle status checks then confirmed both CPU preparation 00 and 01 COMPLETE.
The earlier network failure no longer blocks those saved inputs; its exact
connection-level cause was not established.

The preceding GPU job remained ERROR and had not trained. Its local bundle
still selected one epoch, both official preparation-output dependencies and a
T4 accelerator. The guarded CLI verified both dependencies COMPLETE and
successfully submitted GPU training version 3. GPU input validation, precision
probe, actual training completion and performance are not yet confirmed.
Do not launch a parallel browser execution against this run.

## 2026-11-02 — First full-data UNet++ epoch complete; continuation prepared

Kaggle GPU job `bijaykhadka1710/unetpp-rd-002-train-01`, version 3, completed
successfully in 692.9 seconds. All 4,304 attached inputs were hash-verified; no
videos were downloaded or decoded on the GPU job. The FP16 probe passed with
finite nonzero gradients, no measured skipped updates, physical batch 4 and peak
allocated memory 3,335,413,248 bytes. Only CUDA device 0 is used on the T4 x2
allocation. The actual training/validation epoch took 402.547 seconds (6.71 min).

| Epoch-1 validation measurement | Value |
| --- | ---: |
| Train BCE | 0.5878527716 |
| Validation BCE | 0.5202592794 |
| TP / TN / FP / FN | 0 / 391 / 0 / 40 |
| Sensitivity / specificity at 0.5 | 0 / 1 |
| Accuracy | 0.9071925754 |
| Average precision | 0.1029200869 |
| ROC-AUC | 0.4749360614 |
| Balanced accuracy | 0.5 |

These are one-epoch validation results. Accuracy equals the constant-negative
reference and all 40 positive validation clips are missed at the fixed threshold;
no useful thresholded RD detection is established. The experiment remains
`partial`, 1/50 epochs. Test data were not used. Do not compare this partial
validation result directly with the article's reported test row.

The agent downloaded the results ZIP into ignored
`outputs/kaggle_results/session_01/`, verified all 21 artifact hashes and safely
extracted it under `verified/unetpp_rd_002_amp/`. The authoritative checkpoint
loads with weights_only=True, records epoch_complete, all 3,873 training examples,
81 optimizer-update attempts and zero skipped updates, and contains optimizer,
scheduler, scaler and RNG state. Model tensors are finite; selected best weights
match the checkpoint. Saved validation labels match the official validation
manifest and recomputed metrics match the saved summary. Detailed verification
is in `outputs/kaggle_results/session_01/verification.json`.

Prepared a separate continuation bundle `unetpp-rd-002-train-02` for 10 additional
epochs (2–11), requiring both CPU outputs and saved GPU job 01. The fixed target
remains 50 epochs and no model/training choices change. At the measured epoch
rate this interval projects about 67 minutes of epoch work plus input validation,
probe, saving and platform overhead; this is not a guaranteed duration. The guarded CLI submitted continuation job 02 version 1, and Kaggle reports it
RUNNING. Its embedded configuration and implementation hashes were checked
against the verified epoch-1 artifact and match exactly. Completion and new
metrics remain pending.

## 2026-11-02 — Epochs 2–11 saved; final training interval prepared

The user saw a stale running timer after reopening their laptop. Direct Kaggle
status inspection reported GPU job `unetpp-rd-002-train-02` COMPLETE, and its
saved log shows epoch 11 and ZIP export at approximately 3,676 seconds, followed
by notebook rendering through 3,686 seconds (about 61.4 minutes). Refreshing the
browser confirmed completion. The saved run was not stuck at the excerpt's
last displayed epoch-4 line. This does not establish the state of any separate
interactive draft session.

The actual continuation resumed epoch 2 from the saved first job and completed
all ten requested epochs, reaching 11/50. Epochs 3–11 each took about 321–323
seconds; epoch 2 took 467.75 seconds. All recorded skipped-update counts are zero.
Epoch-11 validation loss is 0.2095855, AP 0.7980883, ROC-AUC 0.9455882; threshold
0.5 still yields TP=0, TN=391, FP=0, FN=40. These are latest-epoch validation
measurements, not selected-checkpoint or test results. Highest-accuracy selection
retains epoch 1 because all validation accuracy values tie; no selection rule
was changed to exploit AP/AUC.

The results ZIP was downloaded under ignored `outputs/kaggle_results/session_02/`.
All 31 artifact hashes were verified. The authoritative checkpoint records epoch
11, epoch_complete, a contiguous 11-epoch history, finite model tensors, and
optimizer/scheduler/scaler/RNG state. Epoch-1 and epoch-11 predictions match the
validation manifest and reproduce the logged metrics. Configuration and input
fingerprint match the first saved run. Detailed checks are saved in
`session_02/verification.json`.

Prepared GPU job `unetpp-rd-002-train-03` for the remaining 39 epochs (12–50),
requiring both CPU preparation outputs and saved job 02. Its embedded config and
implementation hashes match the verified checkpoint. At the latest steady epoch
rate, epoch work projects about 3.5 hours; allow additional setup/save overhead
and runtime variation. The existing six-hour between-epoch budget remains and
may return a partial run if actual timing is slower. This is a measured estimate,
not a completion guarantee. The final test partition remains unused. The guarded CLI confirmed all three input jobs COMPLETE and successfully
submitted job 03 version 1. Kaggle subsequently reported job 03 RUNNING; completion is not yet confirmed.

## 2026-11-02 — 50 epochs verified; epoch 48 frozen before official-test access

GPU job `unetpp-rd-002-train-03`, version 1, reports COMPLETE. Its saved log ends
around 14,058 seconds (3.91 hours). The agent downloaded the results ZIP under
ignored `outputs/kaggle_results/session_03/`, verified all 70 artifact hashes,
recomputed all 50 epochs' metrics from saved predictions and checked model state,
checkpoint selection, configuration and input fingerprints.

Epoch 48 is the first maximum-validation-accuracy checkpoint; epoch 49 ties.
Selected validation results: TP=37, TN=389, FP=2, FN=3; sensitivity 0.925,
specificity 0.99488491, precision 0.94871795, F1 0.93670886, accuracy 0.98839907,
AP 0.97313857 and ROC-AUC 0.99616368. These are validation results, not final-test
metrics. Epoch 50 was not substituted for the fixed selected best checkpoint.
One AMP optimizer update was skipped in epoch 35; the audit reconciles 4,050
attempted updates with 4,049 actual Adam steps. Final weights are finite.

Created `configs/unetpp_rd_002_evaluation.json` with the selected checkpoint
SHA256, config/data/source fingerprints, threshold and inference settings **before
opening test.csv in this evaluation workflow**. Earlier dataset split audits
had already read test membership/counts, without using model test performance.
After freezing, this workflow's test-split audit confirmed 1,077
clips, 977 non-RD and 100 RD, with matching metadata labels and no clip overlap.
No model inference or tuning occurred during that split audit.

Added separate test CPU-preparation and GPU-inference notebooks, a frozen
inference module, version-pinned private bundle generator, and five behavioral
tests. All 62 unit tests passed; notebook schema/embedded execution and synthetic
inference/export/hash checks passed. Final-test CPU preparation was submitted
as `unetpp-rd-002-test-inputs` version 1 and observed RUNNING. GPU official-test
inference and paper-table comparison remain pending. No model architecture,
training split, training implementation or selected-checkpoint rule changed.

See [UNet++ results and freeze](UNETPP_RESULTS.md). Local evidence includes
`session_03/verification.json`, `session_03/test_split_audit.json` and ignored
`outputs/unetpp_verification/evaluation_notebook_check.json`.

## 2026-11-02 — Test CPU preparation complete; original GPU image pinned

The official-test CPU job version 1 completed in 2,656.1 seconds, saving all
1,077 float32 clips (6.78 GB) without a GPU. The user-provided log shows the
frozen epoch-48 checkpoint hash verified, all source hashes recorded and all
clips prepared. Direct status inspection confirms COMPLETE. No official-test
model inference had been performed at this point.

The new CPU container changed Python/Hugging Face versions but retained the
core preprocessing versions required by the existing cache-identity guard.
To preserve the GPU evaluation's unchanged strict runtime check, retrieved the
original final-training container URI by authenticated version-1 metadata pull.
Added `configs/unetpp_rd_002_runtime.json`, taught the evaluation bundle builder
to retain that original Docker image, and extended its existing metadata test.
The model/evaluation freeze, weights, threshold, split and inference module were
not altered. The prior unsubmitted bundle is preserved in ignored diagnostics.
Official CLI request serialization was checked with its network client mocked.

The final inference job requires training version 1 and test-input version 1.
The guarded CLI verified both pinned input jobs COMPLETE and submitted GPU
evaluation version 1. Direct status reports RUNNING. A metadata pull confirms
the server retained the original training image digest. GPU test metrics remain
pending; no further training or threshold tuning is performed. All 62 tests
passed after the bundle-pin change, and the official CLI request-construction
check (network mocked) passed.

## 2026-11-02 — Frozen official-test evaluation completed and verified

Private Kaggle job `bijaykhadka1710/unetpp-rd-002-test-evaluation/1` completed
successfully in about 270.22 seconds, including input verification/export;
inference itself took 31.07 seconds. The original training container was retained,
and every recorded GPU runtime field matched training. The fixed epoch-48
checkpoint, threshold 0.5 and evaluation freeze matched the archived identities.
No training, threshold tuning, calibration or checkpoint selection occurred.

All 1,077 official test clips were evaluated: 977 non-RD and 100 RD.
Confusion counts: TP=86, TN=964, FP=13, FN=14. Sensitivity 0.86, specificity
0.9866939611, precision 0.8686868687, F1 0.8643216080, accuracy 0.9749303621,
balanced accuracy 0.9233469806, AP 0.9268454897 and ROC-AUC 0.9868577277.
Test BCE was 0.0813075756. Thus 14 of 100 RD clips were missed at threshold 0.5.

Downloaded the small test-results ZIP into ignored
`outputs/kaggle_results/test_evaluation/`. Independent CPU verification checked
all 12 artifact hashes, all 1,077 prediction rows against the official test
manifest/order/labels, all thresholded labels and all classification metrics.
Archive SHA256:
`9834929ed281d7cf03a978de4b977c147811fb2f65fbf92be9693970fb5c3e03`.
Evidence: `verification.json`, `verified_comparison.csv`, and the saved job log
in that directory. No test tensor arrays were downloaded locally.

The [paper's Table 8](https://arxiv.org/html/2508.04735v2) reports UNet++ Non-RD
vs. RD accuracy 97.6% and sensitivity 87.9%; ours is lower by 0.11 and 1.90
percentage points, respectively. This completes one adapted, code-aligned
single-seed benchmark with the documented optimizer/decoder/precision differences;
it is not exact numerical replication or clinical validation. The unresolved
paper sensitivity denominator/aggregation discrepancy remains documented.

All 62 unit tests passed after the runtime-pin change; official CLI request
serialization passed with mocked network. Real saved inference artifacts then
passed the independent verification above. The next milestone is the progress
report/presentation; preserve this frozen result when planning a separate
extension using training/validation data. See [UNet++ results](UNETPP_RESULTS.md).

## Template for subsequent runs

## Experiment ID

TODO — evidence pending.

## Date

TODO — evidence pending.

## Research question / hypothesis

TODO — evidence pending.

## Dataset split and manifest hash

TODO — evidence pending.

## Model

TODO — evidence pending.

## Preprocessing

TODO — evidence pending.

## Changed variable

TODO — evidence pending.

## Fixed variables

TODO — evidence pending.

## Random seed

TODO — evidence pending.

## Training configuration and environment

TODO — evidence pending.

## Metrics and decision threshold

TODO — evidence pending.

## Results

TODO — evidence pending.

## Failure modes

TODO — evidence pending.

## Interpretation

TODO — evidence pending.

## Limitations

TODO — evidence pending.

## Next action

TODO — evidence pending.

## Git commit / tag

TODO — evidence pending.
