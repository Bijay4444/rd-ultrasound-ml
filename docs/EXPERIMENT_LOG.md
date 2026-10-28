# Experiment Log

The first 256-clip Colab GPU pilot is complete. Record subsequent runs using the
template below. Validation findings are development results; the test set remains reserved.

Current preparation: [baseline_001 proposal](BASELINE_PLAN.md) and
`configs/baseline_001.json` record the candidate GPU pilot. Input preparation
passed on eight inspection clips locally and in Colab. The GPU pilot is implemented
in `notebooks/baseline_training.ipynb`, using `configs/baseline_001_training.json`.
Its next sections audit the saved predictions and implement the controlled
`baseline_002_weighted_pilot` comparison using `configs/baseline_002_weighted.json`.
The weighted Colab run has not been performed by the agent.

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

## baseline_002_weighted_pilot: planned controlled comparison

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

Results: pending the user's Colab execution. The local synthetic checks are
implementation evidence, not ERDES results.

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
