# UNet++ RD results

Updated 2026-11-02. All **50 training epochs are complete** for
`unetpp_rd_002_amp`. Frozen epoch-48 evaluation on all **1,077 official test
clips is complete and independently verified**. This completes one adapted,
code-aligned UNet++ RD benchmark; the protocol differences below remain part of
the result.

## Official-test results and published comparison

The selected checkpoint and threshold 0.5 were fixed using validation before
this evaluation. RD is the positive class. The test confusion counts are
**TP=86, TN=964, FP=13, FN=14**: 14 of the 100 RD clips were missed.

| Metric | Our official test | Paper, Table 8 | Difference, percentage points |
| --- | ---: | ---: | ---: |
| Sensitivity | 86.00% | 87.9% | -1.90 |
| Specificity | 98.67% | 98.6% | +0.07 |
| Precision | 86.87% | 87.1% | -0.23 |
| F1 | 86.43% | 87.5% | -1.07 |
| Accuracy | 97.49% | 97.6% | -0.11 |

Published values are the **Non-RD vs. RD, UNet++** test row in
[ERDES, Table 8](https://arxiv.org/html/2508.04735v2). Differences use unrounded
local metrics and the displayed paper values. Other local test metrics:
balanced accuracy 0.923347, average precision 0.926845, ROC-AUC 0.986858 and
BCE loss 0.081308. An always-non-RD classifier would have 90.71% accuracy and
zero sensitivity on this partition.

The accuracy is close to the reported row, with lower RD sensitivity. This is
one seed on the official clip-level split, with adapted numerical execution;
it does not establish exact numerical replication, statistical equivalence,
unseen-patient generalization or clinical validity. The source audit also records
an unresolved paper sensitivity denominator/aggregation discrepancy.

Evidence: private Kaggle job `bijaykhadka1710/unetpp-rd-002-test-evaluation/1`
completed successfully. Inference took **31.07 seconds**; the saved job took
about 270.22 seconds including checks and export. The GPU runtime exactly
matched training. The downloaded archive is retained at
`outputs/kaggle_results/test_evaluation/unetpp_rd_002_test_results.zip`.
Independent verification checked all **12 artifact hashes**, all **1,077**
prediction rows against official membership/order/labels, thresholded labels,
all reported classification metrics, and the frozen epoch/checkpoint identity.
Audit evidence is in that directory's `verification.json` and
`verified_comparison.csv`. Archive SHA256:

```text
9834929ed281d7cf03a978de4b977c147811fb2f65fbf92be9693970fb5c3e03
```

## Completed training and frozen selection

The released encoder-classifier equivalent (3,603,553 parameters) trained from
scratch on all 3,873 official training clips, with 431 validation clips. Input:
full-video grayscale/square-padding/trilinear interpolation to `[1,96,128,128]`,
float32 cache, no augmentation. Fixed threshold 0.5. Adam at 1.5e-5, unweighted
BCE, seed 42, effective batch 48 through physical batch 4 and accumulation 12,
50 epochs and a validation-loss ReduceLROnPlateau scheduler. GPU execution used
FP16 autocast and FP32 master weights. Only CUDA 0 was used on T4 x2 allocations.

The fixed rule selects highest validation accuracy and earliest ties. **Epoch 48**
was selected; epoch 49 tied and did not replace it. Epoch 50 remains the last
training checkpoint, not the selected model.

| Selected epoch-48 validation metric | Value |
| --- | ---: |
| TP / TN / FP / FN | 37 / 389 / 2 / 3 |
| Sensitivity | 0.925000 |
| Specificity | 0.994885 |
| Precision | 0.948718 |
| F1 | 0.936709 |
| Accuracy | 0.988399 |
| Balanced accuracy | 0.959942 |
| Average precision | 0.973139 |
| ROC-AUC | 0.996164 |
| Validation BCE | 0.058425 |

These are **validation** measurements. They are not compared to the article's
**test** row as an equivalent benchmark result. A single clip-level split and
seed do not establish unseen-patient generalization or clinical validity.

Final training job 03's saved log ends around 14,058 seconds (3.91 hours),
including input validation and export. Earlier jobs completed epoch 1 and epochs
2–11. Across training, the AMP scaler recorded **one skipped optimizer update in
epoch 35**. The audit reconciles 4,050 attempted updates with 4,049 Adam steps;
this is disclosed rather than silently claiming zero overflows. Final model
weights are finite. The method differs numerically from reference FP32 training.

Evidence: private Kaggle job
`bijaykhadka1710/unetpp-rd-002-train-03`, version 1. The agent downloaded the final
ZIP to ignored `outputs/kaggle_results/session_03/`, verified **all 70 artifact
hashes**, safely extracted it under `verified/`, recomputed all 50 epochs' metrics
from saved predictions, and checked the selected best weights against the
checkpoint. `verification.json` records the audit.

## Freeze before official-test access

[The tracked freeze](../configs/unetpp_rd_002_evaluation.json) fixes epoch 48,
threshold 0.5, checkpoint SHA256, training-config digest, data fingerprint,
source revisions, preprocessing, precision and batch size. It was written before
the official test CSV was first opened by the evaluation workflow. Best checkpoint:

```text
c9dbbd79124dfcc692e7cf3a4a919eb3da7a9533153f9e3b5db39f9165123ca8
```

The freeze-before-access statement refers to this new evaluation workflow. The
project's earlier `split_audit.ipynb` had already read test membership and class
counts for dataset-integrity checks; no model test scores were used for training
or selection. The freeze flag does not deny that earlier metadata audit.

Only after this freeze, the evaluation workflow's split audit confirmed **1,077 official test clips:
977 non-RD and 100 RD**, with no clip overlap with train/validation and labels
matching metadata. No model inference was performed during that audit. Possible
patient overlap in the supplied clip-level splits remains a limitation.

## Official-test workflow

1. `kaggle_unetpp_test_preparation.ipynb`: private CPU job with Internet enabled;
   attach final training version 1; verify the freeze before reading test.csv;
   prepare all test inputs with the unchanged decoder/method/version identity.
   About 6.78 GB of float32 inputs are saved; raw temporary videos are excluded.
2. `kaggle_unetpp_evaluation.ipynb`: private GPU job with Internet disabled;
   attach the saved CPU test output and final training output, both version-pinned.
   Verify hashes/manifest order, load only the frozen best weights, use eval mode
   and inference_mode, preserve model state, and produce one probability per clip.
3. Preserve test predictions, confusion counts, sensitivity/specificity/precision/
   F1/accuracy, AP/ROC-AUC, loss, runtime, freeze and hashes. Compare the **test**
   metrics to the author-reported row, preserving all protocol differences.

No optimizer, additional training, calibration fit, threshold search or checkpoint
selection is performed on test data. An existing result in the same evaluation
output directory stops a repeated evaluation. The training job and its frozen
configuration remain unchanged.

The paper's UNet++ row reports sensitivity 0.879, specificity 0.986, precision
0.871, F1 0.875 and accuracy 0.976. These are author-reported values, not our
results. The source audit records the AdamW-in-text versus Adam-in-code conflict
and the unresolved reported-sensitivity denominator/aggregation discrepancy.
See [the primary-source audit](research/erdes-unetpp-protocol.md) and the
[article, Table 8](https://arxiv.org/html/2508.04735v2).

## How to run from VS Code

The CPU test job `bijaykhadka1710/unetpp-rd-002-test-inputs`, version 1, is now
complete, as is GPU evaluation version 1. **Do not submit another copy for this
result.** The following submission commands are retained for reproducibility:

```bash
python3 scripts/prepare_unetpp_evaluation_job.py evaluate
python3 scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/unetpp-rd-002-test-evaluation --accelerator NvidiaTeslaT4
```

The existing CLI guard refuses GPU execution until all required input jobs
report COMPLETE. The small result ZIP has already been retrieved with:

```bash
python3 scripts/kaggle_cli.py kernels output bijaykhadka1710/unetpp-rd-002-test-evaluation/1 --file-pattern '.*_test_results\.zip$' --page-size 200 -p outputs/kaggle_results/test_evaluation
```

The single-model test-benchmark evaluation is complete. The next milestone is
the progress report/presentation using the verified comparison above. A controlled
extension, such as class weighting, needs a separate documented experiment whose
choices use training/validation data. Preserve this frozen reference result;
do not tune against its test predictions or present future reuse of this exposed
test partition as a fresh untouched holdout.

## Checks and current limits

All **62 unit tests passed** in the existing temporary uv CPU environment.
New checks exercise the pre-test freeze boundary, changed-weight rejection,
held-out membership/labels/order, actual fixed-weight inference with no optimizer,
unchanged model state, cache corruption, version-pinned private bundles and
notebook/config/source agreement. Both new notebooks validate with nbformat.
A synthetic integration executes their standalone embedded modules and actual
inference/export cells and verifies ZIP hashes; this is not real test performance.

Commands: temporary uv execution of `/tmp/freeze_unetpp_final.py`,
`python -m unittest discover -s tests -v`,
`/tmp/verify_unetpp_evaluation_notebooks.py`, the freeze-gated real split audit,
and `git diff --check`. Detailed synthetic evidence is ignored under
`outputs/unetpp_verification/evaluation_notebook_check.json`.

Official-test CPU preparation and GPU inference are complete. Saved predictions
and metrics passed independent verification. No further training, calibration,
threshold tuning or checkpoint selection occurred during official-test evaluation.


## 2026-11-02 — Test preparation saved; inference container pinned

The user supplied a successful test CPU preparation log: all 1,077 clips were
prepared in 2,656.1 seconds (44.27 minutes), 6.78 GB saved output, Accelerator None.
The checkpoint hash was verified before test preparation. The saved job version
`unetpp-rd-002-test-inputs/1` was independently confirmed COMPLETE.

That CPU job reports Python 3.13.18 and huggingface_hub 1.33.0; the training GPU
job used Python 3.13.15 and huggingface_hub 1.29.0. Core preprocessing versions
(torch 2.11.0+cpu, numpy 2.1.3, OpenCV 4.14.0) matched the original CPU-cache
identity, so its unchanged preprocessing guard passed. To preserve the stricter
GPU runtime guard, the evaluation bundle now requests the exact container URI
retrieved from the final training notebook's version-1 metadata. See
`configs/unetpp_rd_002_runtime.json` for the original container digest and runtime.
This adds no training or test-driven model changes.

The prior unsubmitted evaluation bundle was archived under ignored diagnostics
before regenerating its metadata. Tests confirm GPU bundles retain the original
container pin, while CPU bundles do not receive a GPU image override. An official
CLI request-construction check (network mocked) also verified that the Docker pin,
private access, disabled Internet and version-pinned input references are passed
to Kaggle. The guarded CLI verified both required version-pinned inputs COMPLETE
and submitted evaluation version 1. The saved job subsequently completed, its
metadata retained the requested training image digest, and its recorded runtime
matched training exactly. The verified official-test results appear above.
