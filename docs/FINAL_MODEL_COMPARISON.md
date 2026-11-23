# Frozen four-model comparison

Verified 2026-11-22. All four 50-epoch experiments now have selected-checkpoint
validation measurements and saved official-test predictions. The official test
was already exposed earlier in the project; these results are a **reused
official test benchmark**, not a new untouched holdout.

## Frozen choices and provenance

Freeze means fixing the entire experiment list and each model's architecture,
training config, augmentation policy, checkpoint-selection rule, selected
weights and decision threshold before new test inference. It applies to **all
four experiments**, not only U-Net++. A GitHub merge can happen afterward if
the saved code/config/checkpoint identities can already be verified.

| Model | Training | Selected epoch | Threshold | Checkpoint evidence |
| --- | ---: | ---: | ---: | --- |
| U-Net++ | 50 epochs | 48 | 0.5 | [Baseline report](UNETPP_RESULTS.md) |
| R3D-18 | 50 epochs | 18 | 0.5 | [Independent audit](R3D18_VERIFICATION.md) |
| R(2+1)D-18 | 50 epochs | 15 | 0.5 | [Independent audit](R2PLUS1D18_VERIFICATION.md) |
| U-Net++ + augmentation | 50 epochs | 32 | 0.5 | [Augmentation report](UNETPP_AUGMENTATION_RESULTS.md) |

Selection was maximum validation accuracy with earliest ties for every model.
The augmentation policy was fixed from training-only previews before its run.
On November 22 Bijay reported the other choices frozen, with the peer handoff
still uncertain. Inspection found matching R3D-18 Kaggle and GitHub freezes,
explicitly confirmed by that peer on November 18. That resolved the missing
handoff; the augmentation evaluation config records this evidence before its
single test submission. No choices were changed in response to that result.

R3D-18 source is on `origin/feature/r3d18-training-eval` at
`b08803bfe7f4bf8b2508d2d6f747e9df28b74d98`, with an additional
`origin/feature/r3d18-rd` branch. Its archived source and freeze match the inspected
branch. Neither branch was merged or checked out during verification; the
existing working tree was preserved. This is R3D-18, distinct from R(2+1)D-18
and a 2D ResNet.

## Validation, used for selection

Same 431 clips: 40 RD and 391 non-RD. All measurements use threshold 0.5.

| Model | TP / TN / FP / FN | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced accuracy | AP | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net++ | 37 / 389 / 2 / 3 | 92.50% | 99.49% | 94.87% | 93.67% | 98.84% | 95.99% | 0.973139 | 0.996164 |
| R3D-18 | 37 / 391 / 0 / 3 | 92.50% | 100.00% | 100.00% | 96.10% | 99.30% | 96.25% | 0.969428 | 0.989450 |
| R(2+1)D-18 | 38 / 389 / 2 / 2 | 95.00% | 99.49% | 95.00% | 95.00% | 99.07% | 97.24% | 0.979523 | 0.996995 |
| U-Net++ + augmentation | 36 / 388 / 3 / 4 | 90.00% | 99.23% | 92.31% | 91.14% | 98.38% | 94.62% | 0.953729 | 0.990249 |

Augmentation adds one false positive and one false negative compared with
baseline on validation; it corrects no baseline error at the fixed threshold.
This policy did not improve the validation result.

## Reused official test benchmark

Same 1,077 clips: 100 RD and 977 non-RD. Existing baseline and peer predictions
were reused without another forward pass. Only the newly frozen augmentation
model needed its separate single evaluation.

| Model | TP / TN / FP / FN | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced accuracy | AP | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net++ | 86 / 964 / 13 / 14 | 86.00% | 98.67% | 86.87% | 86.43% | 97.49% | 92.33% | 0.926845 | 0.986858 |
| R3D-18 | 89 / 966 / 11 / 11 | 89.00% | 98.87% | 89.00% | 89.00% | 97.96% | 93.94% | 0.961930 | 0.993014 |
| R(2+1)D-18 | 92 / 962 / 15 / 8 | 92.00% | 98.46% | 85.98% | 88.89% | 97.86% | 95.23% | 0.958854 | 0.993321 |
| U-Net++ + augmentation | 87 / 967 / 10 / 13 | 87.00% | 98.98% | 89.69% | 88.32% | 97.86% | 92.99% | 0.918940 | 0.979591 |

R(2+1)D-18 detects the most RD clips at the fixed threshold, with the most false
positives. R3D-18 has the highest observed accuracy and average precision.
Augmented U-Net++ has the fewest false positives. These are descriptive metric
tradeoffs on this benchmark, not evidence of a universally best model.

Relative to baseline, augmentation has one fewer missed RD clip and three fewer
false positives, but lower AP and ROC-AUC. In the paired predictions, eight
baseline errors become correct and four previously correct clips become wrong;
both models are wrong on 19 and correct on 1,046. The modest fixed-threshold
change, adverse validation result and worse ranking metrics support a **mixed
augmentation finding**, not a general benefit claim. AP uses grouped-score
average precision, not trapezoidal PR-AUC. No threshold optimization followed.

## Verification and artifacts

Exact test output versions:

- [Original U-Net++](https://www.kaggle.com/code/bijaykhadka1710/unetpp-rd-002-test-evaluation): `/1`.
- [R3D-18](https://www.kaggle.com/code/namanbabariya/r3d18-rd-001-test-evaluation): `/1`.
- [R(2+1)D-18](https://www.kaggle.com/code/namanbabariya/r2plus1d18-rd-001-test-evaluation): `/1`.
- [Augmented U-Net++](https://www.kaggle.com/code/bijaykhadka1710/unetpp-rd-aug-001-test-evaluation): `/1`.

`scripts/compare_frozen_benchmarks.py` verified all 12 / 16 / 16 / 17 listed
test artifact hashes respectively, exact local/snapshot freezes and source,
weights identities, runtime records, clean shared test-cache identities,
official split checksums, all 1,077 ordered prediction rows per model and all
recomputed metrics. The new augmentation inference used the exact original
U-Net++ Docker/runtime, with no training, augmentation or threshold search.
Its archive SHA256 is
`04fbd794cd0fe123789207e1af1525025d677bbce89c217f7471cee7d4ccf386`.

Ignored outputs under `outputs/augmentation/four_model_comparison_20261122/`:
`verification.json`, `reused_test_metrics.csv`, `paired_test_predictions.csv`,
`test_confusion_matrices.png/pdf` and `test_roc_precision_recall.png/pdf`.
The PNG figures were visually inspected. Augmentation learning curves and
validation figures are in `outputs/augmentation/validation_comparison_20261122/`.
Original archives, model weights and videos remain ignored and are not committed.

Reproduce saved-results verification and plotting with the uv environment in
[the augmentation report](UNETPP_AUGMENTATION_RESULTS.md):

```bash
python scripts/compare_frozen_benchmarks.py --augmentation-archive outputs/augmentation/test_evaluation_v1/unetpp_rd_aug_001_test_results.zip --output outputs/augmentation/new_comparison
```

The command requires the documented downloaded archives and R3D branch snapshot;
it does not download data, run models, alter freezes or overwrite a result folder.
The code snapshot can be reconstructed with `git show <recorded-commit>:<path>`.

## Scope and limits

All runs used seed 42, the official clip splits, scratch initialization,
`[1,96,128,128]` inputs, physical batch 4/effective batch 48 and the inherited
Adam/BCE/scheduler recipe. This compares architectures under one inherited
recipe, not each model's best tuned performance. One seed limits robustness
claims, and possible patient overlap across official clip splits prevents an
unseen-patient generalization claim.

The baseline and augmented U-Net++ have 3,603,553 parameters and InstanceNorm.
R3D-18 has 33,147,969 and R(2+1)D-18 has 31,296,228 parameters, with BatchNorm.
Gradient accumulation does not increase BatchNorm's physical batch. Both peers
used CUDA 13.0, while the U-Net++ pair used CUDA 12.8; cross-architecture
execution was not fully controlled. The augmentation pair matched runtime.

The independent audits do not rerun GPU training or establish patient-level or
clinical validity. Branch integration and its full test suite remain separate
engineering work; no merge was needed for this comparison. Future changes
informed by these results require a separately documented development protocol.
