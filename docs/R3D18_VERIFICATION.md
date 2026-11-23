# R3D-18 saved-artifact verification

Completed 2026-11-22. The supplied [Kaggle test notebook](https://www.kaggle.com/code/namanbabariya/r3d18-rd-001-test-evaluation)
is the completed R3D-18 experiment. It is distinct from R(2+1)D-18 and from a
2D ResNet. Its code is already on the remote branches `feature/r3d18-rd` and
`feature/r3d18-training-eval`; a merge into `dev` is not required to verify it.
No branch was checked out or merged during this audit.

## Identity and freeze

The inspected branch was `origin/feature/r3d18-training-eval` at
`b08803bfe7f4bf8b2508d2d6f747e9df28b74d98`. Its evaluation config exactly
matches both the Kaggle notebook's freeze and the archived evaluation freeze.
That record confirms the team's selection on 2026-11-18: **epoch 18, threshold
0.5**, after 50 epochs, using maximum validation accuracy and earliest ties.

Version-pinned outputs inspected:

- `namanbabariya/r3d18-rd-001-train-03/1`
- `namanbabariya/r3d18-rd-001-test-evaluation/1`
- Reused input cache: `bijaykhadka1710/unetpp-rd-002-test-inputs/1`

Selected checkpoint SHA256:
`5422add5a8ce4544e4c9f6bf0190fa57702520bed55caea5738398ac7d363152`.
Test archive SHA256:
`87c687a84024a8bd8b1a96f98a6a3b183fe373e2a8dd984658e120bd932cc751`.

The factory builds torchvision `r3d_18(weights=None, num_classes=1)`, replacing
the first convolution with a one-channel convolution and retaining BatchNorm.
The inspected FP32 checkpoint contains **33,147,969 parameters**. This is the
standard 18-layer video classifier, not the deeper MONAI ResNet in ERDES.

## Checks and limits

The independent saved-artifact audit passed 7,403 assertions. All 16 test
artifact hashes and 77 downloaded training artifact hashes match. The two
rotating resume slots, `last_a.pt` and `last_b.pt`, were not downloaded or
verified. The selected `best.pt` was verified separately.

All seven frozen training modules match branch snapshots; evaluation source,
resolved config, run fingerprint, cache identities and runtime match the freeze.
Original metadata/split checksums and train/validation/test manifests were
checked. All 50 sets of 431 validation scores reproduce the saved classification
and ranking metrics. The selected epoch is independently confirmed. All 1,077
test rows match official order, paths, labels and threshold decisions; grouped
average precision and rank-based ROC-AUC were recomputed independently of the
training metric function. The saved log records freeze verification before test
inference. This audit performed no new model inference or training.

`torch.load(weights_only=True, map_location='cpu')` confirmed checkpoint epoch,
config digest, fingerprint, `test_used=False`, finite state tensors, grayscale
stem and binary head. Saved epoch training/validation times sum to 9.47 hours;
this excludes preparation, probes and job/export overhead. One skipped AMP
update was recorded. Local GPU inference was not repeated.

Ignored evidence lives in `outputs/kaggle_peer_audit/r3d18/`: original downloads,
read-only branch snapshots with `branch_provenance.json`,
`verify_saved_artifacts.py`, `verification.json`,
`verify_checkpoint_payload.py`, `checkpoint_payload_verification.json`, and
hash-checked flat test exports under `verified_test/`.

## Verified measurements

Validation: 431 clips, including 40 RD. Reused official test benchmark: 1,077
clips, including 100 RD. Validation was used for checkpoint selection.

| Metric | Validation | Reused official test benchmark |
| --- | ---: | ---: |
| TP / TN / FP / FN | 37 / 391 / 0 / 3 | 89 / 966 / 11 / 11 |
| Sensitivity | 92.50% | 89.00% |
| Specificity | 100.00% | 98.87% |
| Precision | 100.00% | 89.00% |
| F1 | 96.10% | 89.00% |
| Accuracy | 99.30% | 97.96% |
| Balanced accuracy | 96.25% | 93.94% |
| Average precision | 0.969428 | 0.961930 |
| ROC-AUC | 0.989450 | 0.993014 |

The measured test forward time excluding the first batch is 39.32 ms/clip on a
Tesla T4, physical batch 4, FP16 autocast. This excludes disk loading. BatchNorm,
physical batch and runtime differ from some other architectures; the experiment
compares architectures under the inherited recipe, not optimized performance.
The R3D and R(2+1)D runs use CUDA 13.0; the original and augmented U-Net++ use
CUDA 12.8. The U-Net++ augmentation ablation matches its baseline runtime.

One seed, possible related patients across official clip splits, and reuse of
an already exposed test benchmark limit interpretation. These are educational
research findings, not clinical validation or unseen-patient performance.

## Commands

Remote references were refreshed with `git fetch origin`; branch files were
read with `git show origin/feature/r3d18-training-eval:<path>`. The Kaggle CLI
wrapper pulled source and downloaded exact version-1 outputs using
`kernels pull` and `kernels output`. The two audit scripts above ran through uv,
using standard-library metrics and CPU PyTorch for safe checkpoint inspection.
No branch integration tests were run because the branch was not merged; this
was a provenance and saved-results audit rather than a complete code review.
