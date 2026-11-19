# Contributions

Who did what, with pointers to the evidence (commits in this repository and Kaggle runs). Kaggle runs are
private; they can be shared with the instructor on request.

## Bijay Khadka

- Project setup, metadata EDA, split audit and data-access documentation.
- Video preparation pipeline and the preprocessing used by every model (cached, version-pinned inputs
  `bijaykhadka1710/unetpp-rd-002-inputs-00/2` and `-01/2`).
- U-Net++ reference model: implementation, training, results and presentation material.
- Shared Kaggle tooling (`scripts/kaggle_cli.py`, resumable multi-session training) and the test-input cache
  (`bijaykhadka1710/unetpp-rd-002-test-inputs/1`) reused by all comparison models.
- Model comparison protocol, augmentation/explainability plans and the model handoff assignments.
- Unit test suites for preparation, baseline training, U-Net++ and its Kaggle/evaluation pipeline.

## Naman Babariya

- **R(2+1)D-18 comparison model** (branch `feature/r2plus1d18-rd`, merged in PR #17): implementation, Kaggle
  training pipeline and tests (`4bead30`); 50-epoch GPU training over four Kaggle sessions
  (`namanbabariya/r2plus1d18-rd-001-train-01`…`-04`, about 22.6 GPU hours); results and validation error analysis
  (`378b09d`, `6bf04bc`, `c4127f2`); gated frozen test evaluation (`70f32af`, `e5ee644`) and the single verified
  test run (`namanbabariya/r2plus1d18-rd-001-test-evaluation`, results in `8609441`).
- **R3D-18 training and testing** (branch `feature/r3d18-rd`): ran the 50-epoch GPU training on Kaggle over three
  sessions (`namanbabariya/r3d18-rd-001-train-01`…`-03`, about 9.5 GPU hours), including the memory probe and
  session sizing; recorded the validation results (`210123c`); built the gated frozen test evaluation and its tests
  (`cbc8986`); after the team confirmed the freeze (`8a29a6e`), ran and verified the single test run
  (`namanbabariya/r3d18-rd-001-test-evaluation`) and recorded the results with paired comparisons against U-Net++
  and R(2+1)D-18 (`3ff50f0`).
- Shared, model-agnostic comparison code (`video_cache.py`, `comparison_evaluation.py`) used by both comparison
  models.

## Ritu

- **R3D-18 comparison model** (branch `feature/r3d18-rd`): the one-channel R3D-18 model with its Grad-CAM layer
  (`dfd9805`), training configuration (`9bf383a`), Kaggle training notebook, builder and job script (`327bd9e`,
  `e95ee38`), unit tests (`f6b0965`) and the initial results report (`82828fd`). The GPU training used this
  notebook unchanged.

## Shared

- Model choices for each comparison model (selected epoch, threshold 0.5) were frozen by team confirmation before
  any test-split access.
- Each person's results reports are under `docs/` (`UNETPP_RESULTS.md`, `R2PLUS1D18_RESULTS.md`,
  `R3D18_RESULTS.md`).
