# R3D-18 validation Grad-CAM extension

Completed 2026-11-22 (local date). Private Kaggle job
`bijaykhadka1710/rd-gradcam-r3d18-validation-001/1` completed all eight cases and
passed the saved-output audit. All **127 new artifact hashes** and the historical
225 hashes match. All **40 attribution arrays** (two targets and three
randomization stages per case) passed shape, finite-value and applicable
normalization checks. No new training or official-test evaluation occurred.

Archive SHA256:
`6fb4ac07dac3304846ad95d8be6b8b3db3dc0db894a17c95622ef5fc2f08afce`.
Ignored archive and audit: `outputs/kaggle_results/gradcam_r3d18_validation_001/`,
with `audit/verification.json` and checked extraction under `audit/verified/`.

## Verified findings

- Exact historical-runtime pin: Tesla T4, Python 3.13.16, torch 2.11.0+cu130,
  torchvision 0.26.0+cu130, CUDA 13.0. Total 1,035.77 seconds (17.26 minutes),
  including two full cache-integrity passes; this is not forward-only timing.
- All eight original-precision probabilities exactly match saved validation
  probabilities. Maximum FP32-versus-original probability difference is 0.000875.
  Displayed explanation probabilities are explicitly FP32.
- Final feature shape `[1,512,12,8,8]`; maximum measured allocated GPU memory
  1,627,598,848 bytes (1.63 GB). Model state/prediction preservation gates passed.
- RD attribution is constant zero on `164267_00003`, `164267_00005` and
  `164267_00944`. High/low occlusion ranking is uninformative on these maps.
- Among the five nonconstant RD maps, masking high-attribution regions produces
  a greater mean RD-logit drop than low-attribution masking in **5/5** cases and
  greater than random masking in **4/5**. On `755384_00001`, random exceeds high.
  These are descriptive controls on a small outcome-stratified panel.
- For `405744_00014`, cumulative randomization correlations are -0.470 after
  head reset, -0.352 after head plus final block, and -0.343 after all learned
  layers. All control outputs are retained, including undefined correlations
  for constant maps; no single statistic certifies anatomical correctness.

| Clip | Dataset label | FP32 RD score | Outcome at 0.5 |
| --- | --- | ---: | --- |
| 102769_00005 | RD | 39.13% | Missed RD |
| 102769_00009 | RD | 98.71% | Correct RD |
| 164267_00003 | non-RD | 1.64% | Correct non-RD |
| 164267_00005 | non-RD | 2.20% | Correct non-RD |
| 164267_00944 | non-RD | 1.19% | Correct non-RD |
| 164267_01736 | non-RD | 45.73% | Correct non-RD |
| 405744_00014 | RD | 86.80% | Correct RD |
| 755384_00001 | RD | 2.76% | Missed RD |

The first audit incorrectly required `1e-12` agreement for a saved float32
mean-absolute-difference reduction. NumPy 2.5.4 locally versus 2.1.3 on Kaggle
gave differences up to one float32 epsilon (1.19e-7). The auditor now accumulates
the same float32 differences in float64 and allows two float32 epsilons
(2.38e-7) for that saved statistic. Correlations retain `1e-12` tolerance; file
and source hashes remain exact. No saved array, control, model or training
implementation was changed.

## Frozen model and original examples

The extension explains R3D-18 epoch **18** at `backbone.layer4`, with head
`backbone.fc`. It uses the exact verified
`configs/r3d18_rd_001_evaluation.json`, selected checkpoint SHA256
`5422add5a8ce4544e4c9f6bf0190fa57702520bed55caea5738398ac7d363152`,
and `namanbabariya/r3d18-rd-001-train-03/1`.

The original eight validation cases must match the baseline-derived selection
before explanation inference: `102769_00005`, `102769_00009`, `164267_00003`,
`164267_00005`, `164267_00944`, `164267_01736`, `405744_00014`, `755384_00001`.
The fixed plan is `configs/r3d18_gradcam_validation_001.json`. It pins the
historical two-model archive SHA256
`1ea45de9e39b4243958c8acef1a62aac56ede3ed9083ac5fdaac6e7f0c37d335`.
No examples are chosen by heatmap appearance or new test results.

U-Net++ epoch 48 supplies the case-selection reference. Both its inputs and the
R3D run/cache identities are verified, then only R3D-18 is instantiated for
explanation inference. Existing U-Net++ and R(2+1)D-18 explanation arrays,
controls and display media are reused from their completed archive unchanged.

## Same method, separate output

The new job pins the historical Grad-CAM Docker image:
`gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01`.
It attaches the original two training/validation preparation outputs (version 2),
original baseline training output (version 1) and frozen R3D training output
(version 1). Internet is disabled; test-preparation outputs are not attached.

FP32, batch-one Grad-CAM targets RD logit `z` and non-RD evidence `-z` at full
`[1,96,128,128]` input shape. Original-precision prediction parity with saved
validation predictions must stay within 0.001 before producing maps. Constant
maps are flagged. Cuboid occlusion (12×16×16, eight per group, seed 42) and
cumulative head/final-block/all-layer randomization retain the original
algorithm. No optimizer, training or automatic resolution/precision fallback is
used. Model state is checked after explanations and controls.

`run_explanations` now accepts a verified subset of models to explain and an
expected historical case manifest. Defaults preserve the original two-model
workflow. New subset/case-drift tests guard these paths. The new output identity
is `gradcam_r3d18_validation_001`, so historical artifacts are not overwritten.

## Files and reproducible commands

- Builder: `scripts/build_r3d18_explainability_notebook.py`.
- Standalone notebook: `notebooks/kaggle_r3d18_explainability.ipynb`.
- Private bundle: `scripts/prepare_r3d18_explainability_job.py`.
- Saved-result verifier: `scripts/verify_r3d18_explanations.py`.
- Portable viewer: `scripts/build_explainability_demo.py`, with
  `--additional-archive <verified-R3D-archive>` after the historical archive.

The builder and bundle commands run through uv. CPU checks use the existing
isolated environment in [EXPLAINABILITY_RESULTS.md](EXPLAINABILITY_RESULTS.md),
with `python -m unittest discover -s tests -p '*explainability*.py' -v`.
No project dependencies or system Python packages were changed.

The saved-output audit verifies every listed hash, exact source/freeze/config,
ordered case membership, original prediction parity, finite full-size attribution
arrays and normalization, fixed occlusion arithmetic and randomized-map
similarity. It does not rerun model inference. The viewer checks case/method/
cache/algorithm equality before combining archives, and preserves both sources'
hashes in its provenance record. Large generated artifacts remain ignored.

Commands run (with the documented uv prefix for the NumPy verifier):

```bash
python scripts/verify_r3d18_explanations.py --archive outputs/kaggle_results/gradcam_r3d18_validation_001/gradcam_r3d18_validation_001.zip --output outputs/kaggle_results/gradcam_r3d18_validation_001/audit
python scripts/build_explainability_demo.py --archive outputs/kaggle_results/gradcam_validation_001/gradcam_validation_001.zip --additional-archive outputs/kaggle_results/gradcam_r3d18_validation_001/gradcam_r3d18_validation_001.zip --output outputs/presentation/gradcam_demo_three_models
```

All **22 explanation/viewer CPU tests** passed, including the original 14,
five R3D extension checks and three archive/viewer checks. Python compilation
and `git diff --check` passed. The Impeccable detector reported no findings.
The unrelated repository suite was not rerun. Real-output browser checks are
documented in [EXPLAINABILITY_DEMO.md](EXPLAINABILITY_DEMO.md).

## Interpretation

These are coarse spatial/temporal score-attribution maps, not verified anatomical
masks. Separate map normalization prevents cross-model brightness comparisons.
The eight examples are outcome-stratified, not a representative performance
sample. Occlusion/randomization controls provide sensitivity evidence rather
than proof of anatomical correctness. Related patients may appear in different
official clip splits; one training seed and educational research scope remain
limitations. Use the separate four-model benchmark table for performance claims.
