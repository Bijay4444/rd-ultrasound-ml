# R3D-18 validation Grad-CAM extension

Status, 2026-11-22: the checked extension was submitted as private Kaggle job
`bijaykhadka1710/rd-gradcam-r3d18-validation-001/1`. Saved-output verification
and the real three-model demo are pending completion. No new training or
official-test evaluation is part of this workflow.

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

## Interpretation

These are coarse spatial/temporal score-attribution maps, not verified anatomical
masks. Separate map normalization prevents cross-model brightness comparisons.
The eight examples are outcome-stratified, not a representative performance
sample. Occlusion/randomization controls provide sensitivity evidence rather
than proof of anatomical correctness. Related patients may appear in different
official clip splits; one training seed and educational research scope remain
limitations. Use the separate four-model benchmark table for performance claims.
