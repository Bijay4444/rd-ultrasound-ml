# Frozen-model explainability

The original two-model run below is **completed and downloaded** on all eight fixed
validation clips. All 225 saved artifact hashes were independently checked from
the results ZIP. No training or final-test evaluation was performed.

Update, 2026-11-22: frozen R3D-18 epoch 18 also completed the same eight cases.
Its 127 new artifact hashes and 40 attribution arrays passed the separate
[extension audit](R3D18_EXPLAINABILITY_RESULTS.md). The default demo now displays
all three models with the original two-model outputs reused unchanged.

Private Kaggle version 1 was submitted successfully at
[RD Grad-CAM Validation 001](https://www.kaggle.com/code/bijaykhadka1710/rd-gradcam-validation-001).
All four pinned sources passed the remote readiness check. Remote status and
internal `status.json` confirm completion of all 16 model/case pairs.

Use the [classroom demo guide](EXPLAINABILITY_DEMO.md) for the offline comparison
page and speaking script under ignored `outputs/presentation/gradcam_demo.zip`.

## Verified execution findings

- Tesla T4, Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130;
  1,085.7 seconds (about 18.1 minutes), including full cache verification.
- Largest original-precision versus historical-probability differences:
  U-Net++ 0.000139; R(2+1)D-18 0.000147. Both meet the 0.001 parity gate.
- FP32 versus original-precision differences reached 0.001479 and 0.001087,
  respectively. Explanation scores are explicitly FP32.
- Recorded peak allocated memory: U-Net++ 1.11 GB; R(2+1)D-18 4.23 GB. Feature
  shapes were `[1,256,6,8,8]` and `[1,512,12,8,8]`, respectively.
- RD attribution was constant on one U-Net++ case and two R(2+1)D-18 cases;
  these maps do not support a meaningful high/low occlusion ranking.
- Among nonconstant maps, high-attribution masking had a greater mean RD-logit
  drop than low-attribution masking on 2/7 U-Net++ cases and 6/6 R(2+1)D-18 cases;
  it exceeded random masking on 1/7 and 5/6. These descriptive observations on
  outcome-stratified examples do not establish general explanation fidelity.
- Randomization outputs are retained at every stage. For R(2+1)D-18 on
  `405744_00014`, map correlations were 0.849 after head reset, -0.079 after head
  plus final block, and -0.145 after all learned layers. No single value certifies
  an explanation.

ZIP SHA256: `1ea45de9e39b4243958c8acef1a62aac56ede3ed9083ac5fdaac6e7f0c37d335`.
Independent integrity record:
`outputs/kaggle_results/gradcam_validation_001/verification.json`.
Extracted records: its `verified/gradcam_validation_001/` directory.

## Run on Kaggle

Build with `uv --cache-dir /tmp/dmml-uv-cache run python scripts/build_explainability_notebook.py`, then upload
`notebooks/kaggle_video_explainability.ipynb` to Kaggle. Enable one T4 GPU and
attach these saved notebook outputs (not test-preparation outputs):

| Output | Version |
| --- | ---: |
| `bijaykhadka1710/unetpp-rd-002-inputs-00` | 2 |
| `bijaykhadka1710/unetpp-rd-002-inputs-01` | 2 |
| `bijaykhadka1710/unetpp-rd-002-train-03` | 1 |
| `namanbabariya/r2plus1d18-rd-001-train-04` | 1 |

Save and execute all cells. Download `gradcam_validation_001.zip`. A fresh output
directory is mandatory; partial or failed runs retain a status/error record and
hash manifest. GPU/cache availability and runtime compatibility remain execution
requirements. No automatic precision, layer or resolution changes are allowed.

Alternatively, prepare the private Kaggle CLI bundle (without submitting):

```bash
uv --cache-dir /tmp/dmml-uv-cache run python scripts/prepare_explainability_job.py --username bijaykhadka1710
```

The bundle is written under ignored `outputs/kaggle_jobs/rd-gradcam-validation-001/`.
Its metadata pins the four source versions above. The account running it needs
permission to attach every source output.

## Local checks

The analytical attribution, real-architecture synthetic inputs, deterministic
controls, state preservation, hook cleanup, source embedding, failure archives,
private job packaging and PNG/GIF exports passed **14 local CPU tests**. These do not establish behavior on
real validation clips. The tests use an isolated uv environment; no dependencies
were added to the base project and no system Python packages were changed.

```bash
MPLCONFIGDIR=/tmp/rd-gradcam-matplotlib uv --cache-dir outputs/.pilot-uv-cache run --offline --no-project --with torch --with torchvision --with pandas --with nbformat --with opencv-python-headless --with ipython --with scikit-learn --with huggingface-hub --with matplotlib --with pillow --index https://download.pytorch.org/whl/cpu python -m unittest discover -s tests -p '*explainability*.py' -v
```

The offline command requires the packages already cached. Initial dependency
download requires network access. The repository's full test suite was not run.
Full-size validation explanation inference subsequently completed on Kaggle.

## Fixed method

| Model | Frozen epoch | Feature layer |
| --- | ---: | --- |
| U-Net++ | 48 | `encoder.4` |
| R(2+1)D-18 | 15 | `backbone.layer4` |

The eight validation cases are selected as two per U-Net++ outcome (TP/TN/FP/FN),
ascending clip ID within each outcome, before viewing maps. Other models retain
their own outcomes on those same cases. This panel is outcome-stratified, not a
representative performance sample. Model probabilities are checked against saved
validation predictions (absolute tolerance 0.001) before explanations.

Batch-one FP32 Grad-CAM targets the scalar RD logit `z` and separately `-z` for
non-RD evidence. Channel weights are the gradient mean over feature depth,
height and width. The channel-weighted activation sum is rectified and resized
to 96x128x128 with trilinear interpolation, `align_corners=False`. Min-max
normalization spans the whole clip. Constant maps render zero and are flagged;
raw attribution arrays distinguish a constant positive map from no attribution.

Spatial and temporal resolution differ across architectures. Color strength
after normalization is not an absolute cross-model attribution comparison.
Frame positions refer to resampled indices, not original acquisition timestamps.
GIF playback is sampled visualization, not original timing. Scores are model
outputs, not calibrated clinical risks.

## Controls and evidence to inspect

- Occlusion uses 12x16x16 cuboids, eight each from high/low attribution and seed-42
  uniform sampling, one zero-filled cuboid per intervention. Positive logit drop
  means masking reduced the RD score. Tied/constant maps cannot support a useful
  high-versus-low ranking; inspect the explicit flag. Zero-fill is artificial.
- Cumulative weight randomization resets the head, head plus final feature block,
  and then all learned layers on disposable copies. Saved map arrays, logits,
  mean absolute differences and Pearson correlations support inspection; constant
  maps have undefined correlation. Similarity alone does not validate a map.
- Inspect `prediction_parity.json`, every case's `result.json`, `occlusion.json`
  and `randomization.json`, plus `occlusion_comparison.csv`. Include unexpected
  findings and failures in the presentation.
- Check `status.json` reports both models and all 16 model/case pairs completed.
  Verify `artifact_sha256.json`, examine all matched panels, and record GPU
  timing/memory and precision differences before describing findings.

## Interpretation and outstanding work

The exported panels, GIFs and temporal plots explain model-score attribution.
They do not supply ground-truth retinal masks, causal validation, anatomical
correctness or clinical validation. The official split may share patients;
results use one training seed. Randomization and occlusion are sensitivity
checks, not certificates of correctness.

Kaggle execution, real-input parity, full-input FP32 memory measurement and saved
control outputs are complete. A matched example was visually inspected. An
exhaustive anatomical review of all maps has not been performed. Controls are
mixed; preserve those findings when presenting the examples.

R3D-18 has now been integrated through the existing adapter with frozen epoch
18, `backbone.layer4` and `backbone.fc`. Its separately pinned extension explains
only the new model on the same eight cases and reuses these original results.
See [R3D18_EXPLAINABILITY_RESULTS.md](R3D18_EXPLAINABILITY_RESULTS.md) for the
current execution status, verification and three-model presentation delivery.

Method references: [Grad-CAM](https://arxiv.org/abs/1610.02391),
[Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292).
