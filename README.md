# Retinal Detachment Analysis from Ocular Ultrasound Using Deep Learning

CSCI 555 — Data Mining and Machine Learning, St. Francis Xavier University.
Three-person research project. Updated 2026-11-22.

**Status: three architecture comparisons and their validation 3D Grad-CAM
explanations are complete and verified.** U-Net++, R3D-18 and R(2+1)D-18 each
completed 50 training epochs, frozen-checkpoint evaluation and explanations on
the same eight validation clips. The offline presentation demo shows all three.

We study video-level retinal detachment (RD) classification using ERDES
(Eye Retinal DEtachment ultraSound). The task is `non_rd_vs_rd`: 0 = normal/PVD
(non-RD), 1 = RD. Our experiments compare architectures under a shared training
recipe and inspect which spatial and temporal regions contribute to their
scores. This is educational research; no clinical validation is claimed.

## Start here

- [Verified results and full comparison](docs/FINAL_MODEL_COMPARISON.md): separate
  validation and reused official test tables, metrics, sources and limitations.
- [Classroom demo guide](docs/EXPLAINABILITY_DEMO.md): open the offline viewer,
  present the examples and interpret the controls.
- [Grad-CAM method and original results](docs/EXPLAINABILITY_RESULTS.md), plus
  [R3D-18 explanation verification](docs/R3D18_EXPLAINABILITY_RESULTS.md).
- [Notebook guide](notebooks/README.md), [Kaggle workflow](docs/KAGGLE_GUIDE.md)
  and [data access/provenance](docs/DATA_ACCESS.md) for reproduction.
- [Team contributions](CONTRIBUTIONS.md).

## Three-model comparison

All three models trained from scratch for 50 epochs with seed 42. Checkpoints
were selected by highest validation accuracy, with the earliest epoch winning
ties. The decision threshold is fixed at 0.5. These are our measured results,
separate from author-reported ERDES paper results.

### Validation: used for checkpoint selection

431 clips: 40 RD and 391 non-RD.

| Model | Selected epoch | Accuracy | RD sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| U-Net++ | 48 | 98.84% | 92.50% | 99.49% |
| R3D-18 | 18 | 99.30% | 92.50% | 100.00% |
| R(2+1)D-18 | 15 | 99.07% | 95.00% | 99.49% |

### Reused official test benchmark

1,077 clips: 100 RD and 977 non-RD. The official test was already exposed earlier
in the project, so this is a **reused benchmark**, not a fresh untouched holdout.
Model choices, selected weights and thresholds were frozen before each new
model's test inference. Existing predictions were retained without rerunning
models for a better result.

| Model | Accuracy | RD detected / 100 | Missed RD (FN) | False positives (FP) | Specificity | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net++ | 97.49% | 86 | 14 | 13 | 98.67% | 86.43% |
| R3D-18 | 97.96% | 89 | 11 | 11 | 98.87% | 89.00% |
| R(2+1)D-18 | 97.86% | 92 | 8 | 15 | 98.46% | 88.89% |

R3D-18 has the highest observed accuracy, specificity and F1 among these three;
it misses three fewer RD clips and produces two fewer false positives than
U-Net++. R(2+1)D-18 detects the most RD clips, with more false positives.
These are descriptive tradeoffs on one benchmark and seed, not evidence of a
universally best model. Accuracy alone is insufficient for this imbalanced task.

Full confusion counts, precision, balanced accuracy, average precision and
ROC-AUC are in [the comparison report](docs/FINAL_MODEL_COMPARISON.md).
Model-specific evidence:
[U-Net++](docs/UNETPP_RESULTS.md),
[R3D-18](docs/R3D18_RESULTS.md) / [independent audit](docs/R3D18_VERIFICATION.md),
[R(2+1)D-18](docs/R2PLUS1D18_RESULTS.md) /
[independent audit](docs/R2PLUS1D18_VERIFICATION.md).

## 3D Grad-CAM for all three models

The completed explanation panel uses the **same eight fixed validation clips**:
two per U-Net++ outcome (TP/TN/FP/FN), selected by clip ID before viewing maps.
Each model retains its own predictions on those examples. The panel illustrates
successes and failures; it does not estimate overall performance.

Grad-CAM attributes the RD logit and its opposite target over space and time.
Explanations use FP32, full-size inputs and frozen weights, with saved-prediction
parity checks, model-state preservation, occlusion and cumulative weight
randomization controls. U-Net++ uses `encoder.4`; the two residual models use
`backbone.layer4`. The original two-model outputs are reused unchanged in the
combined viewer. All **352 saved artifact hashes** were checked.

For example, on RD validation clip `405744_00014`, U-Net++ misses RD at a 0.64%
score, while R(2+1)D-18 detects it at 61.11% and R3D-18 at 86.80%. On another RD
clip, `102769_00005`, U-Net++ detects RD while both residual models miss it.
These are FP32 explanation scores, not calibrated clinical risks.

Controls have mixed findings. R3D-18 has three constant-zero RD maps; among its
five nonconstant RD maps, high-attribution masking produces a greater mean
RD-logit drop than low-attribution masking in 5/5 cases and random masking in
4/5. Constant maps are flagged in the viewer. Neither attractive heatmaps nor
these small-panel controls establish anatomical correctness. Maps are coarse
score attribution, not segmentation masks; colors are normalized separately.

### Open the presentation demo

Generated artifacts are ignored by Git. In the completed local workspace:

1. Open `outputs/presentation/gradcam_demo/index.html`.
2. To present elsewhere, copy `outputs/presentation/gradcam_demo.zip`, extract the
   entire ZIP and open its `index.html`, keeping `assets/` beside it.
3. Use the clip/view selectors, Previous/Next and Full screen. Notes and
   explanation checks are available without moving the desktop comparison.

All three model views and the controls fit in one screen at the tested desktop
sizes, including 1024×600 and 1366×768. Smaller screens use a scrolling layout.
Offline Chromium checks covered eight clips, four views and 96 media loads,
with no JavaScript errors. Other browsers and physical projectors were not
verified. GIFs may loop independently; still-frame panels show matched resampled
indices. No live inference, network or GPU is needed to present saved outputs.

The previous two-model demo is preserved. For artifact locations, rebuilding,
checks and the speaking script, see [the demo guide](docs/EXPLAINABILITY_DEMO.md).

## Reproducibility and limitations

- Official split membership is unchanged: 3,873 training / 431 validation /
  1,077 test clips. Inputs are grayscale `[1,96,128,128]`, using square padding,
  full-volume trilinear interpolation and scaling to `[0,1]`.
- The architecture comparison uses scratch initialization, Adam at `1.5e-5`,
  unweighted BCE, the inherited plateau scheduler, physical batch 4/effective
  batch 48 and FP16 training with FP32 master weights. The three architecture
  runs use no augmentation, class weighting or resampling.
- Our U-Net++ reference is the ERDES-aligned 3D encoder/classifier with global
  pooling, not a nested segmentation decoder. R3D-18 and R(2+1)D-18 are distinct
  torchvision video classifiers, not exact reproductions of the paper's deeper
  MONAI ResNet. See [the source audit](docs/research/erdes-unetpp-protocol.md).
- This compares architectures under one inherited recipe, not each model's best
  tuned performance. U-Net++ uses InstanceNorm; the residual models use
  BatchNorm. Gradient accumulation does not increase BatchNorm's physical batch.
  The U-Net++ run used CUDA 12.8; the residual runs used CUDA 13.0.
- One seed and possible patient overlap across official clip-level splits limit
  generalization claims. No unseen-patient or clinical validation is established.

The separate U-Net++ augmentation experiment has also completed training and
frozen evaluation. Its record is available in
[the augmentation report](docs/UNETPP_AUGMENTATION_RESULTS.md); its interpretation
is separate from the three-architecture comparison above.

## Environment and reproduction

Use **uv**; do not modify system Python. The local non-packaged workspace
requires Python >=3.14 and currently declares no ML dependencies in
`pyproject.toml`. CPU checks use isolated uv environments; full GPU training and
explanation jobs use recorded Kaggle runtimes. Their Python/CUDA versions are
recorded with each experiment rather than assumed to match the local interpreter.

```bash
uv sync --locked --python 3.14
uv run --locked python -c "import sys; sys.path.insert(0, 'src'); import rd_ultrasound_ml; print(sys.version)"
```

These commands initialize/check the local workspace; they do not install the
full ML stack or start a job. Follow [the notebook guide](notebooks/README.md),
[tests guide](tests/README.md) and each experiment's documented uv commands.
Kaggle reproduction requires permission to attach the exact private input and
training-output versions. Preserve frozen configs, source hashes, seeds and
runtime records. Do not use test results to tune thresholds or select checkpoints.

The [initial experiment log](docs/EXPERIMENT_LOG.md), metadata/split exploration
and early Colab pilots remain historical records. The completed runs and
verification reports above describe the current project state.

## Repository organization

- `src/rd_ultrasound_ml/`: preprocessing/cache checks, classifiers, training,
  evaluation, augmentation and explanation workflows.
- `configs/`: recorded experiment recipes and frozen evaluation identities.
- `notebooks/`: exploration and standalone training/evaluation/explanation jobs.
- `scripts/`: notebook/job builders, continuation, saved-output audits and demos.
- `tests/`: behavioral and delivery checks using synthetic inputs.
- `docs/`: research protocol, results, provenance, limitations and presentation guides.
- `data/` and `outputs/`: ignored local data, weights and generated artifacts.
- `prompts/` and `AGENTS.md`: task prompts and recurring project instructions.

## Team workflow

**`dev` is the stable integration branch.** Start feature branches from `dev`,
make focused changes, run relevant checks, review the diff, then commit/push and
open a pull request targeting `dev`. Another teammate should review where
practical. Bijay promotes the checked `dev` branch to `main` when ready.

The final model and explanation integration was merged into `dev` through
[PR #19](https://github.com/Bijay4444/rd-ultrasound-ml/pull/19).
See [the integration workflow](docs/FINAL_INTEGRATION_WORKFLOW.md) for provenance.
Never commit raw videos, checkpoints, large outputs, logs, credentials or secrets.

## Citation and attribution

Dataset and reference implementation: [ERDES](https://github.com/OSUPCVLab/ERDES),
with dataset revision/license evidence in [the acquisition record](docs/DATA_ACCESS.md).
Paper: [ERDES: A Benchmark Video Dataset for Retinal Detachment and Macular
Status Classification in Ocular Ultrasound](https://arxiv.org/abs/2508.04735).
Versioned bibliographic details are in [the paper notes](docs/ERDES_PAPER_NOTES.md).
Our implementation and measured results remain distinguishable from the upstream
reference and its author-reported results. Individual work is recorded in
[CONTRIBUTIONS.md](CONTRIBUTIONS.md).
