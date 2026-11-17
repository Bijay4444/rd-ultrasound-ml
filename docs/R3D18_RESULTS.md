# R3D-18 RD comparison: `r3d18_rd_001_amp`

Updated 2026-11-16. **Status: implemented and tested on CPU with synthetic data; no GPU run has happened.
No training epoch, no GPU memory measurement and no model result exist yet.** Every cell below marked
`pending` is unmeasured. This document must be updated from saved Kaggle logs only. See the
[shared protocol](MODEL_COMPARISON_PLAN.md) and [assignment](handoffs/R3D18.md).

Branch `feature/r3d18-rd`, starting commit `b96cb39ab90d0273288e92ee20f9a87787f3ec8d` (tip of `dev`, which already
contains the merged R(2+1)D-18 work).

## Blockers (what is missing)

| Needed | Status |
| --- | --- |
| GPU | None locally (macOS, CPU only). Training needs a Kaggle T4 under the R3D-18 owner's own Kaggle account. **pending** |
| Kaggle credentials | **Missing.** Kaggle account, GPU/phone verification and an API token (`bash scripts/setup_kaggle.sh`, or submit the notebook from the Kaggle website) are needed. Nothing has been created. |
| Prepared cache access | **Missing.** Bijay must share the two version-pinned U-Net++ preparation outputs (`unetpp-rd-002-inputs-00`, `-01`; the R(2+1)D owner used version 2 of each, the only COMPLETE version) with this Kaggle account. Private files are not accessible just because their names appear in the repo. |
| Physical-batch coordination | Not done. The config uses the inherited 4 x 12. The R(2+1)D-18 owner measured batch 4 on a T4 for their model (peak allocated 8.38 GB, see `docs/R2PLUS1D18_RESULTS.md`); that is **not** a measurement for R3D-18. Agree a common physical batch after this model's own probe. |

## Model

- `torchvision.models.video.r3d_18(weights=None, num_classes=1)`, wrapped as `backbone` in
  `R3D18RDClassifier` (`src/rd_ultrasound_ml/r3d18_baseline.py`).
- **Only `backbone.stem[0]` changes.** The installed stem is `BasicStem` = `Conv3d(3, 64, kernel (3,7,7),
  stride (1,2,2), padding (1,3,3), bias=False)`, `BatchNorm3d(64)`, `ReLU`. The replacement is
  `Conv3d(1, 64, (3,7,7), stride (1,2,2), padding (1,3,3), bias=False)`. The BatchNorm/ReLU, the 16 joint
  `Conv3DSimple` 3x3x3 convolutions in eight BasicBlocks, pooling and `Linear(512,1)` stay at builder defaults.
- Initialization: builder defaults. The replacement gets `kaiming_normal_(mode='fan_out', nonlinearity='relu')`,
  the builder's Conv3d rule (torchvision `resnet.py`). `fan_out = 64*3*7*7` does not depend on input channels, so
  the replacement has the same distribution as the stock conv. Measured weight std 0.01464 vs target 0.01458.
- Output: raw logits shaped `[B]`; sigmoid only for probabilities.
- **Parameters: 33,147,969** (stock 400-class RGB model 33,371,472; `num_classes=1` gives 33,166,785; the one-channel
  stem removes 2*64*3*7*7 = 18,816). Verified with torchvision 0.29.2. The notebook stops if the Kaggle torchvision
  builds a different count.
- Input `[B,1,96,128,128]`, grayscale `[0,1]` from the existing cache; no Kinetics normalization.

### Grad-CAM integration (for Bijay)

- Module path: **`backbone.layer4`** (`gradcam_layer(model)` returns it).
- Feature shape on the full input: **`[B, 512, 12, 8, 8]`** (C, D, H, W). Measured on CPU with `[1,1,96,128,128]`.
- Target the scalar RD logit (model output before sigmoid). Use `model.eval()` with autograd enabled.
- The GPU export also writes `gradcam_layer.json` with the shape recorded on one real validation clip. **pending**

## Cache reuse without reusing U-Net++ identity

Nothing in the shared cache code changed. `src/rd_ultrasound_ml/video_cache.py::load_cache_for_run` (added by the
R(2+1)D owner, model-agnostic) validates the original preparation config against its pinned digest
(`6907d5f9...`), then calls the unchanged `load_prepared_shards`, so every tensor/source hash, label, order and shape
check still runs, and the pinned cache fingerprint (`111e9cb7...`) must match. It returns a run fingerprint that binds
the preparation identity, the cache fingerprint and **this model's own training-config digest**, so an R3D-18 run can
never resume from, or be mixed with, another model's state. `configs/r3d18_rd_001_training.json` carries the same
pins as the R(2+1)D config.

## Protocol (config `configs/r3d18_rd_001_training.json`)

Identical to the R(2+1)D-18 config in every field except the experiment id and the model block (checked field by field,
and checked against `unetpp_rd_002_training.json` by `tests/test_r3d18.py`): seed 42, 50 epochs, 3,873 train / 431
validation official clips, Adam lr 1.5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0, unweighted
`BCEWithLogitsLoss`, ReduceLROnPlateau on validation loss (factor 0.1, patience 10), effective batch 48 = physical 4 x
accumulation 12, FP16 autocast + FP32 master weights + GradScaler, threshold 0.5, highest validation accuracy with
earliest ties. No augmentation, pretrained weights, class weighting or resampling. The config is **not frozen** until the
GPU probe passes and the physical batch is agreed.

## Checks actually run (local, CPU, synthetic data)

Environment: macOS, CPU only, Python 3.14.2, torch 2.15.0, torchvision 0.29.2, OpenCV 5.0.0, pandas 3.1.1, in a uv
virtual environment outside the repository. This is not the Kaggle runtime.

```bash
python -m unittest tests.test_r3d18 -v
```

Result: **15/15 passed.** Covered: the one-channel stem and preserved architecture, the parameter count, seeded and
Kaiming-consistent initialization, no weight download, `[B]` logits with finite nonzero gradients in every parameter
tensor, RGB input rejected, the full-size input shape and `backbone.layer4` shape, the cache adapter accepting valid
shards while rejecting an edited preparation config, wrong pins, changed preprocessing, a corrupted tensor and a
changed label, resume equal to uninterrupted training, resume refused on changed data or config identity, checkpoint
reload reproducing validation probabilities (tolerance 1e-6) with a tampered file rejected, and the tracked notebook
and private version-pinned bundles matching their builders.

Full existing suite (`discover -s tests`, 99 tests including the R(2+1)D and U-Net++ tests): **all pass**. No existing
file was modified except `tests/README.md`.

Separate manual check: a full-size `[2,1,96,128,128]` forward + BCE + backward on this Mac's CPU took 116 s; loss 0.706,
62/62 parameter tensors had finite nonzero gradients. This is CPU code evidence, not a GPU timing.

Not run: any CUDA/GPU run, the FP16 precision probe on real clips, real cache verification, a full epoch.

## How to run (once access exists)

```bash
python scripts/build_r3d18_notebook.py      # only if src/ or the config changed
python scripts/prepare_r3d18_job.py --inputs OWNER/unetpp-rd-002-inputs-00/V OWNER/unetpp-rd-002-inputs-01/V
python scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/r3d18-rd-001-train-01
# after session 1 completes and the probe/epoch timing is reviewed:
python scripts/prepare_r3d18_job.py --inputs ... --session 2 --epochs N
```

Without the CLI, the same notebook can be opened on kaggle.com: upload `notebooks/kaggle_r3d18_training.ipynb`, attach
the two preparation outputs, select GPU T4, turn Internet off and save a version.

Session 1 runs the cache checks, a disposable real-training-clip AMP probe (batch 4, then 2 and 1 on OOM, recorded but
never silently adopted), then **one full epoch**. That epoch starts from seed 42 with fresh weights and is epoch 1 of the
fixed run. Each session returns normally before `R3D_MAX_SESSION_SECONDS` (11 h) and exports: `history.csv`,
`epoch_memory.json`, `precision_probe.json`, `best.pt`, `best_checkpoint_reload.json`,
`best_validation_predictions.csv` (clip_id, label, probability, class, partition), `training_curves.png`,
`gradcam_layer.json`, the runtime (including torchvision), the cache record and `artifact_sha256.json`.
No test split is read.

Resume is strict: `run_training` compares each session's runtime record (GPU model, Python, torch, torchvision,
CUDA/cuDNN and the other library versions) with the one stored in the checkpoint and refuses to continue on any
difference. Every session must therefore get the same GPU type (the bundle requests a T4) and the same Kaggle image.
If a session lands on different hardware or versions, do not edit the check; keep the artifacts, record it, and decide
with the team whether to restart or report a partial run.

## Results

| Item | Value |
| --- | --- |
| Runtime | pending |
| Cache checks | pending |
| Precision probe (physical batch 4, 16-mixed) | pending |
| Peak GPU memory (allocated / reserved) | pending |
| First full epoch wall time | pending |
| Projected 50-epoch time and available GPU budget | pending |
| Epochs completed / 50 | 0 of 50 (not started) |
| Selected epoch and validation metrics (431 clips: 40 RD, 391 non-RD, threshold 0.5) | pending |
| Selected checkpoint SHA256 and reload check | pending |
| Inference timing | pending |
| Private artifact location | pending |

### Validation comparison (official validation split, threshold 0.5; selection also used validation)

| Model | Accuracy | Sensitivity | Source |
| --- | ---: | ---: | --- |
| U-Net++ (epoch 48) | 0.9884 | 0.9250 | `docs/UNETPP_RESULTS.md` |
| R3D-18 | pending | pending | this report |
| R(2+1)D-18 (epoch 15) | 0.9907 | 0.9500 | `docs/R2PLUS1D18_RESULTS.md` |
| U-Net++ + augmentation | pending | pending | Bijay |

Official-test numbers are not reported here. The test split is not touched until the team freezes all final choices;
a frozen R3D-18 test run would be labelled a **reused official test benchmark**, and the model-agnostic
`comparison_evaluation.py` plus a gated notebook (modelled on the R(2+1)D-18 one) still need to be built for it.

## Limitations and deviations

- This compares architectures under the inherited U-Net++ recipe. The small learning rate may disadvantage this model,
  so the result is not its tuned best.
- BatchNorm3d sees 4 clips per forward pass. Accumulation does not make BatchNorm see 48. U-Net++ used InstanceNorm, so
  this is not a normalization-controlled comparison.
- FP16 mixed precision. Kaggle runtime versions may differ from the U-Net++ run; they are recorded, not assumed.
- One seed and clip-level official splits with possible patient overlap. No unseen-patient or clinical claim.
- torchvision R3D-18 is not the ERDES paper's deeper `[4,4,4,4]` MONAI ResNet.
- Code is implemented but untrained: nothing here supports any claim about R3D-18's performance.
