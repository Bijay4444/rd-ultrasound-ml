# R(2+1)D-18 RD comparison: `r2plus1d18_rd_001_amp`

Updated 2026-11-11. **Status: 21 of 50 epochs complete (GPU sessions 1–2 on Kaggle Tesla T4:
`namanbabariya/r2plus1d18-rd-001-train-01`, then `-train-02` for epochs 2–21). The run is partial; the
numbers below are not the model's final result.** Versions 1 and 2 of `-train-01` were allocated no GPU and
stopped at the CUDA check before reading data; the successful session 1 was started from the Kaggle editor
with GPU T4 selected. Session 2 was pushed from the CLI and did get the T4. Session 3 (`-train-03`,
epochs 22–41) was submitted 2026-11-11 10:53 and is running; its results are not included yet.
Every cell below marked `pending` is unmeasured. This document must be
updated from saved Kaggle logs only. See the [shared protocol](MODEL_COMPARISON_PLAN.md)
and [assignment](handoffs/R2PLUS1D18.md).

Branch `feature/r2plus1d18-rd`, starting commit `911a4409f69bfb639ffb1dcde605de0877619e05`.

## Blockers (what is missing)

| Needed | Status |
| --- | --- |
| GPU | None locally; training runs on Kaggle T4 under the teammate's own account (resolved 2026-11-10). |
| Kaggle credentials | Configured in ignored `.env.kaggle`; `kernels list --mine` succeeded (resolved 2026-11-10). |
| Prepared cache access | Resolved 2026-11-10: Bijay shared `bijaykhadka1710/unetpp-rd-002-inputs-00` and `-01`. **Version 2** of each is the only COMPLETE version (version 1 of each: ERROR). Their downloaded `config.json` files hash to their manifest digests and to the pinned `6907d5f9…`; each manifest is complete with 2,152 entries, `test_used: false`; preprocessing identity torch 2.11.0+cpu / OpenCV 4.14.0 / NumPy 2.1.3. Tensor hashes and the cache fingerprint are verified inside the GPU job. |
| Physical-batch coordination | Agree a common physical batch with the R3D-18 owner after both probes. Config currently uses the inherited 4 × 12. |

## Model

- `torchvision.models.video.r2plus1d_18(weights=None, num_classes=1)`, wrapped as
  `backbone` in `R2Plus1D18RDClassifier` (`src/rd_ultrasound_ml/r2plus1d18_baseline.py`).
- **Only `backbone.stem[0]` changes:** `Conv3d(3→1, 45, (1,7,7), stride (1,2,2), padding (0,3,3), bias=False)`.
  The temporal stem convolution `Conv3d(45, 64, (3,1,1))`, both BatchNorm3d/ReLU pairs,
  the 16 `Conv2Plus1D` factorized convolutions, pooling and `Linear(512,1)` head stay at builder defaults.
- Initialization: builder defaults. The replacement gets `kaiming_normal_(mode='fan_out', nonlinearity='relu')`,
  which is the builder's Conv3d rule. `fan_out = 45·1·7·7` does not depend on input channels, so the replacement's
  distribution is the same as the stock conv. Measured weight std 0.0305 vs target 0.0301.
- Output: raw logits shaped `[B]`; sigmoid only for probabilities.
- **Parameters: 31,296,228** (stock 400-class RGB is 31,505,325; `num_classes=1` gives 31,300,638; the 1-channel stem removes 4,410).
  Verified with torchvision 0.29.2. The notebook stops if the Kaggle torchvision builds a different count.
- Input `[B,1,96,128,128]`, grayscale `[0,1]` from the existing cache; no Kinetics normalization.

### Grad-CAM integration (for Bijay)

- Module path: **`backbone.layer4`** (`gradcam_layer(model)` returns it).
- Feature shape on the full input: **`[B, 512, 12, 8, 8]`** (C, D, H, W). Measured on CPU with `[2,1,96,128,128]`.
- Target the scalar RD logit (model output before sigmoid). Use `model.eval()` with autograd enabled.
- The GPU export also writes `gradcam_layer.json` with the shape recorded on one real validation clip.

## Cache reuse without reusing U-Net++ identity

`load_prepared_shards()` requires each manifest's `config_digest` to equal the digest
of the config it receives. That is the full U-Net++ preparation config, so the new
model config would be rejected. `src/rd_ultrasound_ml/video_cache.py::load_cache_for_run`
(model-agnostic; R3D-18 can reuse it):

1. reads each attached shard's own `config.json` and verifies its digest equals that shard's manifest digest, and that both shards agree;
2. requires that digest to equal the pinned `cache.preparation_config_digest`
   = `6907d5f903b39b83ca61ae2ab96a1f2b8b18078f3cab6d9ad614438597fb689f`;
3. requires matching `dataset_revision`, `task`, `label_mapping` and `preprocessing`;
4. calls the **unchanged** `load_prepared_shards` with the original preparation config, so every tensor SHA256, source hash, label, partition/order and shape check runs;
5. requires the returned cache fingerprint to equal the pinned
   `111e9cb7b1163c3be3b4a3dd2374f80434c402aad65168844cfcbb1087187ab0`;
6. returns a **run fingerprint** that binds preparation digest, cache fingerprint, preprocessing identity and the new training-config digest. `run_training` stores it in checkpoints and refuses to resume on any change.

Pin evidence: recomputing the digest from the tracked `notebooks/kaggle_unetpp_preparation.ipynb`
config plus its embedded-module SHA256s reproduces `6907d5f9…`. That equals
`training_config_digest` in the frozen `configs/unetpp_rd_002_evaluation.json`, which the
completed U-Net++ GPU run could only have used if it matched the preparation manifests.
The fingerprint pin equals that freeze's `training_data_fingerprint`. Both pins are
checked by tests. Session 1 confirmed both pins against the attached real outputs: all 4,304 clips verified,
returned cache fingerprint `111e9cb7…` (`input_fingerprint.json`).

## Protocol (frozen config `configs/r2plus1d18_rd_001_training.json`)

Inherited unchanged from `unetpp_rd_002_training.json` (test-checked): seed 42, 50 epochs,
3,873 train / 431 validation official clips, Adam lr 1.5e-5, betas (0.9, 0.999), eps 1e-8,
weight decay 0, unweighted `BCEWithLogitsLoss`, ReduceLROnPlateau on validation loss
(factor 0.1, patience 10, threshold 1e-4 rel, cooldown 0, min_lr 0), effective batch 48 =
physical 4 × accumulation 12, FP16 autocast + FP32 master weights + GradScaler, threshold 0.5,
highest validation accuracy with earliest ties. No augmentation, pretrained weights,
class weighting or resampling. At runtime the notebook adds `implementation_sha256` for the
seven embedded modules to the config before its digest is taken.

## Checks actually run (local, CPU, synthetic data)

Environment: `uv run --no-project` with Python 3.11.17, torch 2.15.0+cpu and torchvision 0.29.2+cpu.
This is not the Kaggle runtime.

```bash
uv run --no-project --with ipython --with numpy --with opencv-python-headless --with pandas \
  --with scikit-learn --with torch --with torchvision \
  --index https://download.pytorch.org/whl/cpu python -m unittest tests.test_r2plus1d18 -v
```

Result: **15/15 passed**. Covered:
- one-channel stem, preserved temporal stem and factorized blocks, `Linear(512,1)`, and the parameter count;
- seeded initialization is deterministic; the replacement follows the Kaiming rule; no weight download is attempted;
- `[B]` logits, finite loss, nonzero finite gradients in all parameter tensors; RGB input is rejected;
- full `[1,1,96,128,128]` forward and the `backbone.layer4` shape;
- the cache adapter accepts valid shards under the new identity, while plain `load_prepared_shards` rejects them;
- it rejects an edited preparation config, a wrong digest pin, a wrong fingerprint pin, a missing pin, changed preprocessing, a corrupted tensor and a changed label;
- the run fingerprint changes when the training config changes;
- two 1-epoch sessions reproduce one 2-epoch call exactly (real R(2+1)D-18 on small synthetic clips, CPU FP32);
- resume is refused when the data fingerprint or config changes;
- reloading `best.pt` reproduces the saved validation probabilities (tolerance 1e-6), and a tampered prediction file is rejected;
- per-clip prediction rows check label/order;
- the tracked notebook matches the builder, embeds the current sources and config, compiles, and contains no test/download path;
- bundles are private, GPU, internet off and version-pinned.

Separate manual check: full-size `[2,1,96,128,128]` forward + BCE + backward on CPU in 34.7 s.
Loss 0.695, 113/113 parameter tensors had finite nonzero gradients, layer4 output `[2,512,12,8,8]`.
This is CPU code evidence, not a GPU timing.

Full existing suite (`discover -s tests`, 77 tests): new tests pass. Four pre-existing tests fail on
this Windows machine for environment reasons, and none of those files were modified:
- two notebook-embedding tests read UTF-8 files with the cp1252 default; they pass with `PYTHONUTF8=1`;
- `test_baseline_training` step-7 hits a Windows memmap file lock;
- `test_weighted_followup` expects the ignored `outputs/` directory to exist.

GPU checks are reported under Results (session 1).

## How to run (once access exists)

```bash
python scripts/build_r2plus1d18_notebook.py      # only if src/ or the config changed
python scripts/prepare_r2plus1d18_job.py --inputs OWNER/unetpp-rd-002-inputs-00/V OWNER/unetpp-rd-002-inputs-01/V
python scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/r2plus1d18-rd-001-train-01
# after session 1 completes and the probe/epoch timing is reviewed:
python scripts/prepare_r2plus1d18_job.py --inputs ... --session 2 --epochs N
```

Session 1 runs the cache checks, a disposable real-training-clip AMP probe (batch 4, then 2 and 1
on OOM, recorded but never silently adopted), then **one full epoch**. That epoch starts from seed 42
with fresh weights and is epoch 1 of the fixed run. Each session returns normally before
`R2P1_MAX_SESSION_SECONDS` (11 h) and exports: `history.csv`, `epoch_memory.json`,
`precision_probe.json`, `best.pt`, `best_checkpoint_reload.json`,
`best_validation_predictions.csv` (clip_id, label, probability, class, partition),
`training_curves.png`, `gradcam_layer.json`, the runtime (including torchvision),
the cache record, and `artifact_sha256.json`.

## Results

| Item | Value |
| --- | --- |
| Runtime | Kaggle Tesla T4 (15.6 GB), Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, cuDNN 92700 |
| Cache checks | 4,304/4,304 attached clips verified; preparation digest `6907d5f9…` and cache fingerprint `111e9cb7…` matched pins; run fingerprint `1103636b…`; `test_used: false` |
| Precision probe | physical batch 4, 16-mixed: passed; median step 1.37 s; peak allocated 8.38 GB, reserved 9.64 GB; finite nonzero gradients; 0 skipped updates. Batch 4 fits, so the inherited 4 × 12 is kept. |
| First full epoch wall time | 1,788.7 s (≈29.8 min) including 84.7 s validation; epoch peak allocated 8.51 GB, reserved 9.57 GB |
| Session overhead | whole session 2,669 s; ≈12 min of it is the per-clip cache verification at start |
| Projected 50-epoch GPU time | epochs 2–21 averaged ≈1,717 s (1,658–1,917 s); 29 epochs remain ≈ 13.8 h plus ≈15 min per session → session 3 (epochs 22–41) and session 4 (42–50) |
| Epochs completed / 50 | 21 (`run_status.json` status `partial`); session 2 resumed from session 1's checkpoint with the same run fingerprint `1103636b…`; epoch-1 history row carried over unchanged |
| Epoch 1 validation (partial run, not a result) | train loss 0.4563, val loss 0.3245; TP 0 / TN 391 / FP 0 / FN 40; accuracy 0.9072 (= all-negative rate), sensitivity 0.000, specificity 1.000, precision undefined, F1 0.000, balanced accuracy 0.500, AP 0.2585, ROC-AUC 0.6679 |
| Best checkpoint so far (partial run) | epoch 15 by validation accuracy, `best.pt` SHA256 `c041f7e2…3ec16`; reload reproduced validation probabilities exactly (max diff 0.0). Epoch 15: TP 38 / TN 389 / FP 2 / FN 2; accuracy 0.9907, sensitivity 0.950, specificity 0.9949, precision 0.950, F1 0.950, balanced accuracy 0.9724, AP 0.9795, ROC-AUC 0.9970. May change in later epochs. |
| Session 2 memory / stability | epoch peak reserved 9.91 GB; 1 AMP skipped update (epoch 17), 0 otherwise; learning rate stayed 1.5e-5 through epoch 21 |
| Validation by epoch (partial run) | see table below |
| Grad-CAM layer (real clip) | `backbone.layer4`, `[1,512,12,8,8]` for input `[1,1,96,128,128]` |
| Artifact integrity | session 1: 20 small files match `artifact_sha256.json`; session 2: 40 small files match |
| Inference timing | pending |
| Private artifact location | Kaggle output of `namanbabariya/r2plus1d18-rd-001-train-01` (latest version); local copy under ignored `outputs/kaggle_results/` |


### Validation by epoch (official validation split, 431 clips, threshold 0.5)

From `history.csv` of `-train-02`. Selection rule: highest accuracy, earliest tie.

| Epoch | Train loss | Val loss | Acc | Sens | Spec | Bal. acc | ROC-AUC | AP | TP/FN/FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4563 | 0.3245 | 0.9072 | 0.0000 | 1.0000 | 0.5000 | 0.6679 | 0.2585 | 0/40/0 |
| 2 | 0.3127 | 0.2942 | 0.9072 | 0.0000 | 1.0000 | 0.5000 | 0.6903 | 0.2795 | 0/40/0 |
| 3 | 0.2773 | 0.2566 | 0.9049 | 0.3250 | 0.9642 | 0.6446 | 0.8315 | 0.4592 | 13/27/14 |
| 4 | 0.2165 | 0.1787 | 0.9466 | 0.6250 | 0.9795 | 0.8023 | 0.9338 | 0.7579 | 25/15/8 |
| 5 | 0.1648 | 0.1818 | 0.9513 | 0.6500 | 0.9821 | 0.8160 | 0.9260 | 0.7560 | 26/14/7 |
| 6 | 0.1490 | 0.1136 | 0.9675 | 0.7250 | 0.9923 | 0.8587 | 0.9548 | 0.8677 | 29/11/3 |
| 7 | 0.1206 | 0.0844 | 0.9745 | 0.7500 | 0.9974 | 0.8737 | 0.9892 | 0.9310 | 30/10/1 |
| 8 | 0.1007 | 0.0968 | 0.9698 | 0.7750 | 0.9898 | 0.8824 | 0.9785 | 0.9038 | 31/9/4 |
| 9 | 0.0844 | 0.0781 | 0.9791 | 0.8000 | 0.9974 | 0.8987 | 0.9909 | 0.9550 | 32/8/1 |
| 10 | 0.0719 | 0.0755 | 0.9791 | 0.8250 | 0.9949 | 0.9099 | 0.9922 | 0.9481 | 33/7/2 |
| 11 | 0.0645 | 0.0779 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9904 | 0.9431 | 31/9/0 |
| 12 | 0.0562 | 0.0677 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9941 | 0.9597 | 31/9/0 |
| 13 | 0.0456 | 0.0649 | 0.9745 | 0.9000 | 0.9821 | 0.9410 | 0.9952 | 0.9667 | 36/4/7 |
| 14 | 0.0443 | 0.0680 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9923 | 0.9470 | 31/9/0 |
| 15 | 0.0382 | 0.0490 | 0.9907 | 0.9500 | 0.9949 | 0.9724 | 0.9970 | 0.9795 | 38/2/2 |
| 16 | 0.0280 | 0.0500 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9988 | 0.9892 | 33/7/0 |
| 17 | 0.0299 | 0.1024 | 0.9652 | 0.6250 | 1.0000 | 0.8125 | 0.9779 | 0.9361 | 25/15/0 |
| 18 | 0.0319 | 0.0539 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9970 | 0.9770 | 33/7/0 |
| 19 | 0.0233 | 0.0546 | 0.9768 | 0.8500 | 0.9898 | 0.9199 | 0.9951 | 0.9646 | 34/6/4 |
| 20 | 0.0227 | 0.0649 | 0.9768 | 0.7500 | 1.0000 | 0.8750 | 0.9973 | 0.9773 | 30/10/0 |
| 21 | 0.0150 | 0.0682 | 0.9791 | 0.9250 | 0.9847 | 0.9548 | 0.9957 | 0.9682 | 37/3/6 |

Validation sensitivity is unstable across epochs (0.625–0.950 from epoch 4 on) with only 40 RD clips; one clip moves sensitivity by 0.025.

Reference for later comparison (U-Net++ epoch 48, **validation**): accuracy 0.988399,
sensitivity 0.925000. No test inference is planned until the team freezes all choices;
any later test result must be labelled **reused official benchmark**.

## Limitations and deviations

- This compares architectures under the inherited U-Net++ recipe. The small learning rate may disadvantage this model, so the result is not this model's tuned best.
- BatchNorm3d sees 4 clips per forward pass. Accumulation does not make BatchNorm see 48. U-Net++ used InstanceNorm, so this is not a normalization-controlled comparison.
- FP16 mixed precision. Kaggle runtime versions may differ from the U-Net++ run; they are recorded, not assumed.
- One seed and clip-level official splits with possible patient overlap. No unseen-patient or clinical claim.
- torchvision R(2+1)D-18 is not an ERDES Table 8 row.
