# R(2+1)D-18 RD comparison: `r2plus1d18_rd_001_amp`

Updated 2026-11-12. **Status: training complete, 50 of 50 epochs (`run_status.json` status `completed`),
on Kaggle Tesla T4 over four sessions: `namanbabariya/r2plus1d18-rd-001-train-01` epoch 1, `-train-02` epochs
2–21, `-train-03` epochs 22–41, `-train-04` epochs 42–50. Selected checkpoint: epoch 15. The single frozen evaluation on the reused official test split ran on
2026-11-13 (see "Reused official test benchmark"); all other metrics are on the validation split, which
was also used for checkpoint selection.** Versions 1 and 2 of `-train-01` were allocated no GPU and stopped at the CUDA check before reading
data; the successful session 1 was started from the Kaggle editor with GPU T4 selected. Sessions 2–4 were
pushed from the CLI and got the T4.
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
| **Selected epoch 15, validation (431 clips: 40 RD, 391 non-RD), threshold 0.5** | TP 38 / TN 389 / FP 2 / FN 2; accuracy **0.9907**, sensitivity **0.950**, specificity 0.9949, precision 0.950, F1 0.950, balanced accuracy 0.9724, AP 0.9795, ROC-AUC 0.9970 |
| Final epoch 50, validation (not selected) | TP 35 / TN 390 / FP 1 / FN 5; accuracy 0.9861, sensitivity 0.875, specificity 0.9974, balanced accuracy 0.9362, ROC-AUC 0.9944 |
| Total GPU training time | 50 epochs summed 22.6 h (`epoch_memory.json`), plus ≈15 min cache verification per session (4 sessions) |
| Epochs completed / 50 | **50** (`run_status.json` status `completed`); sessions 2–4 each resumed with the same run fingerprint `1103636b…`; earlier history rows carried over unchanged (checked row by row each session) |
| Selected checkpoint (final) | **epoch 15** by highest validation accuracy (earliest tie rule; no later epoch reached 0.9907). `best.pt` SHA256 `c041f7e2f635e9f3203e76045fbad247c3243788d527425d27cc73b27b93ec16`; final-session reload reproduced validation probabilities exactly (max diff 0.0) |
| Memory / stability | epoch peak reserved 9.91 GB of 15.6 GB; 1 AMP skipped update in 50 epochs (epoch 17). ReduceLROnPlateau: 1.5e-5 → 1.5e-6 (epoch 27) → 1.5e-7 (epoch 38) → 1.5e-8 (epoch 50) |
| Validation by epoch | see table below |
| Grad-CAM layer (real clip) | `backbone.layer4`, `[1,512,12,8,8]` for input `[1,1,96,128,128]` |
| Artifact integrity | every downloaded small file matched `artifact_sha256.json` in each session (20, 40, 60, 69 files); checkpoint `.pt` files were not downloaded locally |
| Inference timing | Tesla T4, batch 4, FP16: 91.4 ms per clip model forward (test run; see reused test section) |
| Private artifact location | Kaggle output of `namanbabariya/r2plus1d18-rd-001-train-04` (final `best.pt`, `last_*.pt`, full history and per-epoch validation files); sessions 1–3 outputs hold intermediate checkpoints. Local small-file copies under ignored `outputs/kaggle_results/r2p1_s0{1..4}_small/` |


### Validation by epoch (official validation split, 431 clips, threshold 0.5)

From `history.csv` of `-train-04` (rows 1–41 identical to earlier sessions). Selection rule: highest accuracy, earliest tie.

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
| 22 | 0.0146 | 0.0545 | 0.9814 | 0.8000 | 1.0000 | 0.9000 | 0.9978 | 0.9817 | 32/8/0 |
| 23 | 0.0122 | 0.1776 | 0.9350 | 0.3000 | 1.0000 | 0.6500 | 0.9914 | 0.9580 | 12/28/0 |
| 24 | 0.0229 | 0.0645 | 0.9838 | 0.8500 | 0.9974 | 0.9237 | 0.9900 | 0.9529 | 34/6/1 |
| 25 | 0.0193 | 0.0755 | 0.9838 | 0.8500 | 0.9974 | 0.9237 | 0.9797 | 0.9205 | 34/6/1 |
| 26 | 0.0139 | 0.0764 | 0.9675 | 0.8250 | 0.9821 | 0.9035 | 0.9874 | 0.9334 | 33/7/7 |
| 27 | 0.0107 | 0.0636 | 0.9791 | 0.8000 | 0.9974 | 0.8987 | 0.9917 | 0.9516 | 32/8/1 |
| 28 | 0.0085 | 0.0566 | 0.9814 | 0.8000 | 1.0000 | 0.9000 | 0.9933 | 0.9598 | 32/8/0 |
| 29 | 0.0093 | 0.0553 | 0.9814 | 0.8250 | 0.9974 | 0.9112 | 0.9925 | 0.9594 | 33/7/1 |
| 30 | 0.0083 | 0.0507 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9927 | 0.9664 | 34/6/0 |
| 31 | 0.0082 | 0.0523 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9929 | 0.9667 | 33/7/0 |
| 32 | 0.0082 | 0.0604 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9937 | 0.9635 | 31/9/0 |
| 33 | 0.0077 | 0.0544 | 0.9814 | 0.8000 | 1.0000 | 0.9000 | 0.9936 | 0.9644 | 32/8/0 |
| 34 | 0.0075 | 0.0623 | 0.9768 | 0.7500 | 1.0000 | 0.8750 | 0.9945 | 0.9676 | 30/10/0 |
| 35 | 0.0087 | 0.0721 | 0.9768 | 0.7500 | 1.0000 | 0.8750 | 0.9955 | 0.9694 | 30/10/0 |
| 36 | 0.0080 | 0.0508 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9955 | 0.9727 | 33/7/0 |
| 37 | 0.0083 | 0.0515 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9951 | 0.9722 | 33/7/0 |
| 38 | 0.0080 | 0.0469 | 0.9861 | 0.9000 | 0.9949 | 0.9474 | 0.9953 | 0.9721 | 36/4/2 |
| 39 | 0.0080 | 0.0494 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9946 | 0.9718 | 34/6/0 |
| 40 | 0.0094 | 0.0509 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9952 | 0.9699 | 33/7/0 |
| 41 | 0.0079 | 0.0525 | 0.9884 | 0.8750 | 1.0000 | 0.9375 | 0.9946 | 0.9682 | 35/5/0 |
| 42 | 0.0071 | 0.0603 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9942 | 0.9638 | 31/9/0 |
| 43 | 0.0082 | 0.0644 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9952 | 0.9706 | 31/9/0 |
| 44 | 0.0072 | 0.0533 | 0.9814 | 0.8000 | 1.0000 | 0.9000 | 0.9937 | 0.9668 | 32/8/0 |
| 45 | 0.0079 | 0.0517 | 0.9884 | 0.8750 | 1.0000 | 0.9375 | 0.9944 | 0.9669 | 35/5/0 |
| 46 | 0.0069 | 0.0499 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9949 | 0.9703 | 34/6/0 |
| 47 | 0.0084 | 0.0484 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9948 | 0.9710 | 34/6/0 |
| 48 | 0.0079 | 0.0625 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9950 | 0.9696 | 31/9/0 |
| 49 | 0.0082 | 0.0513 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9948 | 0.9709 | 34/6/0 |
| 50 | 0.0063 | 0.0466 | 0.9861 | 0.8750 | 0.9974 | 0.9362 | 0.9944 | 0.9695 | 35/5/1 |

Validation sensitivity is unstable across epochs (0.300–0.950 from epoch 4 on; 0.300 at epoch 23) with only 40 RD clips; one clip moves sensitivity by 0.025. Epoch 15 was selected on this same validation split, so its metrics are optimistically biased as an estimate of performance on new data; after the learning-rate drops (epochs 27–50) accuracy stayed in 0.977–0.988 and sensitivity in 0.750–0.900.

### Errors of the selected checkpoint (epoch 15, validation, threshold 0.5)

From `best_validation_predictions.csv` (`-train-04`). Labels: 1 = RD, 0 = non-RD.

| Clip | Dataset label | Predicted | RD probability | Error |
| --- | --- | --- | --- | --- |
| 825315_00096 | 0 | 1 | 0.6220 | false positive |
| 825315_00102 | 0 | 1 | 0.6091 | false positive |
| 755384_00001 | 1 | 0 | 0.1300 | false negative |
| 102769_00005 | 1 | 0 | 0.0795 | false negative |

Observations (not yet explained):
- Both false positives are just above the 0.5 threshold and share the clip-ID prefix `825315`, so they may come
  from the same source recording; they are probably not independent errors.
- Both false negatives are confident misses (probability ≤ 0.13); lowering the threshold enough to catch them
  would add many false positives.
- These clips must **not** be added to training: they belong to the validation split used for checkpoint
  selection and reporting, so training on them would leak and inflate the reported metrics.
- Suggested follow-up: Grad-CAM (`backbone.layer4`, epoch-15 `best.pt`) on these four clips. Augmentation,
  class weighting or threshold changes would be a new, separately labelled run agreed by the team, with
  threshold choices made on validation and the result reported on test.

Reference for later comparison (U-Net++ epoch 48, **validation**): accuracy 0.988399,
sensitivity 0.925000. The R(2+1)D-18 test result is reported separately below as a **reused official test benchmark**.

## Reused official test benchmark (single frozen evaluation, 2026-11-13)

Private Kaggle job `namanbabariya/r2plus1d18-rd-001-test-evaluation` version 1 (Tesla T4, torch
2.11.0+cu130; runtime identical to training) evaluated the frozen **epoch 15** checkpoint
(`c041f7e2…3ec16`) once at **threshold 0.5** on all **1,077** official test clips (100 RD / 977 non-RD).
No training, threshold search or epoch selection. The official test split was already used for the
U-Net++ result, so this is a **reused official test benchmark**, not an untouched holdout.

Freeze: `configs/r2plus1d18_rd_001_evaluation.json`, committed before the job (`70f32af`); the gate was set
in `e5ee644`, confirmed by the R(2+1)D-18 owner. It fixes this model's choices only; a team-wide freeze of
all models is not recorded there.

| Reused official test | R(2+1)D-18 epoch 15 | U-Net++ epoch 48 (reference) |
| --- | ---: | ---: |
| TP / TN / FP / FN | 92 / 962 / 15 / 8 | 86 / 964 / 13 / 14 |
| Sensitivity | **0.9200** | 0.8600 |
| Specificity | 0.9846 | 0.9867 |
| Precision | 0.8598 | 0.8687 |
| F1 | 0.8889 | 0.8643 |
| Accuracy | 0.9786 | 0.9749 |
| Balanced accuracy | 0.9523 | 0.9233 |
| Average precision | 0.9589 | 0.9268 |
| ROC-AUC | 0.9933 | 0.9869 |
| BCE loss | 0.0654 | 0.0813 |

Inference timing (Tesla T4, physical batch 4, FP16 autocast): model forward 102.6 s for 1,077 clips,
**91.4 ms per clip** excluding the first (warm-up) batch of 4.49 s; 111.4 s including loading cached
tensors from disk. Inputs are pre-decoded `[1,96,128,128]` float32 tensors, so video decoding is excluded.

Verification (local, independent of the notebook): all 16 artifact hashes match `artifact_sha256.json`;
1,077 unique prediction rows match the test manifest's clip order and labels (977/100); thresholded labels
match the probabilities; TP/TN/FP/FN and every metric above were recomputed from the predictions with
scikit-learn and match `test_metrics.json` exactly; the saved freeze equals the tracked config; the
test-input provenance carries the pinned preparation-freeze digest `33c392ef…`.

Paired comparison with U-Net++ on the same 1,077 clips (U-Net++ predictions downloaded from
`bijaykhadka1710/unetpp-rd-002-test-evaluation/1`, hashes verified): R(2+1)D-18 alone was correct on 15
clips, U-Net++ alone on 11, both wrong on 12; exact McNemar p = 0.56, so overall accuracy is **not**
distinguishable. On the 100 RD clips, R(2+1)D-18 alone caught 6 and U-Net++ alone 0 (exact two-sided
p = 0.031). This is one post-hoc, unadjusted test on one seed and clip-level splits; treat it as suggestive,
not established. R(2+1)D-18 also made 2 more false positives.

Error pattern (observed, not explained): 11 of the 15 false positives have clip prefix `825315`, the same
prefix as both validation false positives; 4 of the 8 missed RD clips have prefix `102769`, as did one
validation miss. These clips may share source recordings; clip-level splits can place related clips in
different partitions.

Local copies (ignored): `outputs/kaggle_results/r2p1_test/` and `outputs/kaggle_results/unetpp_test_ref/`.

### How the test stage was built

- `src/rd_ultrasound_ml/comparison_evaluation.py` (model-agnostic, reusable for R3D-18) verifies the
  training artifact **before** opening `test.csv`, then reuses the U-Net++ test-input cache
  (`bijaykhadka1710/unetpp-rd-002-test-inputs/1`) without relabeling it: the cache's own freeze must hash
  to its manifest and to the pinned digest, the official archive MD5s and `test.csv` hash are re-checked,
  and all 1,077 tensors, labels, order and the preprocessing identity are verified. One forward pass per
  clip, no optimizer, model state checked unchanged; inference timing recorded.
- `scripts/build_r2plus1d18_evaluation_notebook.py` → `notebooks/kaggle_r2plus1d18_evaluation.ipynb`;
  `scripts/prepare_r2plus1d18_evaluation_job.py` writes the private, GPU, internet-off bundle pinned to
  `namanbabariya/r2plus1d18-rd-001-train-04/1` and the test-input output, and refuses unless
  `team_freeze_confirmed` is true. The GPU runtime must equal training or the job stops before test access.
- `tests/test_r2plus1d18_evaluation.py` (7 tests, synthetic CPU). Run with pandas 2.3.3 (Kaggle's version)
  because Windows Application Control blocked pandas 3.1.1's DLL on this machine.

## Limitations and deviations

- This compares architectures under the inherited U-Net++ recipe. The small learning rate may disadvantage this model, so the result is not this model's tuned best.
- BatchNorm3d sees 4 clips per forward pass. Accumulation does not make BatchNorm see 48. U-Net++ used InstanceNorm, so this is not a normalization-controlled comparison.
- FP16 mixed precision. Kaggle runtime versions may differ from the U-Net++ run; they are recorded, not assumed.
- One seed and clip-level official splits with possible patient overlap. No unseen-patient or clinical claim.
- torchvision R(2+1)D-18 is not an ERDES Table 8 row.
