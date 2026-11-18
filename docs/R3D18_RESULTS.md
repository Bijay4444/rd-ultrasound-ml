# R3D-18 RD comparison: `r3d18_rd_001_amp`

Updated 2026-11-18. **Status: training complete, 50 of 50 epochs (`run_status.json` status `completed`), on
Kaggle Tesla T4 over three sessions: `namanbabariya/r3d18-rd-001-train-01` epoch 1, `-train-02` epochs 2–36,
`-train-03` epochs 37–50. Selected checkpoint: epoch 18. All metrics are on the official validation split, which
was also used for checkpoint selection. The official test split has not been read.** See the
[shared protocol](MODEL_COMPARISON_PLAN.md) and [assignment](handoffs/R3D18.md).

Implementation (model, config, notebook builder, job script, tests and the first version of this report) by the
R3D-18 owner on branch `feature/r3d18-rd`, starting commit `b96cb39ab90d0273288e92ee20f9a87787f3ec8d` (tip of
`dev`, which already contains the merged R(2+1)D-18 work). The GPU training was run on 2026-11-17/18 by the
R(2+1)D-18 owner under their Kaggle account, with the team informed, using the unchanged tracked notebook
(rebuilding it from `scripts/build_r3d18_notebook.py` produced no diff). Results below come from the saved Kaggle
logs only.

## Blockers (resolved)

| Needed | Status |
| --- | --- |
| GPU | Resolved 2026-11-17: Kaggle Tesla T4 under the `namanbabariya` account, requested by the bundle (`machine_shape: NvidiaTeslaT4`); all three sessions got the T4. |
| Kaggle credentials | Configured in the ignored `.env.kaggle` of that account. |
| Prepared cache access | Resolved: `bijaykhadka1710/unetpp-rd-002-inputs-00/2` and `-01/2`, the same version-pinned outputs used by R(2+1)D-18; both reported COMPLETE before every submission. |
| Physical-batch coordination | Resolved: the real-clip probe passed at physical batch 4 with 3.31 GB peak allocated, so the inherited 4 × 12 is kept. This matches R(2+1)D-18, which also trained at 4 × 12. |

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
  stem removes 2*64*3*7*7 = 18,816). Verified locally with torchvision 0.29.2; the Kaggle run (torchvision
  0.26.0+cu130) passed the notebook's parameter-count check.
- Input `[B,1,96,128,128]`, grayscale `[0,1]` from the existing cache; no Kinetics normalization.

### Grad-CAM integration (for Bijay)

- Module path: **`backbone.layer4`** (`gradcam_layer(model)` returns it).
- Feature shape on the full input: **`[B, 512, 12, 8, 8]`** (C, D, H, W). Measured on CPU with `[1,1,96,128,128]`
  and confirmed on the GPU on real validation clip `825315_00443` (`gradcam_layer.json`).
- Target the scalar RD logit (model output before sigmoid). Use `model.eval()` with autograd enabled.
- Use the epoch-18 `best.pt` from the `-train-03` output (SHA256 below).

## Cache reuse without reusing U-Net++ identity

Nothing in the shared cache code changed. `src/rd_ultrasound_ml/video_cache.py::load_cache_for_run` (added by the
R(2+1)D owner, model-agnostic) validates the original preparation config against its pinned digest
(`6907d5f9...`), then calls the unchanged `load_prepared_shards`, so every tensor/source hash, label, order and shape
check still runs, and the pinned cache fingerprint (`111e9cb7...`) must match. It returns a run fingerprint that binds
the preparation identity, the cache fingerprint and **this model's own training-config digest**, so an R3D-18 run can
never resume from, or be mixed with, another model's state. `configs/r3d18_rd_001_training.json` carries the same
pins as the R(2+1)D config. On Kaggle, all three sessions verified all 4,304 attached clips and returned the pinned
cache fingerprint and the same run fingerprint `c0eb4bbf…` (`input_fingerprint.json`).

## Protocol (config `configs/r3d18_rd_001_training.json`)

Identical to the R(2+1)D-18 config in every field except the experiment id and the model block (checked field by field,
and checked against `unetpp_rd_002_training.json` by `tests/test_r3d18.py`): seed 42, 50 epochs, 3,873 train / 431
validation official clips, Adam lr 1.5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0, unweighted
`BCEWithLogitsLoss`, ReduceLROnPlateau on validation loss (factor 0.1, patience 10), effective batch 48 = physical 4 x
accumulation 12, FP16 autocast + FP32 master weights + GradScaler, threshold 0.5, highest validation accuracy with
earliest ties. No augmentation, pretrained weights, class weighting or resampling. The config was not edited for the
GPU run; the probe passed at batch 4, so no physical-batch change was needed (training config digest
`69e65651…`).

## Checks actually run

### Local, CPU, synthetic data (R3D-18 owner, 2026-11-16)

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

Full existing suite (`discover -s tests`, 99 tests including the R(2+1)D and U-Net++ tests): **all pass**.

Separate manual check: a full-size `[2,1,96,128,128]` forward + BCE + backward on this Mac's CPU took 116 s; loss 0.706,
62/62 parameter tensors had finite nonzero gradients. This is CPU code evidence, not a GPU timing.

The unit tests were not re-run on the Windows machine used for the GPU run (no local torch environment there).

### On the downloaded GPU outputs (2026-11-18, local, independent of the notebook)

- Every downloaded small file matched `artifact_sha256.json` in each session (20, 55, 69 files; 0 mismatches).
  Checkpoint `.pt` files and the results ZIP were not downloaded locally.
- History row 1 is identical in all three sessions, and rows 1–36 of `-train-03` equal `-train-02` exactly, so
  resumed sessions carried earlier epochs over unchanged.
- The recorded runtime (GPU, Python, torch, torchvision, CUDA, cuDNN and other libraries) is identical in all three
  sessions, as `run_training` requires for resume.
- The three misclassified clips listed below were read directly from `best_validation_predictions.csv` (431 rows,
  40 RD), consistent with the TP/TN/FP/FN of epoch 18 in `history.csv`.

## How to run

```bash
python scripts/build_r3d18_notebook.py      # only if src/ or the config changed
python scripts/prepare_r3d18_job.py --inputs OWNER/unetpp-rd-002-inputs-00/V OWNER/unetpp-rd-002-inputs-01/V
python scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/r3d18-rd-001-train-01
# after session 1 completes and the probe/epoch timing is reviewed:
python scripts/prepare_r3d18_job.py --inputs ... --session 2 --epochs N
```

Commands actually used for this run (inputs `bijaykhadka1710/unetpp-rd-002-inputs-00/2` and `-01/2`):
session 1 `--session 1 --epochs 1`, session 2 `--session 2 --epochs 35`, session 3 `--session 3 --epochs 14`.

Windows note: from session 3 on, Windows Application Control blocked the `kaggle.exe` launcher that
`scripts/kaggle_cli.py` creates through `uv` (os error 4551). The same Kaggle CLI (`kaggle==2.2.5`, same
credentials from `.env.kaggle`) was then called through Python (`kaggle.cli.main`) instead. That path skips
`kaggle_cli.py`'s pre-push dependency check, so the three attached inputs were confirmed COMPLETE by hand before
pushing session 3. `scripts/kaggle_cli.py` itself was not changed.

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

## Results

| Item | Value |
| --- | --- |
| Runtime | Kaggle Tesla T4 (15.6 GB), Python 3.13.16, torch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, cuDNN 92700 (same as the R(2+1)D-18 run) |
| Cache checks | 4,304/4,304 attached clips verified in every session; preparation digest `6907d5f9…` and cache fingerprint `111e9cb7…` matched pins; run fingerprint `c0eb4bbf…`; `test_used: false` |
| Precision probe | physical batch 4, 16-mixed: passed first try; median step 0.536 s; peak allocated 3.31 GB, reserved 4.15 GB; finite nonzero gradients; 0 skipped updates. Batch 4 fits, so the inherited 4 × 12 is kept. |
| First full epoch wall time | 875.2 s (≈14.6 min) including 56.3 s validation; epoch peak allocated 3.31 GB, reserved 4.45 GB |
| Later epoch wall time | ≈10.6–15.1 min including validation (≈11.4 min typical in session 2, ≈12.5 min in session 3) |
| Session overhead | whole session 1 took 1,837 s; ≈15 min of each session is the per-clip cache verification at start |
| **Selected epoch 18, validation (431 clips: 40 RD, 391 non-RD), threshold 0.5** | TP 37 / TN 391 / FP 0 / FN 3; accuracy **0.9930**, sensitivity **0.925**, specificity 1.0000, precision 1.0000, F1 0.9610, balanced accuracy 0.9625, AP 0.9694, ROC-AUC 0.9895, BCE loss 0.0643 |
| Final epoch 50, validation (not selected) | TP 36 / TN 390 / FP 1 / FN 4; accuracy 0.9884, sensitivity 0.900, specificity 0.9974, balanced accuracy 0.9487, ROC-AUC 0.9933 |
| Total GPU training time | 50 epochs summed 9.47 h training + 0.40 h validation = 9.87 h (`history.csv`), plus ≈15 min cache verification per session (3 sessions); session wall times 0.51 h, 6.68 h, 3.12 h |
| Epochs completed / 50 | **50** (`run_status.json` status `completed`); sessions 2–3 each resumed with the same run fingerprint `c0eb4bbf…` |
| Selected checkpoint (final) | **epoch 18** by highest validation accuracy (earliest tie rule; no later epoch reached 0.9930). `best.pt` SHA256 `5422add5a8ce4544e4c9f6bf0190fa57702520bed55caea5738398ac7d363152`; final-session reload reproduced validation probabilities exactly (max diff 0.0) and the confusion counts |
| Memory / stability | epoch peak allocated 3.32 GB, reserved 4.51 GB of 15.6 GB; 1 AMP skipped update in 50 epochs (epoch 27). ReduceLROnPlateau: 1.5e-5 → 1.5e-6 (epoch 41); no further reduction by epoch 50 |
| Validation by epoch | see table below |
| Grad-CAM layer (real clip) | `backbone.layer4`, `[1,512,12,8,8]` for input `[1,1,96,128,128]` |
| Artifact integrity | every downloaded small file matched `artifact_sha256.json` in each session; checkpoint `.pt` files were not downloaded locally |
| Inference timing | pending (would come from a frozen test-evaluation job, as for R(2+1)D-18) |
| Private artifact location | Kaggle output of `namanbabariya/r3d18-rd-001-train-03` (final `best.pt`, `last_*.pt`, full history and per-epoch validation files); sessions 1–2 outputs hold intermediate checkpoints. Local small-file copies under ignored `outputs/kaggle_results/r3d_s0{1,2,3}_small/` |

### Validation by epoch (official validation split, 431 clips, threshold 0.5)

From `history.csv` of `-train-03` (rows 1–36 identical to `-train-02`). Selection rule: highest accuracy, earliest tie.

| Epoch | Train loss | Val loss | Acc | Sens | Spec | Bal. acc | ROC-AUC | AP | TP/FN/FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3913 | 0.3021 | 0.9072 | 0.0000 | 1.0000 | 0.5000 | 0.6912 | 0.2398 | 0/40/0 |
| 2 | 0.2790 | 0.2557 | 0.9165 | 0.5000 | 0.9591 | 0.7295 | 0.8797 | 0.5670 | 20/20/16 |
| 3 | 0.2073 | 0.1794 | 0.9466 | 0.6500 | 0.9770 | 0.8135 | 0.9051 | 0.7329 | 26/14/9 |
| 4 | 0.1731 | 0.1193 | 0.9652 | 0.7000 | 0.9923 | 0.8462 | 0.9603 | 0.8584 | 28/12/3 |
| 5 | 0.1370 | 0.1165 | 0.9652 | 0.7500 | 0.9872 | 0.8686 | 0.9788 | 0.8899 | 30/10/5 |
| 6 | 0.1302 | 0.1109 | 0.9629 | 0.6750 | 0.9923 | 0.8337 | 0.9646 | 0.8849 | 27/13/3 |
| 7 | 0.1088 | 0.0707 | 0.9791 | 0.8500 | 0.9923 | 0.9212 | 0.9928 | 0.9517 | 34/6/3 |
| 8 | 0.0992 | 0.0811 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9916 | 0.9407 | 31/9/0 |
| 9 | 0.0821 | 0.0728 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9969 | 0.9751 | 34/6/0 |
| 10 | 0.0767 | 0.0830 | 0.9814 | 0.8000 | 1.0000 | 0.9000 | 0.9873 | 0.9349 | 32/8/0 |
| 11 | 0.0743 | 0.0771 | 0.9768 | 0.7750 | 0.9974 | 0.8862 | 0.9917 | 0.9433 | 31/9/1 |
| 12 | 0.0671 | 0.0866 | 0.9745 | 0.7250 | 1.0000 | 0.8625 | 0.9798 | 0.9188 | 29/11/0 |
| 13 | 0.0554 | 0.0606 | 0.9791 | 0.8000 | 0.9974 | 0.8987 | 0.9946 | 0.9648 | 32/8/1 |
| 14 | 0.0469 | 0.0612 | 0.9745 | 0.7500 | 0.9974 | 0.8737 | 0.9942 | 0.9597 | 30/10/1 |
| 15 | 0.0401 | 0.0541 | 0.9814 | 0.8750 | 0.9923 | 0.9337 | 0.9944 | 0.9695 | 35/5/3 |
| 16 | 0.0412 | 0.0616 | 0.9791 | 0.7750 | 1.0000 | 0.8875 | 0.9965 | 0.9751 | 31/9/0 |
| 17 | 0.0368 | 0.0985 | 0.9722 | 0.7000 | 1.0000 | 0.8500 | 0.9767 | 0.9252 | 28/12/0 |
| 18 | 0.0297 | 0.0643 | 0.9930 | 0.9250 | 1.0000 | 0.9625 | 0.9895 | 0.9694 | 37/3/0 |
| 19 | 0.0281 | 0.0550 | 0.9884 | 0.9000 | 0.9974 | 0.9487 | 0.9815 | 0.9702 | 36/4/1 |
| 20 | 0.0254 | 0.0794 | 0.9722 | 0.7750 | 0.9923 | 0.8837 | 0.9843 | 0.9503 | 31/9/3 |
| 21 | 0.0182 | 0.1077 | 0.9606 | 0.9000 | 0.9668 | 0.9334 | 0.9877 | 0.9350 | 36/4/13 |
| 22 | 0.0226 | 0.0593 | 0.9814 | 0.8750 | 0.9923 | 0.9337 | 0.9806 | 0.9602 | 35/5/3 |
| 23 | 0.0177 | 0.0653 | 0.9838 | 0.8750 | 0.9949 | 0.9349 | 0.9889 | 0.9516 | 35/5/2 |
| 24 | 0.0202 | 0.0684 | 0.9884 | 0.8750 | 1.0000 | 0.9375 | 0.9809 | 0.9539 | 35/5/0 |
| 25 | 0.0160 | 0.0436 | 0.9838 | 0.8500 | 0.9974 | 0.9237 | 0.9949 | 0.9703 | 34/6/1 |
| 26 | 0.0119 | 0.0507 | 0.9791 | 0.9000 | 0.9872 | 0.9436 | 0.9947 | 0.9675 | 36/4/5 |
| 27 | 0.0095 | 0.0796 | 0.9745 | 0.7250 | 1.0000 | 0.8625 | 0.9895 | 0.9650 | 29/11/0 |
| 28 | 0.0104 | 0.0917 | 0.9698 | 0.9250 | 0.9744 | 0.9497 | 0.9926 | 0.9565 | 37/3/10 |
| 29 | 0.0119 | 0.0428 | 0.9907 | 0.9000 | 1.0000 | 0.9500 | 0.9947 | 0.9744 | 36/4/0 |
| 30 | 0.0089 | 0.0618 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9905 | 0.9503 | 33/7/0 |
| 31 | 0.0065 | 0.0465 | 0.9907 | 0.9000 | 1.0000 | 0.9500 | 0.9946 | 0.9698 | 36/4/0 |
| 32 | 0.0071 | 0.0600 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9857 | 0.9679 | 34/6/0 |
| 33 | 0.0066 | 0.0656 | 0.9791 | 0.9000 | 0.9872 | 0.9436 | 0.9738 | 0.9542 | 36/4/5 |
| 34 | 0.0056 | 0.0675 | 0.9791 | 0.8000 | 0.9974 | 0.8987 | 0.9811 | 0.9475 | 32/8/1 |
| 35 | 0.0060 | 0.2027 | 0.9188 | 0.1250 | 1.0000 | 0.5625 | 0.9644 | 0.9153 | 5/35/0 |
| 36 | 0.0070 | 0.1266 | 0.9582 | 0.9250 | 0.9616 | 0.9433 | 0.9797 | 0.9356 | 37/3/15 |
| 37 | 0.0186 | 0.1523 | 0.9258 | 0.2000 | 1.0000 | 0.6000 | 0.9800 | 0.9075 | 8/32/0 |
| 38 | 0.0071 | 0.0551 | 0.9861 | 0.8750 | 0.9974 | 0.9362 | 0.9907 | 0.9607 | 35/5/1 |
| 39 | 0.0137 | 0.0647 | 0.9814 | 0.8750 | 0.9923 | 0.9337 | 0.9735 | 0.9428 | 35/5/3 |
| 40 | 0.0100 | 0.0468 | 0.9838 | 0.8500 | 0.9974 | 0.9237 | 0.9901 | 0.9684 | 34/6/1 |
| 41 | 0.0048 | 0.0414 | 0.9861 | 0.9000 | 0.9949 | 0.9474 | 0.9907 | 0.9732 | 36/4/2 |
| 42 | 0.0038 | 0.0536 | 0.9838 | 0.8250 | 1.0000 | 0.9125 | 0.9879 | 0.9715 | 33/7/0 |
| 43 | 0.0048 | 0.0771 | 0.9722 | 0.7000 | 1.0000 | 0.8500 | 0.9883 | 0.9641 | 28/12/0 |
| 44 | 0.0038 | 0.0412 | 0.9884 | 0.9000 | 0.9974 | 0.9487 | 0.9929 | 0.9731 | 36/4/1 |
| 45 | 0.0041 | 0.0396 | 0.9884 | 0.9000 | 0.9974 | 0.9487 | 0.9889 | 0.9735 | 36/4/1 |
| 46 | 0.0034 | 0.0392 | 0.9884 | 0.9000 | 0.9974 | 0.9487 | 0.9919 | 0.9733 | 36/4/1 |
| 47 | 0.0044 | 0.0398 | 0.9861 | 0.8750 | 0.9974 | 0.9362 | 0.9907 | 0.9724 | 35/5/1 |
| 48 | 0.0031 | 0.0549 | 0.9861 | 0.8500 | 1.0000 | 0.9250 | 0.9877 | 0.9708 | 34/6/0 |
| 49 | 0.0039 | 0.0434 | 0.9838 | 0.9000 | 0.9923 | 0.9462 | 0.9894 | 0.9716 | 36/4/3 |
| 50 | 0.0029 | 0.0387 | 0.9884 | 0.9000 | 0.9974 | 0.9487 | 0.9933 | 0.9738 | 36/4/1 |

Validation sensitivity is unstable across epochs (0.125–0.925 from epoch 4 on; 0.125 at epoch 35 and 0.200 at
epoch 37) with only 40 RD clips; one clip moves sensitivity by 0.025. Training loss fell to ≈0.003–0.007 from
epoch 31 on while validation swung, which indicates fitting the training set closely. Epoch 18 was selected on this
same validation split, so its metrics are optimistically biased as an estimate of performance on new data; after the
learning-rate drop (epochs 41–50) accuracy stayed in 0.972–0.988 and sensitivity in 0.700–0.900.

### Errors of the selected checkpoint (epoch 18, validation, threshold 0.5)

From `best_validation_predictions.csv` (`-train-03`). Labels: 1 = RD, 0 = non-RD. No false positives.

| Clip | Dataset label | Predicted | RD probability | Error | Also missed by R(2+1)D-18 epoch 15? |
| --- | --- | --- | --- | --- | --- |
| 755384_00001 | 1 | 0 | 0.0276 | false negative | yes (0.1300) |
| 102769_00040 | 1 | 0 | 0.1727 | false negative | no |
| 102769_00005 | 1 | 0 | 0.3912 | false negative | yes (0.0795) |

Observations (not yet explained):
- Two of the three misses are the same clips R(2+1)D-18 missed (`docs/R2PLUS1D18_RESULTS.md`), so they may be hard
  cases for both video models rather than model-specific errors.
- Two misses share the clip-ID prefix `102769`, so they may come from the same source recording; they are probably not
  independent errors. R(2+1)D-18 also had 4 of its 8 test misses under that prefix.
- R3D-18's two validation false positives are zero, while R(2+1)D-18 epoch 15 had 2 (both prefix `825315`).
- These clips must **not** be added to training: they belong to the validation split used for checkpoint selection
  and reporting, so training on them would leak and inflate the reported metrics.
- Suggested follow-up: Grad-CAM (`backbone.layer4`, epoch-18 `best.pt`) on these three clips.

### Validation comparison (official validation split, threshold 0.5; selection also used validation)

| Model | Accuracy | Sensitivity | Specificity | TP/FN/FP | Source |
| --- | ---: | ---: | ---: | --- | --- |
| U-Net++ (epoch 48) | 0.9884 | 0.9250 | — | — | `docs/UNETPP_RESULTS.md` |
| R3D-18 (epoch 18) | 0.9930 | 0.9250 | 1.0000 | 37/3/0 | this report |
| R(2+1)D-18 (epoch 15) | 0.9907 | 0.9500 | 0.9949 | 38/2/2 | `docs/R2PLUS1D18_RESULTS.md` |
| U-Net++ + augmentation | pending | pending | pending | pending | Bijay |

The differences between the three models are one or two clips out of 431 on one seed, with checkpoints selected on
this same split; they do not establish that any one model is better. Summed per-epoch training time (`epoch_memory.json`, validation
excluded) was 9.47 h for R3D-18 versus 22.6 h for R(2+1)D-18 under the same recipe, hardware and Kaggle image, and
R3D-18's peak allocated GPU memory was 3.32 GB versus 8.51 GB.

Official-test numbers are not reported here. The test split is not touched until the team freezes all final choices;
a frozen R3D-18 test run would be labelled a **reused official test benchmark**. The model-agnostic
`comparison_evaluation.py` exists (added for R(2+1)D-18), but an R3D-18 evaluation config, gated notebook, job script
and tests (modelled on the R(2+1)D-18 ones) still need to be built for it.

## Limitations and deviations

- This compares architectures under the inherited U-Net++ recipe. The small learning rate may disadvantage this model,
  so the result is not its tuned best.
- BatchNorm3d sees 4 clips per forward pass. Accumulation does not make BatchNorm see 48. U-Net++ used InstanceNorm, so
  this is not a normalization-controlled comparison.
- FP16 mixed precision. Kaggle runtime versions may differ from the U-Net++ run; they are recorded, not assumed.
- One seed and clip-level official splits with possible patient overlap. No unseen-patient or clinical claim.
- torchvision R3D-18 is not the ERDES paper's deeper `[4,4,4,4]` MONAI ResNet.
- These are validation results of an experimental model trained for a course research project. They are not evidence
  of clinical diagnostic performance.
