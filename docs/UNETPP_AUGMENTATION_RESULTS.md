# U-Net++ training augmentation

Updated 2026-11-22. **All 50 epochs completed and passed independent verification.**
The automatic 1 + 25 + 24 epoch handoffs finished on November 19. The final source
is `bijaykhadka1710/unetpp-rd-aug-001-train-03/1`; epoch **32** was selected by
maximum validation accuracy with earliest ties, at the fixed threshold **0.5**.
This policy did not improve validation performance. Preserve this negative
result rather than changing the policy in response to the test benchmark.

The original baseline and augmented run used the same runtime, inputs and
training recipe. After confirming the other models' freezes and independently
verifying the peer R3D-18 handoff, the separate single frozen augmentation test
evaluation completed on November 22 and passed saved-artifact verification.
Earlier first-epoch and runtime-incident sections are
historical records, not the current completion status.

## Completed validation comparison

Both models completed 50 epochs; selection used 431 clean validation clips
(40 RD, 391 non-RD), not test scores.

| Metric | Baseline U-Net++, epoch 48 | Augmented U-Net++, epoch 32 |
| --- | ---: | ---: |
| TP / TN / FP / FN | 37 / 389 / 2 / 3 | 36 / 388 / 3 / 4 |
| Sensitivity | 92.50% | 90.00% |
| Specificity | 99.49% | 99.23% |
| Precision | 94.87% | 92.31% |
| F1 | 93.67% | 91.14% |
| Accuracy | 98.84% | 98.38% |
| Balanced accuracy | 95.99% | 94.62% |
| Average precision | 0.973139 | 0.953729 |
| ROC-AUC | 0.996164 | 0.990249 |

The models are both correct on 424 clips and both wrong on five. Augmentation
adds one false positive (`825315_00096`) and one false negative (`258033_00001`),
and corrects no baseline error at threshold 0.5. This describes one policy and
one seed; it does not establish that augmentation generally harms performance.

The final audit verified **139 artifact hashes**, all 50 sets of 431 validation
scores, source/config/runtime identities, original train/validation membership,
selected checkpoint, finite weights/optimizer state and the saved GPU reload
check (maximum probability difference 0). Adam completed 4,048 of 4,050 attempted
updates; two were skipped by AMP. Baseline completed 4,049 with one skipped.
Epoch training/validation segment times sum to 5.34 hours for augmentation and
4.69 hours for baseline; setup, probes and exports are excluded.

Final archive:
`outputs/augmentation/training_pipeline/results/session_03_v1/unetpp_rd_aug_001_amp_results.zip`.
SHA256: `fc463a26460cea639c126cc0877da87dc29b801868ad87539efa46e6a3b499d5`.
Selected checkpoint SHA256:
`2e398e36340af01077fb900e3df522a048ad4b854e4812f0f070a4772fbbbaa5`.
Audits and checked extraction are beside that archive. Validation figures
(PNG/PDF), metrics, paired predictions and error CSVs are under
`outputs/augmentation/validation_comparison_20261122/`. Generate them with
`scripts/analyze_augmentation_results.py --archive <final-archive> --output <new-folder>`
using the uv environment documented below; the command verifies both runs first.

## Completed reused official test benchmark

The private, Internet-disabled job
`bijaykhadka1710/unetpp-rd-aug-001-test-evaluation/1` completed one clean-input
evaluation of epoch 32. It used the original baseline Docker image and the
version-pinned original test cache; no optimizer, augmentation, calibration or
threshold search ran. All 17 test artifact hashes, exact freeze/source/runtime,
shared cache provenance, 1,077 ordered rows and recomputed metrics passed.

| Metric | Baseline U-Net++, epoch 48 | Augmented U-Net++, epoch 32 |
| --- | ---: | ---: |
| TP / TN / FP / FN | 86 / 964 / 13 / 14 | 87 / 967 / 10 / 13 |
| Sensitivity | 86.00% | 87.00% |
| Specificity | 98.67% | 98.98% |
| Precision | 86.87% | 89.69% |
| F1 | 86.43% | 88.32% |
| Accuracy | 97.49% | 97.86% |
| Balanced accuracy | 92.33% | 92.99% |
| Average precision | 0.926845 | 0.918940 |
| ROC-AUC | 0.986858 | 0.979591 |

At threshold 0.5 augmentation has one fewer missed RD clip and three fewer false
positives; AP and ROC-AUC are lower. Eight baseline errors are corrected and
four previously correct clips become wrong; 19 remain wrong and 1,046 correct
under both models. Together with the adverse validation result, this is a mixed
finding for one mild policy and one seed. Do not tune the threshold or policy on
these results or claim a general improvement.

The saved inference took 31.00 seconds including disk loading; model forward
time excluding the first batch was 19.65 ms/clip on a Tesla T4, batch 4, FP16
autocast. The original baseline does not have an identical forward-only timing
record, so this does not establish a speed comparison.

Archive: `outputs/augmentation/test_evaluation_v1/unetpp_rd_aug_001_test_results.zip`.
SHA256: `04fbd794cd0fe123789207e1af1525025d677bbce89c217f7471cee7d4ccf386`.
The evaluation log is beside it. The matching freeze is
`configs/unetpp_rd_aug_001_evaluation.json`; builder:
`scripts/prepare_augmentation_evaluation.py`; standalone notebook:
`notebooks/kaggle_unetpp_augmentation_evaluation.ipynb`.
Full four-model tables, sources, verified plots and reproducible saved-results
comparison commands are in [FINAL_MODEL_COMPARISON.md](FINAL_MODEL_COMPARISON.md).
No new baseline or peer inference was performed for these tables.

## Question and controlled comparison

Does mild, temporally consistent, training-only augmentation change U-Net++ RD
sensitivity, specificity and average precision under the existing recipe?

Compare `unetpp_rd_002_amp` with new run `unetpp_rd_aug_001_amp`. The new run
starts from fresh weights, not the epoch-48 checkpoint. Architecture, seed 42,
3873/431 train/validation membership and order, full-video `[1,96,128,128]`
float32 cached preprocessing, Adam 1.5e-5, unweighted BCE, physical batch 4,
effective batch 48, FP16 AMP, scheduler, 50-epoch target, threshold 0.5 and
highest-validation-accuracy/earliest-tie selection are inherited unchanged.

Augmentation lives in `training_augmentation`; cached `preprocessing` retains
`augmentation: none`. The existing cache adapter verifies both original
preparation shards against the original digest and cache fingerprint. New
config/source identity is bound into resume checks. The notebook also requires
the baseline's recorded runtime and unchanged four baseline source hashes.
Runtime drift stops the run for explicit documentation.

The job metadata now pins the exact image retrieved from
`bijaykhadka1710/unetpp-rd-002-train-03/1`:
`gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435`.
The original four source hashes and runtime checks remain enforced. Image pinning
is supported by the installed official CLI's `docker_image` metadata handling;
see [Kaggle's environment documentation](https://www.kaggle.com/docs/notebooks).

## Policy chosen after training-only previews

| Transform | Resolved range | Probability per clip |
| --- | --- | ---: |
| Joint spatial affine | Rotation ±3°, translation ±1% per axis, isotropic scale 0.98–1.02 | 0.5 |
| Multiplicative intensity gain | 0.9–1.1; clamp to `[0,1]` | 0.5 |

Apply affine, then gain. Use bilinear spatial interpolation, zero padding and
`align_corners=False`. One parameter set applies to all 96 temporal positions;
there is no frame shuffling, reversal or temporal interpolation in augmentation.
Both labels use the same policy, without class weighting or resampling.

The original proposal used ±7°, ±3%, and 0.95–1.05. Eight sampled training
previews and combined-boundary stress views showed tight original framing and
visible border wedges/displacement, so the bounds were reduced **before** any
GPU augmentation training. The revised views were inspected again. This reduces
the perturbation; it does not establish clinical invariance. Interpolation,
clipping and corner trimming remain possible. Gain is a simple intensity
perturbation, not a physical simulation of ultrasound acquisition.

Frozen policy digest:
`f6baf624e9eae6468829b324355456241a72ab122de3934178ba57b3236a395b`.
See [the config](../configs/unetpp_rd_aug_001_training.json).

## Real previews and provenance

Reuse the eight previously fixed training-inspection IDs, selected before this
augmentation work: `164267_03674`, `164267_02201`, `825315_00216`,
`825315_00468`, `102769_00077`, `258033_00019`, `755384_00079`, `405744_00139`.
They cover four non-RD and four RD clips (two per inspected subtype); this is an
inspection panel, not the prevalence or a resampled training set.

Official metadata/split archive checksums, training membership, labels, source
sizes and all eight source SHA256s were checked before decoding. Original
source: `pcvlab/erdes`, revision `7867e2504ed3845a46ff7790a2f8c3ffcdc08947`;
author-declared CC-BY-4.0. Dataset/architecture attribution remains in
[the ERDES source notes](ERDES_PAPER_NOTES.md). No upstream code was modified.

Ignored artifacts under `outputs/augmentation/unetpp_rd_aug_001/`:

- `final_previews/original_vs_augmented.png`: original/augmented positions 0, 47, 95.
- `final_previews/boundary_stress.png`: original and two combined range endpoints.
- `final_previews/<clip_id>.gif`: paired original/augmented sampled playback.
- `final_previews/preview_manifest.json`: labels, parameters, per-case RNG seed,
  shapes/ranges, source hashes and original/augmented tensor hashes.
- `final_previews/array_sha256.json`, `visual_review.json`, and `runtime.json` in
  the parent run folder: input integrity, review observations and local runtime.

All eight outputs are finite float32 `[1,96,128,128]` in `[0,1]`; deterministic
replay and unchanged input arrays passed. The first preview process was interrupted
when the execution runtime was replaced after six saved inputs; that partial
folder is retained as `unetpp_rd_aug_001_interrupted`. A fresh retry completed all
eight. Initial larger-range previews and the intermediate figure are retained.
The final figure corrects overlapping row labels. Local decoding is preview-only;
it does not claim bitwise equality with Kaggle's earlier decoder runtime. GPU
training reads the verified original Kaggle cache and exports its own previews.

## Reproducibility and implementation

`video_augmentation.py` derives a local CPU RNG seed from SHA256 of
`video_augmentation_v1:<seed>:<epoch>:<clip_id>` (first eight bytes, big-endian,
masked to 63 bits). It does not consume the global training RNG. The wrapper
restores the epoch from the authoritative checkpoint, including a mid-epoch
cursor. A changed policy/config cannot resume the old run. CPU grid resampling
occurs on a fresh tensor; cache arrays and labels are not edited. Validation
receives the unwrapped dataset. Reproducibility on CUDA is not promised to be
bitwise merely because a seed is recorded.

Implementation files: `src/rd_ultrasound_ml/video_augmentation.py`,
`augmentation_preview.py`; scripts `preview_video_augmentation.py`,
`freeze_augmentation_config.py`, `build_augmentation_notebook.py`, and
`prepare_augmentation_job.py`; and
[the standalone notebook](../notebooks/kaggle_unetpp_augmentation.ipynb).
The baseline engine/configs/notebooks and completed artifacts remain preserved.

## Checks performed

- **24 augmentation checks passed:** pixel-space translation/rectangular rotation
  oracles, temporal consistency, identity/gain/range behavior, keyed replay,
  cache/label immutability, policy/partition guards, exact synthetic interrupted
  CPU training (model, optimizer, scheduler, scaler and metrics), frozen source
  hashes, matched config, notebook schema/source freshness, private version-pinned
  bundles, a runtime-drift refusal, and actual notebook training/reload/plot/ZIP
  cells on synthetic CPU data, refusal of corrupt result archives before
  extraction, and automatic continuation through verified boundaries. Failed jobs,
  failed verification, and uncertain submissions cannot launch a later job.
  Paired comparison alignment/error categories and the frozen evaluation
  notebook's source/schema, confirmation gate and private pinned bundle also pass.
- **7 shared comparison-evaluation checks passed**, including fixed-weight CPU
  inference, freeze-before-test guards and cache identity/corruption rejection.
- **9 existing training tests passed**, including accumulation and resume.
- **9 existing Kaggle tests passed**, including cache corruption/order checks,
  private submission metadata and required-source submission guards.
- Python compilation and `git diff --check` passed.

Synthetic metrics are implementation evidence, not ERDES results. The full GPU
run and final validation comparison are verified above. No test data were used
to choose augmentation bounds or select its checkpoint. The existing baseline
test inference is not repeated.

## Historical first epoch and measured budget

The matched-image run passed its real-input FP16 precision probe at physical
batch 4: finite nonzero gradients, no measured skipped updates, about 3.34 GB
peak allocated GPU memory and 4.12 GB reserved. Probe weights were discarded.
Epoch 1 took **488.13 seconds (8.14 minutes)**, including 43.47 seconds of clean
validation; no optimizer update was skipped. There are 81 completed Adam steps.

The epoch-1 validation predictions were all non-RD: TP 0, TN 391, FP 0, FN 40;
sensitivity 0, specificity 1, accuracy 0.907193, AP 0.101888 and ROC-AUC 0.461797.
This is an initial partial-training result, not evidence of an augmentation gain.
Do not compare it as an equal-budget result against the completed baseline.

Independent local verification checked **all 41 artifact hashes**, exact training
config and all embedded sources, full runtime equality, original cache/new-run
fingerprints, 3,873/431 official manifest membership/order/labels, all 431
validation probabilities and recomputed metrics, epoch/checkpoint selection,
finite weights/optimizer state, 81 Adam steps, and the saved GPU reload check
(maximum probability difference 0). Local GPU inference was not repeated; the
27 GB original cache stays on Kaggle.

Archive: `outputs/kaggle_results/unetpp_aug_001_session_01_v2/unetpp_rd_aug_001_amp_results.zip`.
SHA256: `3706ac6368956256d9a3b4c04941d1880e7e7277bc4fb7d5e0854713a7de5bc7`.
Audit: `verification.json` beside the archive; checked extraction: `verified/`.

At the observed timing, 49 remaining epochs project to **6.64 training hours**,
excluding setup, queueing, probes and export. That exceeds one six-hour saved-job
window, so continuation is bounded to at most 25 epochs, with additional timing
headroom and the notebook's own normal-return guard. Current estimates are
3.39 hours for the next 25 and 3.25 hours for the final 24, before overhead;
actual availability and speed can differ.

`scripts/verify_augmentation_results.py` performs the audit.
`scripts/continue_augmentation_training.py` checks each completed archive before
submitting another private, source-version- and image-pinned job. It persists
submission phases to avoid a silent duplicate submission after interruption.
On a remote error or failed audit it stops future submissions, preserving prior
checkpoints. It never opens test data. Final validation comparison exports only
after all 50 epochs are verified.

The controller runs **locally** and needs WSL/the computer to remain running for
automatic handoffs. An already submitted Kaggle job runs independently. State,
PID information and logs are in `outputs/augmentation/training_pipeline/`:
`state.json`, `controller_pid.json`, `controller.log`. Completed archives/audits
are retained under its `results/`; the final table will be
`validation_comparison.csv`. Training itself is entirely on Kaggle.

## Training commands and historical continuation procedure

Local libraries use the existing isolated uv environment; no dependencies were
added to the project or system Python. The test/preview prefix was:

```bash
MPLCONFIGDIR=/tmp/rd-gradcam-matplotlib uv --cache-dir outputs/.pilot-uv-cache run --offline --no-project --with torch --with torchvision --with pandas --with nbformat --with opencv-python-headless --with ipython --with scikit-learn --with huggingface-hub --with matplotlib --with pillow --index https://download.pytorch.org/whl/cpu python
```

Run `scripts/preview_video_augmentation.py --output outputs/augmentation/new_preview`
with that prefix to create a fresh preview. Existing outputs are never overwritten.
The final rendering reused the retry's eight verified arrays and applied the revised
policy; the raw videos were not changed. Tests used
`-m unittest discover -s tests -p '*augmentation*.py' -v` and
`-m unittest discover -s tests -p test_unetpp_training.py -v` with the same prefix.

Config/notebook/bundle commands run:

```bash
uv --cache-dir /tmp/dmml-uv-cache run python scripts/freeze_augmentation_config.py --preview outputs/augmentation/unetpp_rd_aug_001/final_previews/preview_manifest.json --reference-runtime outputs/kaggle_results/session_03/verified/unetpp_rd_002_amp/runtime.json
uv --cache-dir /tmp/dmml-uv-cache run python scripts/build_augmentation_notebook.py
uv --cache-dir /tmp/dmml-uv-cache run python scripts/prepare_augmentation_job.py --username bijaykhadka1710 --output-root outputs/kaggle_jobs/augmentation_pinned_runtime
uv --cache-dir /tmp/dmml-uv-cache run python scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/augmentation_pinned_runtime/unetpp-rd-aug-001-train-01
```

The submission is private, Internet-disabled, requests one T4 allocation and
attaches only `bijaykhadka1710/unetpp-rd-002-inputs-00/2` and
`bijaykhadka1710/unetpp-rd-002-inputs-01/2`. Both completed source statuses were
verified before submission. It uses only GPU 0. Wait for successful saved output,
verify/download its archive and measured budget before continuing, for example:

```bash
uv --cache-dir /tmp/dmml-uv-cache run python scripts/prepare_augmentation_job.py --username bijaykhadka1710 --session 2 --epochs 25 --previous bijaykhadka1710/unetpp-rd-aug-001-train-01/2
```

That command only prepares a continuation. Based on the measured first epoch,
do **not** request all 49 in one six-hour job; the launched controller uses at
most 25, then the remainder. Retain an adverse or
inconclusive result. A one-seed improvement would not establish broad benefit,
unseen-patient generalization or clinical validity.

## Runtime incident before training

Version 1 ran in the current default image: Python 3.13.16, torch 2.11.0+cu130,
CUDA 13.0, cuDNN 92700, huggingface_hub 1.33.0. The baseline used Python
3.13.15, torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, huggingface_hub 1.29.0.
It stopped at the identity check before cache loading, the probe or training;
no augmentation checkpoint or ERDES metric was produced. The log is retained in
`outputs/kaggle_results/unetpp_aug_001_session_01/`. No guard was weakened.

Read-only `kernels pull .../unetpp-rd-002-train-03/1 -m` recovered the original
Docker image into `outputs/kaggle_runtime_pins/baseline/`. A fresh local bundle
under `outputs/kaggle_jobs/augmentation_pinned_runtime/` preserves the failed
bundle and requests that image for version 2. All 15 augmentation checks were
rerun successfully after pinning, including the new runtime-drift regression.

Continuation commands run after verifying epoch 1:

```bash
uv --cache-dir outputs/.pilot-uv-cache run --offline --no-project --with torch --with torchvision --with pandas --with nbformat --with opencv-python-headless --with ipython --with scikit-learn --with huggingface-hub --with matplotlib --with pillow --index https://download.pytorch.org/whl/cpu python scripts/verify_augmentation_results.py --archive outputs/kaggle_results/unetpp_aug_001_session_01_v2/unetpp_rd_aug_001_amp_results.zip --expected-epochs 1
uv --cache-dir /tmp/dmml-uv-cache run python -u scripts/continue_augmentation_training.py --workspace outputs/augmentation/training_pipeline --initial-verification outputs/kaggle_results/unetpp_aug_001_session_01_v2/verification.json
```

The controller was started in a separate local process with its output appended
to `controller.log`. To resume it after a local restart, use the second command;
a process lock prevents two controllers from running together. An uncertain
submission requires inspecting the saved/remote state before retrying. The
expanded 20-check augmentation/continuation suite passed, as did Python
compilation and `git diff --check`; the complete unrelated model suite was not
rerun for the orchestration-only additions.
