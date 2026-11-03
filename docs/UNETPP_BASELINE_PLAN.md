# UNet++ RD baseline and next experiment

Updated 2026-11-02. The user clarified the professor's requirement: reproduce
**one manageable published baseline** in the available Colab environment, then
add a controlled extension. The target is not all 40 models. The progress talk
is after the weekend, reported as three days away; its exact duration/date is
not yet confirmed.

Project naming follows the user's preference: model/task names for files,
configuration IDs and variables (`unetpp_resource_check`, `unetpp_rd_001`,
`UNetPPRDClassifier`). Methods notes retain the source attribution and explain
which protocol details our own implementation follows or adapts.

## Current state — training and official-test evaluation complete

The Kaggle `unetpp_rd_002_amp` run completed all 50 epochs. All 70 saved artifact
hashes and all epochs' recorded metrics were verified. Epoch 48 was selected by
the unchanged validation-accuracy rule: TP=37, TN=389, FP=2, FN=3; sensitivity
0.925, specificity 0.994885 and accuracy 0.988399 on **validation**. The selected
checkpoint and threshold 0.5 were frozen before the evaluation workflow opened
the official test CSV; earlier split-integrity audits had read membership/counts.

CPU preparation and frozen GPU inference on all 1,077 official test clips are
complete. Verified test counts: TP=86, TN=964, FP=13, FN=14; accuracy 0.974930,
sensitivity 0.86, specificity 0.986694. All 12 evaluation artifact hashes and
all prediction rows/metrics passed independent checks. The next milestone is
the progress presentation/report; a controlled extension remains a separate
experiment planned using training/validation evidence.
See [UNet++ results](UNETPP_RESULTS.md),
[the frozen evaluation config](../configs/unetpp_rd_002_evaluation.json) and
[the Kaggle guide](KAGGLE_GUIDE.md). Earlier preflight and Colab instructions
below are retained as historical implementation context, not current pending work.

## What is already complete

The engineering pipeline and two custom-CNN pilots are complete. Both use 256
training clips, 16 sampled frames, 128×128 images, and five epochs. They are
useful pipeline checks, but neither is a paper-benchmark reproduction.

| Validation result | Custom unweighted pilot | Custom weighted pilot |
| --- | ---: | ---: |
| Average precision | 0.090683 | 0.145320 |
| ROC-AUC | 0.449105 | 0.611061 |
| RD sensitivity at 0.5 | 0 | 0 |
| Specificity at 0.5 | 1 | 1 |

Source: completed outputs in `notebooks/baseline_training.ipynb`. These are
single-seed validation findings; no improvement in thresholded RD detection
was demonstrated. Retain both runs rather than rerunning or relabeling them.

## Selected reproduction target

**UNet++ classifier, Non-RD versus RD, Table 8.** This refers to the authors'
five-block encoder-only classifier with global pooling and a two-layer head,
not a full nested segmentation network or our small custom CNN. It has
3,603,553 parameters, verified against the pinned MONAI reference.
See [the primary-source protocol audit](research/erdes-unetpp-protocol.md).

Use all 3,873 supplied training clips and all 431 validation clips. Convert the
complete decoded video to grayscale, pad to square, and trilinearly resize time
and space to `[1,96,128,128]`. Train from scratch for 50 epochs with unweighted
BCEWithLogitsLoss and the reference effective batch of 48. The released code
uses Adam at 1.5e-5, zero weight decay, ReduceLROnPlateau on validation loss,
seed 42, and highest-validation-accuracy checkpoint selection. Keep the original
official split membership and threshold 0.5.

**Disclose the protocol conflict:** the paper describes AdamW, while its pinned
executable configuration uses Adam. This initial attempt follows the released
code and must be named accordingly. Current software versions, OpenCV decoding,
one-GPU execution and gradient accumulation are also recorded adaptations;
they are not silently presented as an identical three-GPU experiment.

Published UNet++ values are *author-reported test metrics*, not targets to tune
against: sensitivity 0.879, specificity 0.986, precision 0.871, F1 0.875, accuracy
0.976. The source audit records an unresolved denominator/aggregation issue with
the reported recall. A successful run does not guarantee those exact numbers.

## Measured T4 resource check — completed

The saved execution of `unetpp_resource_check.ipynb` passed on a Tesla T4
(PyTorch 2.11.0+cu130, Python 3.13.16). Batch 4 used 6,543,866,368 peak allocated
bytes (6.54 GB; peak reserved 7.95 GB). Its median FP32 training step was 1.22137
seconds and validation step 0.32998 seconds. The resulting **50-epoch compute-only
projection is 16.92 hours**, excluding data preparation, cache reads, checkpoint
IO and disconnects. This is hardware evidence, not a training result or a
completion guarantee.

The report is already in Drive under
`MyDrive/rd-ultrasound-ml/outputs/unetpp_resource_check/20261101T193236373065Z/`.
The two diagnostic training clips passed range/shape checks; their disposable
weights were discarded. No test clips were used. Keep this executed notebook.

## Immediate action: full-data training

Open [unetpp_training.ipynb](../notebooks/unetpp_training.ipynb), connect the same
Colab T4 and run from the top. It is standalone and embeds the exact project
modules and [training config](../configs/unetpp_rd_001_training.json). Naming
uses model/task identifiers; source attribution remains in the methods notes.

Because the FP32 estimate is long for a free session, the default run explicitly
uses **FP16 autocast, FP32 master parameters and GradScaler** (`16-mixed`). It
first measures disposable real-input steps and requires stable finite nonzero
gradients before full acquisition. This is a documented numerical adaptation;
its speed and full-epoch wall time have not yet been measured on the user's GPU.
The implementation keeps the scaler fixed across each effective batch and
steps/updates it only at accumulation boundaries, following
[PyTorch's AMP accumulation guidance](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html#gradient-accumulation).
FP32 remains an explicit configuration option for a separate run; changing a
saved run's config or code is rejected.

1. Mount Drive when prompted. Default persistent folder:
   `My Drive/rd-ultrasound-ml/outputs/unetpp_training/unetpp_rd_001_amp/`.
2. Pass the new precision check. Acquire immutable-revision source hashes for all
   4,304 train/validation videos, then verify/decode/cache each input. The test CSV
   stays unopened. Cached partial progress is reusable after an HTTP error by
   rerunning the acquisition cell in the same runtime.
3. Run section 6 with its default `UNETPP_EPOCHS_THIS_SESSION = 1`. This is a
   full-data epoch timing check, while the fixed experiment target remains 50.
4. Read epoch time, losses, confusion counts, AP and skipped AMP updates. Set
   `UNETPP_EPOCHS_THIS_SESSION = None` and rerun **section 6 only** to continue
   toward epoch 50 from the saved checkpoint. The summary stays `partial` until
   all 50 epochs finish. No automatic early stopping or test tuning is added.
5. After a Colab reset, run this notebook from the top with the **same run name**.
   Raw videos and the 27.08 GB prepared cache are rebuilt on runtime disk; they
   are deliberately not put in free Drive storage. Checkpoints and manifests
   remain in Drive. Exact config, code, runtime-version/GPU and input-fingerprint
   checks reject an incompatible resume.
6. After 50 epochs, freeze the configuration and selected validation-accuracy
   checkpoint. Plan the separate official test evaluation next; compare **test**
   metrics with the published test row, with the documented source discrepancies.

The effective batch is 48: physical batch 4 and 12 accumulation steps. Each
seeded epoch permutation includes every training clip exactly once. The final
33-example group is normalized by 33 rather than 48. No class weighting,
resampling or augmentation is added. Accumulation and AMP are not claimed to
produce bit-identical three-GPU FP32 training.

Recovery checkpoints save every 10 effective updates, at training end, and after
validation. They include model, Adam, scheduler, scaler, RNG state, epoch order,
completed-example cursor, loss totals, history and best weights. Two alternating
files plus an atomic checksum pointer retain the preceding valid generation.
A disconnected run repeats only the work after its last complete checkpoint.
As with any remote filesystem, allow Drive to finish syncing. Run only one
kernel against a run folder. `best.pt` and best prediction exports are rebuilt
from the authoritative checkpoint/prediction records after interrupted exports.

The export cell writes a ZIP next to the run folder in Drive, including source
snapshots, config, manifests, predictions, metrics and checkpoints; it excludes
raw clips and prepared arrays. Saving or mounting the notebook alone does not
move outputs from previous custom-CNN runs.

## Three-day priorities

| Time | Priority and evidence |
| --- | --- |
| Today | T4 FP32 preflight is complete. Run the mixed-precision check, prepare full inputs, and time the first full epoch. |
| Next day | Continue the resumable fixed 50-epoch run using measured settings; monitor actual epoch time and preserve checkpoints. |
| Before the talk | Finish and audit whatever the logs substantiate; prepare data/protocol/results/limitations figures and rehearse. State incomplete work plainly. |

The goal is a completed single baseline before the talk, but neither free Colab
availability nor 50-epoch completion has been established. If the measured budget
does not fit, report that evidence and remaining work rather than renaming a
smaller/shorter experiment as a completed reproduction.

After the reference is established, define one extension on that same model/data
protocol, such as training-only class weighting with all other settings fixed.
That is a controlled extension, not automatically a novel method. The earlier
custom-CNN weighting result does not substitute for this matched comparison.

## Verification and remaining work

The independent PyTorch architecture was checked against the read-only authors'
MONAI 1.5.2 model with CPU PyTorch 2.5.1: initial seeded tensors matched exactly;
synthetic forward logits and backward gradients agreed. Seven new behavioral
checks cover model size, gradients, temporal interpolation, padding/range,
decoder/metadata failures, protocol agreement, train-only preflight selection,
and rejection of a CPU-only session as a GPU benchmark. Batch/accumulated-gradient
algebra is checked in float64; an initial FP32 comparison showed numerical
differences and is not used to promise bitwise equivalence.

The two actual cached training clips passed source SHA256 and preprocessing
comparison with the direct reference tensor pipeline using the same decoded
frames. This does not establish cross-library decoder identity. Notebook schema,
syntax and ZIP export were checked on CPU. No author checkpoints, full dataset,
GPU training, or final test evaluation were run for this work.

Evidence is ignored under `outputs/unetpp_verification/`. Commands:
temporary uv execution of `/tmp/verify_unetpp_resource_inputs.py`; the read-only
upstream interpreter with `PYTHONDONTWRITEBYTECODE=1` for
`/tmp/verify_unetpp_parity.py` (the reference dependency stack is already installed
there); `python -m unittest discover -s tests -v` in the existing temporary uv
CPU environment; notebook validation and `git diff --check`. All 39 unit checks
passed. Upstream tracked source is unchanged; its untracked `uv.lock` predates
this work (2026-10-06) and was left untouched. The primary-source audit used a
research sub-agent. Both intended protocol and embedded module agree with their
tracked files; no dependency metadata changed. The saved user execution now verifies FP32 CUDA fit and timing. The new AMP
probe, full-data training and final test evaluation remain unrun locally.

Course evidence: the available syllabus states weekly incremental progress
presentations and a finalized report in December. The professor's more specific
single-baseline reproduction requirement is supplied by the user and governs
this plan. The repository still contains `docs/course-guideline/`, while
`AGENTS.md` names `docs/course-guidance/`; no dedicated marking rubric was found.


## Full-run implementation checks (2026-11-01)

`unetpp_training.py` contains accumulation, validation, metric computation and
checkpoint recovery. `unetpp_data.py` contains official-partition validation,
incremental source-catalog persistence and atomic per-video tensor caching.
The standalone training notebook embeds these modules and `unetpp_baseline.py`.

Behavioral CPU checks compare accumulated updates with true effective batches,
including a short final group; inject a disconnect after a saved optimizer
boundary; and compare resumed model parameters **exactly** with an uninterrupted
synthetic Dropout run. They also exercise safe RNG reload, checkpoint-corruption
fallback followed by an interrupted write, cache corruption rejection, source-API
chunk recovery, manifest/label guards, and notebook/module/config agreement.
Synthetic equality is not a claim of deterministic CUDA behavior. The standalone
embedded imports, saved config and actual official CSV joins are exercised with
two locally cached real training clips. At this November 1 implementation check,
full acquisition, mixed-precision CUDA execution, all 50 epochs and official-test
inference had not been run. They subsequently completed on Kaggle; see the
current state above and [verified results](UNETPP_RESULTS.md). Local synthetic
evidence remains in ignored `outputs/unetpp_verification/`.


Verification commands used a temporary uv environment (no dependency metadata
changes):

```bash
uv --cache-dir outputs/.pilot-uv-cache run --offline --no-project --with torch --with pandas --with nbformat --with opencv-python-headless --with ipython --with scikit-learn --with huggingface-hub --index https://download.pytorch.org/whl/cpu python -m unittest discover -s tests -v
# Same uv environment:
python /tmp/verify_unetpp_training_notebook.py
python /tmp/verify_unetpp_training_export.py
git diff --check
```

All **48 unit tests passed**. Notebook schema/embedded execution, official CSV
joins and two real diagnostic inputs passed. The actual training/export cells
were also exercised with synthetic CPU inputs, checking partial-run ZIP hashes,
resume files and continuation to a fixed target; those are implementation checks,
not ERDES model-performance results. The new GPU probe and full research training
still require the user's Colab session. Existing executed resource-check outputs
and baseline notebook were preserved.


## Kaggle execution alternative (2026-11-01)

The user reached `READY: 3873 train, 431 validation` in Colab, then lost GPU
access before the first full-data training epoch. Those constant non-RD metrics
are reference calculations, not trained-model results. The runtime cache was
temporary; no full-data model checkpoint exists if section 6 never ran.

The Kaggle route uses `configs/unetpp_rd_002_training.json`, two CPU preparation
outputs (2,152 clips / about 13.54 GB each), and a separate GPU notebook that
attaches both. All float32 array values, official partitions and target epochs
remain; CPU preparation is no longer charged against an allocated GPU session.
This split fits Kaggle's documented 20 GB saved-output limit per notebook.
No arrays are quantized or downloaded to the laptop by the setup workflow.

Continuation explicitly imports a preceding successfully saved GPU output into
a new job. In-runtime checkpoints alone are not guaranteed to survive Kaggle
termination. Bounded jobs finish/export normally, then hand off through saved
outputs; a failed unsaved job may need repeating from its last saved predecessor.
Only CUDA 0 is used, even on T4 x2. No authenticated Kaggle execution has been
performed locally. See [the setup and run guide](KAGGLE_GUIDE.md).
