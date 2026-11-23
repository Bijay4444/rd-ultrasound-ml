# Tests

`test_video_preview.py` exercises the video-inspection notebook's playback loop
with synthetic local files. It checks embedded HTML and replays the Colab
constructor path check that rejected `Video(filename=...)` when `data` was unset.
No dataset, decoder, GPU, or network is needed for these tests.

`test_baseline_preparation.py` checks the preparation notebook's actual helper
functions using synthetic frames and a three-frame MJPG video. It covers frame
repetition/order, square padding, channel averaging/scaling, output layout,
metadata mismatch, and missing files. These tests additionally need numpy and OpenCV.
They also reproduce float32 area-resize overshoot on saturated frames and check
that only tiny numerical drift is corrected while nonfinite/material errors fail.

`test_baseline_training.py` checks deterministic training-only subset selection,
label/path validation, reserved test data, actual model shape and gradients,
classification metrics (including undefined precision), checkpoint ties, validation
ordering/unchanged weights, cached corruption rejection, and agreement with the
preparation functions and tracked training configuration. These checks need CPU
PyTorch, pandas, and scikit-learn, without videos, network access, or a GPU.
The step 7 regression injects a preprocessing failure, then reruns in the same
folder and checks that complete arrays are rebuilt in the unchanged manifest order.

`test_training_downloads.py` replays HTTP 503 after 100 cached clips, recovery,
and a rerun with no further network calls. It also checks retry exhaustion,
Retry-After, permanent HTTP errors, interrupted reads, connection errors, and
checksum failures. It uses only the standard library and mocked HTTP/waits:

```bash
python3 -m unittest discover -s tests -p test_training_downloads.py -v
```

`test_weighted_followup.py` checks the training-only weight calculation, unchanged
control settings, stronger positive loss/gradient, independent validation loss,
prediction/manifest audit failures, and a synthetic end-to-end follow-up run.
It executes the actual audit/training/export cells on CPU, reloads the saved
checkpoint, verifies prediction agreement, and checks that original baseline
artifacts are byte-for-byte unchanged. Synthetic metrics are not ERDES results.

`test_unetpp_baseline.py` checks the 3,603,553-parameter paper-reference model,
full-time-axis trilinear preprocessing, padding/range/metadata guards, standalone
notebook/module/config agreement, train-only preflight selection, and rejection
of CPU-only execution as a CUDA benchmark. Instance-normalized gradient
accumulation is checked algebraically in float64, not as a promise of bitwise
FP32/GPU equality. The separate read-only reference parity check is documented
in `docs/UNETPP_BASELINE_PLAN.md`; it uses the already installed upstream
MONAI dependency stack without changing it. No MONAI dependency is required by
the independent model or these unit tests.

Run with a temporary uv environment:

```bash
uv run --no-project --with ipython --with numpy --with opencv-python-headless --with pandas --with scikit-learn --with torch --index https://download.pytorch.org/whl/cpu python -m unittest discover -s tests -v
```

If these libraries are already declared in the selected project environment, use
`uv run python -m unittest discover -s tests -v`. Keep future data/pipeline
tests independent of private videos where possible.


`test_unetpp_training.py` checks sample-weighted accumulation including a short
final group, interrupted-epoch/session recovery against uninterrupted stochastic
CPU training, safe RNG reload, rotating-checkpoint corruption fallback and
interrupted writes, cache/source integrity, partial source-catalog persistence,
official partition guards, metric edge cases and standalone notebook agreement.
It uses synthetic data and mocked API calls; no CUDA or network is needed.
Real cached input checks and notebook validation are documented in the UNet++
plan. These tests do not establish GPU AMP stability or full-run performance.


`test_unetpp_kaggle.py` checks complete/disjoint CPU shards within the saved-output
budget, private submission metadata and dependencies, credential parsing as data,
attached tensor provenance/order/hash failures, checkpoint copying from read-only
inputs, and notebook/config/source agreement. No Kaggle token/network/GPU is
required. A separate synthetic integration exercises actual CPU preparation and
training/export cells and the saved-output handoff. CLI request construction is
checked against official Kaggle CLI 2.2.5 with its network client mocked.


The Kaggle CLI guard regression exercises the actual wrapper entry point: a
failed CPU input must prevent the GPU submission call, while a continuation must
check both CPU jobs and its preceding GPU job before submission. The two checks
failed before the guard and passed afterward. Remote status/output calls are
mocked; no GPU is allocated by the tests.


`test_unetpp_evaluation.py` checks pre-test checkpoint freezing, rejection of
changed weights/selection before test access, official test counts/label/order/
clip-disjointness, actual fixed-weight CPU inference with no optimizer or model
updates, duplicate result rejection, prepared test tensor corruption, exact
notebook/config/module embedding and private version-pinned submission bundles.
All 62 tests pass. Synthetic inference/export verification is implementation
evidence; it is not a real ERDES test result.

`test_video_augmentation.py` and `test_augmentation_delivery.py` check training-only
video augmentation: pixel-space affine oracles, temporal consistency, keyed RNG
replay without changing global training RNG, input/label immutability, exact
interrupted CPU training and optimizer resume, matched config, frozen sources,
standalone notebook schema/embedding, private pinned bundles, and the actual
training/validation/export notebook cells on synthetic CPU data. Together with
`test_augmentation_continuation.py`, the 20 checks also reject corrupt archives
before extraction and verify that failed jobs, failed audits or uncertain
submissions cannot launch a later training job. The continuation test uses fake
Kaggle responses and checks the 25/24-epoch handoff without allocating a GPU.

`test_augmentation_analysis.py` adds four checks for aligned paired validation
errors, invalid scores/labels, evaluation notebook freshness/schema and a
confirmation gate that precedes test access, plus private version-pinned
submission metadata. The augmentation suite totals 24 checks. The seven shared
checks in `test_r2plus1d18_evaluation.py` also cover the model-agnostic evaluation
helper reused by the augmented U-Net++ notebook.
These checks
need the existing CPU torch, pandas, scikit-learn, OpenCV, nbformat, matplotlib
and Pillow stack. They never use ERDES test data or allocate a GPU. Run with
`python -m unittest discover -s tests -p '*augmentation*.py' -v` in the isolated
uv environment documented in `docs/UNETPP_AUGMENTATION_RESULTS.md`.


`test_r2plus1d18.py` checks the R(2+1)D-18 comparison model: only the first stem
convolution becomes one-channel (temporal stem and factorized blocks preserved),
builder-consistent initialization, no weight download, `[B]` logits with finite
nonzero gradients, the full `[1,1,96,128,128]` shape and the `backbone.layer4`
Grad-CAM feature shape. It also checks the shared cache adapter
(`video_cache.py`): verified U-Net++ shards load under the new training identity,
while edited preparation configs, wrong digest/fingerprint pins, changed
preprocessing, corrupted tensors and changed labels are rejected. Further checks:
session resume equal to uninterrupted CPU training, resume refusal on changed
identity, selected-checkpoint reload reproducing validation probabilities,
notebook freshness/embedding, and version-pinned private bundles. Needs
torchvision in addition to the libraries above (`--with torchvision`). Synthetic
CPU data only; not an ERDES result.

`test_r2plus1d18_evaluation.py` checks the gated R(2+1)D-18 frozen test evaluation on
synthetic CPU data: the team-freeze gate blocks before any test access, training and
reused test-cache verification reject changed pins, identities, split hashes and tensors,
inference uses frozen weights and threshold with timing and no optimizer, the tracked freeze
matches the training sources and the U-Net++ test-input freeze, and the notebook matches its
builder. It never opens real test data.
