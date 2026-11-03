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
