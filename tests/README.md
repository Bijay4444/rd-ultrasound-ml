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

Run with a temporary uv environment:

```bash
uv run --no-project --with ipython --with numpy --with opencv-python-headless --with pandas --with scikit-learn --with torch --index https://download.pytorch.org/whl/cpu python -m unittest discover -s tests -v
```

If these libraries are already declared in the selected project environment, use
`uv run python -m unittest discover -s tests -v`. Keep future data/pipeline
tests independent of private videos where possible.
