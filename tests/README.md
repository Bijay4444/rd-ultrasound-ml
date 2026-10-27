# Tests

`test_video_preview.py` exercises the video-inspection notebook's playback loop
with synthetic local files. It checks embedded HTML and replays the Colab
constructor path check that rejected `Video(filename=...)` when `data` was unset.
No dataset, decoder, GPU, or network is needed for these tests.

`test_baseline_preparation.py` checks the preparation notebook's actual helper
functions using synthetic frames and a three-frame MJPG video. It covers frame
repetition/order, square padding, channel averaging/scaling, output layout,
metadata mismatch, and missing files. These tests additionally need numpy and OpenCV.

Run with a temporary uv environment:

```bash
uv run --with ipython --with numpy --with opencv-python-headless python -m unittest discover -s tests -v
```

If these libraries are already declared in the selected project environment, use
`uv run python -m unittest discover -s tests -v`. Keep future data/pipeline
tests independent of private videos where possible.
