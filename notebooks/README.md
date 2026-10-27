# Notebooks

Use one notebook for each focused stage:

| Order | Notebook | Purpose | Results folder |
| --- | --- | --- | --- |
| 1 | [setup_and_eda.ipynb](setup_and_eda.ipynb) | Metadata checks, labels, class imbalance, and clip lengths | `outputs/metadata_eda/` |
| 2 | [split_audit.ipynb](split_audit.ipynb) | Supplied non-RD/RD partitions, label agreement, coverage, clip overlap, and grouping limitations | `outputs/split_audit/` |
| 3 | [video_inspection.ipynb](video_inspection.ipynb) | Eight training clips: source hashes, decoding, frame previews, playback, and visual notes | `outputs/video_inspection/` |
| 4 | [baseline_preparation.ipynb](baseline_preparation.ipynb) | Verify fixed-size inputs and save the proposed first GPU-pilot configuration | `outputs/baseline_preparation/` |

Each notebook runs independently from a fresh kernel and downloads its own small
inputs when needed. All use a CPU runtime, with no GPU, PyTorch, or upstream
clone required. The first two need no videos; the third downloads about 4.3 MB
of selected training clips. Keep later modeling work in a separate notebook.

## Run with Colab in VS Code

1. Install Google's [Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
   (`Google.colab`).
2. Open the notebook, then choose **Select Kernel → Colab → Auto Connect** and
   sign in. A CPU runtime is sufficient for these four notebooks. The planned
   training pilot will use a free Colab GPU in a separate notebook.
3. Run section 1 first. Confirm that imports work and read the printed storage paths.
4. Continue one cell at a time with **Shift+Enter**, reading each result.

See Google's [extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
You can also upload the notebook to [Colab in a browser](https://colab.research.google.com/).

The notebook is edited locally in VS Code, but a Colab kernel runs Python on a
remote machine. Local files and `.venv` are separate from that runtime. This
notebook downloads its inputs directly, so no project upload or clone is needed.
The metadata notebook uses pandas and matplotlib; the split audit needs only
pandas. Video inspection adds OpenCV for decoding and notebook video display.
Each checks imports and records versions in section 1, using the managed Colab
runtime. None installs our local project.
If an import fails, stop and inspect the selected runtime before adding packages.

## Optional: run on your own computer

Use uv from the project root to declare the analysis dependencies:

```bash
uv add --group analysis pandas matplotlib ipykernel
```

For local video inspection, also declare the decoder dependency:

```bash
uv add --group analysis opencv-python-headless
```

Colab normally already provides `cv2`; do not install a second OpenCV distribution
there just to match the local package name. Check the actual selected kernel.

Select the project's `.venv` as the notebook kernel. Review and commit the
resulting `pyproject.toml` and `uv.lock` changes with the notebook. Teammates can
then recreate that environment with `uv sync --locked --group analysis`.
The existing Python requirement is provisional; check compatibility when
resolving dependencies. A local uv environment does not configure Colab.

## Save and review

- Metadata and the split archive go under `data/erdes/`; unchanged CSV copies go
  under `data/erdes/splits/non_rd_vs_rd/`. Result folders are listed in the table
  above. Inputs and outputs are ignored by Git. Colab uses `/content` when no
  project checkout is present.
- The selected videos go under `data/erdes/clips/`, preserving the metadata paths.
- Colab storage is temporary. Preserve `metadata_summary.json` and
  `metadata_overview.png` from the first notebook, and `split_summary.csv` and
  `split_audit.json` from the second, by downloading them or copying them to Drive.
  Saving a notebook does not save its data files.
- From video inspection, preserve `inspection_summary.json`, `decode_report.csv`,
  `training_frame_examples.png`, and your completed `visual_review_notes.csv`.
- Agent inspection notes and measurements are recorded in local
  `outputs/video_inspection/visual_review_notes.csv` and `observations_summary.json`.
  The user's executed notebook was archived in that ignored folder before clearing
  embedded video/image outputs. The tracked notebook includes a concise review summary.
- From baseline preparation, preserve `baseline_001.json`, `preparation_summary.json`,
  and `preprocessed_examples.png`. The notebook embeds the tracked config so Colab
  does not need a project upload. It does not train a model or evaluate validation/test data.
- When a kernel restarts, run the selected notebook from its first code cell.
  No execution state is shared between notebooks; cached inputs are checked and reused.
- Save the notebook after editing. Review outputs and clear sensitive details
  before committing. Move reusable logic into `src/` when later work needs it.
- Clear **all outputs** from `video_inspection.ipynb` before committing it:
  playback embeds video bytes and the frame overview contains dataset images.
  Retain those artifacts under ignored output folders instead.

## What comes next

1. Run `baseline_preparation.ipynb` and inspect the transformed inputs.
2. Review the [proposed baseline plan](../docs/BASELINE_PLAN.md), including the
   documented possible patient overlap and distinction between inspection clips
   and the future pilot training subset.
3. Implement the separate GPU pilot: forward/backward checks, tiny-batch overfit
   check, then the recorded train/validation experiment. Training dependencies
   and model code are not implemented yet.

Reserve final test evaluation until model development is complete.
