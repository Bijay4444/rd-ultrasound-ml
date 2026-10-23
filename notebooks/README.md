# Notebooks

Start with [setup_and_eda.ipynb](setup_and_eda.ipynb). It downloads the official
ERDES metadata CSV, checks it, shows class counts and clip lengths, and saves a
small summary and plot. It needs no GPU, PyTorch, upstream clone, or videos.

## Run with Colab in VS Code

1. Install Google's [Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
   (`Google.colab`).
2. Open the notebook, then choose **Select Kernel → Colab → Auto Connect** and
   sign in. A CPU runtime is sufficient for this notebook.
3. Run section 1 first. Confirm that imports work and read the printed storage paths.
4. Continue one cell at a time with **Shift+Enter**, reading each result.

See Google's [extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
You can also upload the notebook to [Colab in a browser](https://colab.research.google.com/).

The notebook is edited locally in VS Code, but a Colab kernel runs Python on a
remote machine. Local files and `.venv` are separate from that runtime. This
notebook downloads its input directly, so no project upload or clone is needed.
It uses pandas and matplotlib from the managed Colab runtime; section 1 checks
their imports and records their versions. It does not install our local project.
If an import fails, stop and inspect the selected runtime before adding packages.

## Optional: run on your own computer

Use uv from the project root to declare the analysis dependencies:

```bash
uv add --group analysis pandas matplotlib ipykernel
```

Select the project's `.venv` as the notebook kernel. Review and commit the
resulting `pyproject.toml` and `uv.lock` changes with the notebook. Teammates can
then recreate that environment with `uv sync --locked --group analysis`.
The existing Python requirement is provisional; check compatibility when
resolving dependencies. A local uv environment does not configure Colab.

## Save and review

- Metadata goes under `data/erdes/`; results go under `outputs/metadata_eda/`.
  Both are ignored by this repository. Colab uses `/content` when no project
  checkout is present.
- Colab storage is temporary. Download the summary and plot or copy them to
  Drive before the runtime ends. Saving a notebook does not save its data files.
- Save the notebook after editing. Review outputs and clear sensitive details
  before committing. Move reusable logic into `src/` when later work needs it.

## What comes next

1. Finish this metadata notebook and explain the labels, imbalance, and clip lengths.
2. Audit the official split CSVs: counts, label mapping, clip overlap, and
   available patient/eye grouping. Preserve the supplied splits and document limitations.
3. Obtain videos with recorded provenance and access terms; inspect a few training clips.
4. Agree on the baseline task, preprocessing, metrics, and compute budget.
   Then select ML dependencies and build a small training/validation run.

Reserve final test evaluation until model development is complete.
