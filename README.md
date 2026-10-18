# Retinal Detachment Analysis from Ocular Ultrasound Using Deep Learning

CSCI 555 — Data Mining and Machine Learning, St. Francis Xavier University.
Three-person group project. **Status: Week 1, research and setup.**

We are learning to investigate retinal detachment using the ERDES (Eye Retinal
DEtachment ultraSound) dataset. The final research question, architecture, and
experimental design are still being refined. No experiments or results exist.
This is educational research; no clinical validation is claimed.

## Start here

1. Read [scope](docs/PROJECT_SCOPE.md) and [learning guide](docs/GETTING_STARTED.md).
2. Follow the [weekly plan](docs/WEEKLY_PLAN.md); align dates with the actual course schedule.
3. Review the paper and [upstream inspection](docs/UPSTREAM_ERDES_REVIEW.md).
4. Obtain the grading rubric and confirm the proposed contribution with the professor.

## Environment

Use uv; do not modify system Python. No ML dependencies are installed yet.
The metadata floor is Python 3.14, supported by the current PyTorch wheel evidence;
the final interpreter and ML dependency set remain open. See
[setup sources](docs/research/setup-sources.md). No `.python-version` is pinned yet.

```bash
uv sync --locked --python 3.14
uv run --locked python -c "import sys; print(sys.version)"
```

Commit `uv.lock` when dependencies change. `uv sync` creates `.venv`; activation
is optional when using `uv run`. The project is currently a non-packaged research
workspace (`tool.uv.package = false`). Source imports can be checked with:

```bash
uv run --locked python -c "import sys; sys.path.insert(0, 'src'); import rd_ultrasound_ml"
```

## Organization

- `src/rd_ultrasound_ml/`: future reusable data/model/evaluation code.
- `notebooks/`: exploration; move reusable logic into `src/` later.
- `tests/`: future behavioral tests when implemented code needs them.
- `data/`: ignored local data; [acquisition checklist](docs/DATA_ACCESS.md).
- `docs/`: scope, paper notes, decisions, learning and experiment records.
- `prompts/`: reusable task prompts; `AGENTS.md`: recurring project rules.

## Team workflow

Use `main` and `feature/<short-description>`. Create a feature branch, make focused
changes, run relevant checks, review `git diff`, commit clearly, push, and open a
pull request. Another teammate should review before merging where practical.
There are no commits yet at initialization; no commit or push was made by this setup.
Never commit raw videos, checkpoints, credentials, or large outputs.

## Experiments and results

Pending dataset audit and baseline design. Record runs using
[the experiment template](docs/EXPERIMENT_LOG.md). Keep the final test set out of
model selection, threshold tuning, and calibration fitting.

## Citation and attribution

Dataset and reference implementation: [ERDES](https://github.com/OSUPCVLab/ERDES).
Paper: [A Benchmark Video Dataset for Retinal Detachment and Macular Status
Classification in Ocular Ultrasound](https://arxiv.org/abs/2508.04735).
Full bibliographic details and dataset version/license must be verified before use.
Our implementation is separate; no upstream code has been copied.
