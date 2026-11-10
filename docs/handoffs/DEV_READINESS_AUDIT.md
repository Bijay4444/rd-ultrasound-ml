# Dev branch handoff audit

Checked 2026-11-10 against GitHub `dev` commit
`911a4409f69bfb639ffb1dcde605de0877619e05`, fetched read-only for inspection.
The local checkout matched that snapshot before the branch-start clarifications
and this audit were added. No merge, training, commit or push was performed by
the audit.

**Result: the branch contains the context and reusable infrastructure needed for
teammate 1 to implement R(2+1)D-18. A ready-to-run R(2+1)D training workflow is
the assigned deliverable, not an existing artifact.**

## Repository coverage

| Requirement | Verified in the inspected remote snapshot |
| --- | --- |
| Project instructions and course material | `AGENTS.md`, existing course PDF under `docs/course-guideline/` |
| Assignment and prompt | `docs/handoffs/R2PLUS1D18.md`, with teammate 1 assignment, WhatsApp message, Claude prompt and references |
| Comparison rules and known pitfalls | `docs/MODEL_COMPARISON_PLAN.md` and `docs/research/video-model-comparison-sources.md` |
| Completed baseline and frozen identity | `docs/UNETPP_RESULTS.md`, training/runtime/evaluation configs |
| Reusable model/data/training code | All seven existing `src/rd_ultrasound_ml/` Python files |
| Preparation, resource and training examples | Existing U-Net++ CPU-preparation, resource-check and GPU-training notebooks |
| Remote submission/recovery infrastructure | Kaggle CLI wrapper, setup helper, notebook builder and job-preparation scripts |
| Data acquisition and split rules | `docs/DATA_ACCESS.md`, `docs/SPLIT_AUDIT.md`, pinned dataset revision and checksum checks |
| Existing checks and environment metadata | `tests/`, `tests/README.md`, `pyproject.toml`, `uv.lock`, `.gitignore` |

The fresh-checkout instructions now explicitly select `dev`. Starting from the
older default `main` would omit the required recent work. The source/tests do not
require Bijay's absolute workspace path or the optional sibling upstream checkout.

## Items outside Git or still to be implemented

- **Model-specific work:** `r2plus1d18_baseline.py`, its resolved training config,
  standalone training notebook/submission workflow, cache-identity adapter,
  focused checks and results report are not implemented yet. The handoff tells
  the teammate to create them. The existing U-Net++ submission command does not
  launch the new model.
- **ML environment:** `pyproject.toml` declares no ML dependencies. `uv sync`
  alone does not install torch, torchvision or the other training libraries.
  Use the documented managed GPU runtime or prepare a compatible isolated uv
  environment; record exact versions. New model runtime compatibility is untested.
- **Prepared data:** recorded private preparation slugs are
  `bijaykhadka1710/unetpp-rd-002-inputs-00` and
  `bijaykhadka1710/unetpp-rd-002-inputs-01`. Their exact saved versions and the
  teammate's account access remain to be verified before reuse. Otherwise prepare
  the same release independently using the existing acquisition workflow.
  Original preparation `config.json`, manifests and provenance travel with the
  attached outputs; retain them when adapting the cache loader.
- **GPU/account access:** quota, GPU availability, real-input memory, AMP stability
  and full-epoch timing are not established for either new model.
- **Large artifacts:** raw videos, cached arrays, checkpoints and detailed
  experiment outputs are intentionally ignored. Baseline weights are unnecessary
  for the teammate's from-scratch model; baseline prediction archives may be
  shared privately later when integrating comparison plots.

## Checks actually run

- `git ls-remote --heads origin dev`, then `git fetch origin dev`: verified the
  remote revision without merging or changing tracked working files.
- Temporary stdlib audit via `uv --cache-dir /tmp/dmml-uv-cache run --locked
  --offline python /tmp/audit_dev_handoff.py`: all **36 required files** were
  tracked; all **36 local handoff/navigation links** targeted tracked files.
- Parsed **23 Python files**, **8 JSON configs**, **12 notebook JSON structures**
  and **126 code cells** from the remote snapshot. No cell needed an IPython
  syntax exclusion. Parsing is not execution or full notebook schema validation.
- All **four reference training source SHA256 hashes** matched the tracked frozen
  evaluation config.
- `uv --cache-dir /tmp/dmml-uv-cache run --locked --offline python -m unittest
  discover -s tests -p test_training_downloads.py -v`: **6 tests passed**.
- Documentation links/formatting and `git diff --check` were checked after the
  handoff clarifications; the final response records their result.

Not run: the full ML test suite, notebook execution, private Kaggle access checks,
data download, model-specific GPU probes or training. No new performance claim
follows from this readiness audit. The pre-existing course directory naming
mismatch remains documented; no grading rubric was found.
