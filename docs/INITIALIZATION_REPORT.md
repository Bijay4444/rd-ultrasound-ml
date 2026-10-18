# Initialization evidence — 2026-10-18

Created: README.md, .gitignore, pyproject.toml, uv.lock, src package initializer,
data placeholder, notebook/test guidance, research/learning/data-access notes,
scope, weekly plan, literature/paper/experiment templates, decision log and four
reusable task prompts with their README. Source research is in research/setup-sources.md.
Existing AGENTS.md, initialization prompt and course PDF were preserved.
No upstream files were modified. No commit or push was made.

## Commands and checks

- Inspected repository with `rg --files`, `ls`, `cat`, `git status --short`;
  inspected selected upstream code/configs and recorded its Git revision.
- `pdftotext` was unavailable. Temporary `uv run --no-project --with pymupdf`
  extracted all course PDF pages after network access approval. It added no project dependency.
- Used `uv --cache-dir /tmp/rd-uv-cache` because the default cache was read-only.
- `uv lock --offline --python 3.14`: passed; no third-party dependencies.
- `uv sync --locked --offline --python 3.14`: passed; isolated environment created.
- `uv run --locked --offline python ...`: Python 3.14.4 and source import passed.
- Standard-library checks: TOML parsing, required files, local Markdown links,
  Git ignore rules (data/secrets/checkpoints/logs/environment), and new-file sizes passed.
- `git diff --check`: passed. All initial content is untracked in this repository
  with no commits, so ordinary `git diff` has no content to compare; generated
  content and `git status` were inspected directly. Nothing was staged.

## Limits and next action

No model, dataset, GPU, video decoding or upstream test checks were run.
No final ML stack or interpreter pin was selected; 3.14 is the initialization
floor/candidate only. No paper results were populated without full-text review.
Course guidance directory spelling differs from AGENTS.md; supplied PDF was read.
Rubric, syllabus, deadlines, compute budget and final research question remain open.
Next: read the ERDES paper together, obtain the grading rubric, and audit metadata
and patient/eye grouping before proposing the first baseline.
