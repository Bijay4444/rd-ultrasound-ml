# Project Instructions

## Project

This repository contains a CSCI 555 Data Mining and Machine Learning
research project at St. Francis Xavier University.

Project title:

Retinal Detachment Analysis from Ocular Ultrasound Using Deep Learning

Primary dataset:

ERDES — Eye Retinal DEtachment ultraSound

The project is currently in the research and setup stage.

Do not assume the final research question, model architecture,
Python version, or experimental design has already been decided.

---

## Research Workflow

For substantial work:

1. Inspect relevant files and existing work first.
2. Understand the problem before implementing.
3. Propose a short plan before major changes.
4. Make small, checkable changes.
5. Run appropriate checks after changes.
6. Review the resulting diff.
7. Report:
   - files changed,
   - commands run,
   - checks passed,
   - checks not run,
   - unresolved issues.

Do not claim that something works without evidence.

Do not weaken or remove tests simply to make them pass.

---

## Development Environment

Use `uv` for Python environment and dependency management unless
there is a documented technical reason not to.

Do not modify the system Python installation.

Never use:

`--break-system-packages`

Do not pin Python to an older version solely because the upstream
ERDES implementation used it.

Prefer the newest stable Python version that is mutually compatible
with the required ML libraries.

Do not add dependencies unnecessarily.

---

## Upstream ERDES

The original ERDES implementation is external reference material.

If a local upstream ERDES repository is available:

- inspect it when useful,
- treat it as read-only,
- do not modify it,
- do not blindly copy large portions of its code,
- document anything reused or adapted,
- preserve attribution.

Our implementation and experiments must remain clearly distinguishable
from the upstream project.

---

## Data and Large Files

Never commit:

- raw ERDES ultrasound videos,
- downloaded datasets,
- model checkpoints,
- large experiment outputs,
- logs,
- secrets or credentials.

Keep raw data under ignored directories such as `data/`.

The repository should contain reproducible instructions for obtaining
external data instead.

---

## Machine Learning Rules

Preserve experimental reproducibility.

When experimentation begins:

- use documented train/validation/test splits,
- avoid data leakage,
- fit learned preprocessing only using appropriate training data,
- record random seeds,
- record model and training configurations,
- record metrics and experiment results,
- keep final test evaluation separate from model development.

Do not change dataset splits without documenting why.

---

## Medical Research Boundaries

This is an educational/research machine-learning project.

Do not describe experimental models as clinically validated diagnostic
systems unless evidence explicitly supports that conclusion.

Distinguish clearly between:

- ground-truth dataset labels,
- model predictions,
- experimental findings,
- clinical claims.

Personal experience with retinal detachment motivates the project but
is not scientific evidence.

---

## Course Guidance

Before substantial project work, check relevant materials under:

`docs/course-guidance/`

Follow applicable professor/TA guidance.

If instructions conflict or are unclear, report the conflict rather
than silently choosing one.