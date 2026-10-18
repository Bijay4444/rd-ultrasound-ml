We are initializing a new research-oriented machine learning project.

Before changing anything, inspect the current repository and any course/project guidance files that are already present.

IMPORTANT:
- If there is a professor/TA setup report, course project guide, coding-agent guide, README, PDF-derived notes, or any other project instruction file available in this repository or nearby course materials, read it first and incorporate its relevant setup recommendations.
- Do not silently ignore course-specific guidance.
- If a TA/professor setup document is referenced but not actually available, note that clearly rather than inventing its contents.

==================================================
PROJECT CONTEXT
==================================================

Course:
CSCI 555 — Data Mining and Machine Learning
St. Francis Xavier University

Project theme:
Retinal Detachment Analysis from Ocular Ultrasound Using Deep Learning

Dataset:
ERDES — Eye Retinal DEtachment ultraSound

This is a three-person academic group project.

The project has been approved on the basis of using the ERDES dataset.

The project is currently at WEEK 1:
project initialization, literature review, dataset/repository understanding, and research-question refinement.

We are NOT ready to begin full model implementation or training yet.

The eventual project will likely investigate retinal detachment classification from ocular ultrasound video and may explore model reliability, confidence/calibration, false negatives, class imbalance, or related research questions.

However, DO NOT assume that the final research question has already been fixed.

The research question must emerge from:
1. understanding the ERDES paper,
2. understanding the dataset,
3. inspecting the original ERDES implementation,
4. reviewing relevant literature,
5. identifying a meaningful gap that is achievable within one semester.

==================================================
IMPORTANT DEVELOPMENT PRINCIPLES
==================================================

Use the coding-agent workflow recommended in the course material:

1. Explore before implementing.
2. Plan substantial changes before making them.
3. Make small, checkable changes.
4. Require evidence that work succeeded.
5. Do not claim success without running an appropriate check.
6. Review diffs after meaningful changes.
7. Record what was changed, what was tested, and what was not tested.
8. Do not weaken or remove tests merely to make code pass.
9. Keep recurring project instructions in AGENTS.md.
10. Keep AGENTS.md concise and focused on rules that apply repeatedly.
11. Preserve reproducibility.
12. Avoid data leakage.
13. Keep preprocessing learned from training data separate from validation/test data where relevant.
14. Record random seeds and experimental settings once experimentation begins.

Do NOT use --break-system-packages.

Do NOT modify system Python.

Use uv for Python/dependency/project management unless there is a strong technical reason not to.

Do NOT pin the project to Python 3.10 merely because an older upstream implementation used it.

Before choosing a Python version:
- inspect current dependency requirements,
- inspect the latest stable supported versions of the required ML stack,
- prefer the newest stable mutually compatible Python version.

Do not install large ML dependencies yet unless they are necessary for repository initialization.

==================================================
UPSTREAM ERDES PROJECT
==================================================

The original ERDES authors' repository may exist separately from this project.

If an upstream ERDES repository is available locally:
- inspect it as read-only reference material,
- do not modify it,
- do not copy large portions of code into this repository,
- document what appears reusable,
- document what should be implemented independently,
- preserve proper attribution.

Our repository must remain clearly independent from the upstream ERDES repository.

==================================================
DATA RULES
==================================================

Never commit:
- raw ERDES ultrasound videos,
- downloaded datasets,
- large model checkpoints,
- experiment logs,
- generated outputs,
- secrets,
- environment files containing credentials.

The repository should contain code, documentation, configuration, lightweight metadata where appropriate, and reproducible instructions for obtaining external data.

==================================================
MEDICAL / RESEARCH RULES
==================================================

This is an educational/research machine-learning project.

Do not describe any model as a clinically validated diagnostic system unless evidence explicitly supports that claim.

Clearly distinguish:
- dataset labels,
- model predictions,
- experimental conclusions,
- clinical claims.

Do not infer medical conclusions from the student's personal history.

Personal experience with retinal detachment is motivation for choosing the topic, not scientific evidence.

==================================================
INITIAL REPOSITORY STRUCTURE
==================================================

Create a clean minimal research repository.

Use approximately this structure unless inspection shows a better reason to modify it:

.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── PROJECT_SCOPE.md
│   ├── WEEKLY_PLAN.md
│   ├── LITERATURE_NOTES.md
│   ├── ERDES_PAPER_NOTES.md
│   ├── UPSTREAM_ERDES_REVIEW.md
│   ├── EXPERIMENT_LOG.md
│   └── DECISIONS.md
├── prompts/
│   ├── README.md
│   ├── research_review.md
│   ├── dataset_inspection.md
│   ├── baseline_experiment.md
│   └── experiment_review.md
├── notebooks/
├── src/
│   └── rd_ultrasound_ml/
│       └── __init__.py
├── tests/
└── data/
    └── .gitkeep

Do not add unnecessary infrastructure.

Do NOT add yet unless justified later:
- Docker
- DVC
- MLflow
- Weights & Biases
- complex CI/CD
- deployment infrastructure
- web applications
- large config frameworks

==================================================
FILES TO CREATE
==================================================

1. AGENTS.md

Keep it concise.

It should contain only recurring project-wide instructions such as:
- research/project context,
- development workflow,
- use of uv,
- no system-Python modification,
- evidence-based verification,
- testing expectations,
- reproducibility,
- dataset protection,
- upstream repository rules,
- medical-claim limitations,
- Git expectations.

Do not turn AGENTS.md into a giant project report.

--------------------------------------------------

2. README.md

Create a professional initial README containing:

- project title:
  "Retinal Detachment Analysis from Ocular Ultrasound Using Deep Learning"

- course context,
- project status: Week 1 / research and setup,
- high-level motivation,
- ERDES dataset name,
- statement that the final research question is still being refined,
- initial goals,
- repository organization,
- instructions that raw data is not stored in Git,
- placeholder sections for setup, experiments, results, and citation.

Do not claim results that do not exist.

--------------------------------------------------

3. docs/PROJECT_SCOPE.md

Include:

- problem area,
- motivation,
- current project theme,
- known dataset,
- current understanding of possible research directions,
- explicit "not yet decided" items,
- likely primary task: RD vs non-RD,
- possible secondary task: macula intact vs detached,
- candidate research directions:
  - false-negative analysis,
  - class imbalance,
  - probability confidence,
  - calibration,
  - reliability,
  - sample-level error estimation.

Clearly label these as candidate directions rather than commitments.

--------------------------------------------------

4. docs/WEEKLY_PLAN.md

Create the semester milestone plan.

Use these stages:

Week 1:
project foundation, paper review, upstream inspection

Week 2:
dataset acquisition and exploratory data analysis

Week 3:
working end-to-end baseline pipeline on a small subset

Week 4:
full reproducible baseline

Week 5:
error analysis, especially false negatives

Week 6:
first research experiment

Week 7:
reliability/confidence/calibration and/or second controlled experiment

Week 8+:
secondary task if feasible, final comparison, report/presentation preparation

For each week include:
- objective,
- concrete deliverables,
- evidence to show in weekly presentation,
- risks/blockers,
- next milestone.

--------------------------------------------------

5. docs/LITERATURE_NOTES.md

Create a structured literature-review template with fields:

- Citation
- Research question
- Dataset
- Input modality
- Sample size
- Model(s)
- Evaluation metrics
- Main findings
- Limitations
- Relevance to our project
- Possible follow-up question

Do not invent papers.

--------------------------------------------------

6. docs/ERDES_PAPER_NOTES.md

Create a structured template for the ERDES paper.

Sections:

- Citation
- Dataset purpose
- Clinical problem
- Data source
- Dataset composition
- Labels/classes
- Preprocessing
- Train/validation/test strategy
- Architectures
- Training setup
- Evaluation metrics
- Main results
- Error analysis
- Limitations
- Reproducibility resources
- Questions for our professor
- Potential research gaps
- What we should reproduce
- What we should NOT simply repeat

If the ERDES paper is locally available, fill sections only from evidence in the paper.
Otherwise leave evidence-dependent sections as TODOs.

Do not invent details.

--------------------------------------------------

7. docs/UPSTREAM_ERDES_REVIEW.md

If the upstream ERDES repository is available, inspect it and document:

- repository structure,
- environment/dependency files,
- data-loading implementation,
- preprocessing,
- training entry points,
- evaluation entry points,
- configuration system,
- implemented models,
- supported tasks,
- logging/checkpoint behavior,
- testing,
- assumptions,
- potential outdated dependencies,
- reusable components,
- components we may want to implement ourselves,
- questions requiring further investigation.

Do NOT modify upstream code.

If upstream is not available, leave this document as a review checklist.

--------------------------------------------------

8. docs/EXPERIMENT_LOG.md

Create a reusable experiment template.

Each experiment should eventually record:

- Experiment ID
- Date
- Research question / hypothesis
- Dataset split
- Model
- Preprocessing
- Changed variable
- Fixed variables
- Random seed
- Training configuration
- Metrics
- Results
- Failure modes
- Interpretation
- Limitations
- Next action
- Git commit / tag if available

No experiments exist yet, so do not fabricate any.

--------------------------------------------------

9. docs/DECISIONS.md

Create a lightweight decision log.

Format:

## YYYY-MM-DD — Decision title

Context:
...

Decision:
...

Reason:
...

Alternatives considered:
...

Consequences:
...

Initial decisions may include:
- use ERDES as approved dataset,
- keep upstream implementation separate,
- use uv for project management,
- defer final Python version until dependency compatibility is inspected,
- do not commit raw data.

--------------------------------------------------

10. prompts/

Create reusable Codex prompt templates.

prompts/README.md should explain that these prompts are starting points and should be customized with concrete goals, context, constraints, and completion criteria.

Create:

research_review.md
- for reviewing a paper/literature source without inventing claims.

dataset_inspection.md
- for inspecting ERDES metadata/files/splits before model work.

baseline_experiment.md
- for planning and running a controlled baseline once we reach that stage.

experiment_review.md
- for reviewing results, logs, metrics, diffs, limitations, and reproducibility after an experiment.

Prompts must follow this structure:

Goal
Context
Constraints
Completion criteria
Checks / evidence required

==================================================
GITIGNORE
==================================================

Create a suitable .gitignore covering at least:

.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
.env
.env.*
.ipynb_checkpoints/
data/*
!data/.gitkeep
checkpoints/
*.ckpt
*.pt
*.pth
logs/
outputs/
runs/
wandb/
.vscode/
.idea/
.DS_Store

==================================================
PYPROJECT / UV
==================================================

Initialize the repository as a uv-managed Python project if it is not already initialized.

Do not select an older Python version merely for compatibility with historical upstream code.

Do not add the full ML dependency stack yet.

Keep pyproject.toml minimal until dependency requirements have been researched.

If uv initialization generates sample application files that do not suit this package structure, clean them up carefully.

==================================================
GIT WORKFLOW
==================================================

Assume this is a three-person group repository.

Document a lightweight workflow:

main
feature/<short-description>

Developers should:
1. create a feature branch,
2. make focused changes,
3. run relevant checks,
4. review git diff,
5. commit with clear messages,
6. push,
7. open a Pull Request,
8. have another teammate review before merging where practical.

Do not configure complex branch automation yet.

==================================================
COMPLETION CRITERIA
==================================================

This initialization task is complete only when:

1. The repository structure is clean and understandable.
2. AGENTS.md exists and is concise.
3. README.md accurately describes the current project state.
4. All requested docs/templates exist.
5. Reusable prompt templates exist.
6. .gitignore protects data/checkpoints/secrets.
7. pyproject.toml is minimal and compatible with uv.
8. No raw dataset or large binary assets have been added.
9. No model training code has been implemented.
10. No unsupported research claims have been invented.
11. Any professor/TA guidance actually available was incorporated and explicitly noted.
12. Repository files have been reviewed with git diff/status.
13. Basic structural checks have been run.
14. You provide a final report containing:
    - files created,
    - files modified,
    - commands run,
    - checks performed,
    - anything not checked,
    - assumptions made,
    - unresolved questions,
    - recommended next task.

Before making the changes, briefly summarize your planned approach.

Then implement the initialization.

After implementation, inspect the final diff and report the evidence that the initialization is complete.