# Learning by doing

The fellowship is relevant: its Git, documentation, dependency isolation and
configuration lessons transfer. Its `app/`, CRUD, ORM database, API schemas and
Docker services support a deployed Iris API. Our immediate deliverable is a
research experiment, so start with data understanding and evaluation. See
[the checked sources](research/setup-sources.md).

## Twelve-factor practices now and later

| Principle | Application here |
| --- | --- |
| Codebase | One team Git repository; feature branches and review. |
| Dependencies | Declare with uv; retain the lockfile; isolate `.venv`. |
| Config | Keep experiment settings versioned; secrets and machine-specific paths outside Git. |
| Backing services | Revisit if a database or hosted service becomes necessary. |
| Build/release/run | Record code and environment for each experiment; deployment stages can wait. |
| Processes | Scripts should receive explicit inputs and write identifiable outputs. |
| Port binding | Revisit if an API is a justified deliverable. |
| Concurrency | Plan training resources when a baseline exists. |
| Disposability | Checkpoint long runs when training begins. |
| Dev/prod parity | Teammates share the lockfile and documented environment. |
| Logs | Capture experiment evidence, keeping bulky logs outside Git. |
| Admin processes | Use documented one-off scripts for data checks later. |

These are selective adaptations. Research also needs data provenance, suitable
splits, seeds, controlled comparisons and honest reporting, which the app
methodology does not supply by itself.

## Learn in this order

1. Python basics: functions, modules, paths, lists/dictionaries; inspect a small CSV.
2. Samples and labels: distinguish patient, eye, video and frame; draw their relationships.
3. Train/validation/test: training fits weights; validation informs choices; test estimates final performance.
4. Classification: confusion matrix, sensitivity/recall, specificity, precision and class imbalance.
5. Baseline: input tensor, prediction, loss, optimizer, epoch and overfitting.
6. Controlled comparison: change one factor; record seeds, configuration and limitations.
7. Calibration if selected: confidence should match observed frequencies across groups.

For each topic: explain it in plain words, work a tiny example, predict the
answer, run it, then explain any mismatch. Keep a learning note with: question,
current explanation, worked example, what was checked, remaining confusion.
Ask for a from-scratch explanation whenever a step is unclear; understanding the
experiment matters more than adding infrastructure.

## Course evidence

Read: `course-guideline/AI_Coding_Agents_Practical_Reading_Guide.pdf` (all pages,
text extracted using a temporary PyMuPDF tool). It recommends explore/plan/change/
check/review, clear completion criteria, concise recurring instructions, leakage
prevention and careful calibration evaluation. Those practices inform this setup.
`AGENTS.md` names `course-guidance/`; the supplied directory is `course-guideline/`.
No syllabus, grading rubric or TA setup report was found. Obtain these before
claiming the plan meets maximum-mark requirements. Track contributions and show
weekly evidence: a question, an artifact, a check, and the next decision.
