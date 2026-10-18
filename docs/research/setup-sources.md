# Project setup: source notes

Checked 2026-10-18. These notes inform initial setup; they do not select a final
model, research question, or ML dependency stack.

## Fellowship material and scope

The supplied fellowship homepage is accessible. It describes an Iris classifier
served through FastAPI, with database configuration and Docker Compose. Its
navigation includes documentation, Git, project setup, and twelve-factor topics.
This makes it useful background for software organization and eventual model
serving. Individual project-setup links were not retrievable in this check, so
this is an assessment of the homepage and the structure supplied by the student,
not a review of every lesson.
[Source: fellowship homepage](https://docs.regmiaayush.com.np/fellowship2025/).

The twelve-factor methodology explicitly targets applications delivered as a
service. It covers revision control, dependency isolation, configuration,
deployment, processes, service ports, scaling, and logging.
[Source: The Twelve-Factor App](https://12factor.net/).

**Project recommendation:** apply revision control, explicit dependencies,
documented commands, and clear configuration now. Keep experimental settings
(seeds, split definitions, preprocessing, training parameters) in versioned
configuration; reserve environment variables for secrets and machine-specific
paths. Defer CRUD, ORM models, HTTP schemas, deployment services, and Docker
until an actual research or course requirement needs them. A local research
workflow does not require a web service. These are engineering judgments for
this project's current scope, not requirements imposed by twelve-factor.

## Python selection

Python's official downloads page currently lists Python 3.15.0, released
2026-10-09, as stable, and Python 3.14.8 as the latest 3.14 patch release.
[Source: Python downloads](https://www.python.org/downloads/).

The PyTorch installation guide recommends Python 3.10–3.14 on Linux. Its static
selector displays an older PyTorch version, so it should not be used alone to
infer the latest package version. The PyTorch team's published PyPI release
2.14.1 includes standard CPython 3.14 Linux x86-64 wheels requiring glibc 2.28+;
the checked release-file listing contains no CPython 3.15 wheel.
[Sources: installation guide](https://pytorch.org/get-started/locally/),
[PyTorch release files](https://pypi.org/project/torch/#files).

**Recommendation:** Python 3.14 is the newest demonstrated candidate for a
current PyTorch workflow in this environment. Do not assume Python 3.15 works
because a package declares only a lower Python bound. Before selecting the
project interpreter, verify all actually needed packages, video readers, target
platform, and compute backend together. A successful dependency resolution and
small import/data-reading checks should precede training. No compatibility
installation or ML execution was performed for these notes.

## uv and compute setup

uv uses `pyproject.toml` for project metadata and dependency declarations,
`.python-version` for the preferred interpreter, `.venv/` for the isolated
environment, and `uv.lock` for resolved dependencies. Commit the lockfile;
exclude the environment. `uv sync` recreates the environment, while `uv run`
runs commands within it.
[Source: uv project guide](https://docs.astral.sh/uv/guides/projects/).

PyTorch provides different package sources for CPU and accelerator builds.
uv's official integration documents explicit package indexes and optional
CPU/CUDA choices. Select the backend after checking the machine and intended
training environment, and record the selected source with the dependencies.
[Source: uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

**Recommendation:** start with the smallest useful environment. Add ML packages
when the first data inspection or baseline requires them. Recheck current
release compatibility at that point rather than copying the upstream ERDES
environment or this note's observed versions blindly.
