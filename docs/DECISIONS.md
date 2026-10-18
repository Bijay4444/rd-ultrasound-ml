# Decisions

## 2026-10-18 — Minimal research foundation

Context: Week 1; existing initialization brief and course reading guide inspected.

Decision: use ERDES, preserve upstream as a separate read-only reference, use uv,
protect data and artifacts from Git, and defer deployment infrastructure.

Reason: prioritize a checkable experiment and learning before application delivery.

Alternatives considered: fellowship API layout; full upstream framework/template.

Consequences: no API, database, Docker or training implementation yet. The final
research question, stack and interpreter pin remain undecided. Python metadata
starts at 3.14 for setup only; recheck compatibility when choosing dependencies.
No upstream code reuse has occurred.
