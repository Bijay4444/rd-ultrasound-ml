# ERDES release split audit — 2026-10-25

Candidate task: `non_rd_vs_rd`. This audit does not settle the final research
question, architecture, or experimental protocol.

## Sources and local storage

Downloaded the small inputs from [Zenodo v1, record 18644370](https://zenodo.org/records/18644370):

| Input | Published MD5 verified |
| --- | --- |
| `erdes_metadata.csv` | `95828fb5bdffdbf8173b5bcfc99e8148` |
| `erdes_metadata.json` | `0d62fdff53d769d286a37c901fcf298e` |
| `splits.zip` | `c5630e3f056ce358dcd384aaed946e64` |

Inputs live under ignored `data/erdes/`. The notebook saves byte-for-byte copies
of the three candidate-task CSVs under `data/erdes/splits/non_rd_vs_rd/` and aggregate
results under ignored `outputs/split_audit/`. Downloads total 629,178 bytes.
No videos were downloaded for this audit. The separate upstream checkout was read only.

The numeric label mapping is **0 = non-RD** (normal or PVD), **1 = RD**,
supported by the authors' [task definitions at the inspected revision](https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/data/create_data_splits.py)
and checked against every matched metadata row. No upstream source code was copied.

## Observed counts and integrity checks

| Partition | Clips | Non-RD | RD |
| --- | ---: | ---: | ---: |
| Training | 3,873 | 3,511 | 362 |
| Validation | 431 | 391 | 40 |
| Test | 1,077 | 977 | 100 |
| Total | 5,381 | 4,879 | 502 |

Observed: no repeated paths within a partition; no shared paths between any pair
of partitions; all paths match metadata; all numeric labels agree with metadata;
the three partitions cover all 5,381 metadata rows. Supplied assignments were preserved.
These are dataset-integrity findings, not model results or final test evaluation.

## Grouping follow-up and remaining video checks

The release metadata has no explicit patient/eye identifier. Its accompanying
JSON describes `clip_id` only as a unique clip identifier. The inspected upstream
split generator stratifies video rows by label and has no explicit patient/eye
grouping step. Neither observation establishes whether the supplied partitions
are patient independent. Do not interpret filename prefixes as patient IDs
without a documented definition.

Follow-up: [paper v2, Data Record](https://arxiv.org/html/2508.04735v2#S3) explicitly
states that splits were generated at clip level because patient identifiers were
removed. Clips from one patient may cross partitions. Thus the supplied partitions
do not enforce patient separation; actual patient-overlap counts are unavailable.
Keep supplied splits for benchmark work and report this limitation rather than
claiming evaluation on unseen patients. See [grouping/access research](research/erdes-grouping-and-access.md).

The split audit does not check video readability or duplicated video content.
The separate [video inspection](VIDEO_INSPECTION.md) checks a small training sample;
a full video audit remains pending.

## Reproduce

Run `notebooks/split_audit.ipynb` from its first code cell on a CPU runtime, following
`notebooks/README.md`. It runs independently of the metadata exploration notebook:
it chooses storage, verifies the metadata, and downloads the split archive itself.
It uses pandas and standard-library ZIP handling. No training dependencies are
required. It saves `split_summary.csv` and `split_audit.json`,
including source checksums and runtime versions. Preserve these files outside
temporary Colab storage.

## Verification performed

Initially executed the three split-audit code cells locally against the real release
inputs using Python 3.14.4, pandas 3.1.0, and matplotlib 3.11.2 in a temporary
uv environment. Notebook schema validation and saved JSON/CSV output checks
passed. Cached-input reuse, a simulated first download, rejection of a corrupt
archive, and rejection without overwriting a modified CSV passed. The three
release split CSVs also matched the inspected upstream copies byte for byte.

After moving the audit into `notebooks/split_audit.ipynb`, reran all five code
cells in order from a fresh execution namespace against cached release inputs,
with network access disabled in the verification harness. All checks passed and
outputs were saved under `outputs/split_audit/`. Both notebook schemas validated;
the first notebook's original code and saved outputs were verified unchanged.
The standalone notebook's recorded runtime includes Python and pandas only.

Commands included `python3 /tmp/erdes_download_audit_inputs.py` for acquisition,
`uv --cache-dir /tmp/dmml-uv-cache run --offline --with pymupdf --with pandas --with matplotlib --with nbformat python /tmp/verify_erdes_split_notebook.py`
for the temporary verification script, `git diff --check`, and `git check-ignore`
for local input/output paths. Temporary verification tools did not change
`pyproject.toml` or `uv.lock`. Reviewed the resulting notebook and documentation diff.

The user's saved split-audit notebook now also contains completed Colab executions
1–5, with Python 3.13.16 and pandas 2.2.3, and all displayed checks passed. Its
code and saved results were preserved during the paper follow-up; only the
grouping explanation and next-notebook link were updated.

Not run by the agent in the split audit: a fresh Colab execution, a full rerun of
the original metadata sections, or training/evaluation. Video checks are recorded
separately in `VIDEO_INSPECTION.md`. The user's saved audit executions and the
first notebook's metadata outputs were preserved. Course guidance was read from the available
`docs/course-guideline/AI_Coding_Agents_Practical_Reading_Guide.pdf`; the directory
name still differs from `docs/course-guidance/` specified in `AGENTS.md`.
