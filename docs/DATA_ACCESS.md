# ERDES acquisition and provenance

Start from the authors' [repository](https://github.com/OSUPCVLab/ERDES) and
[dataset page](https://huggingface.co/datasets/pcvlab/erdes). The local upstream
README also lists [Zenodo](https://zenodo.org/records/18644370).
Verify the selected release, license/access terms and paper citation before download.
No dataset was downloaded or copied by initialization.

1. Record source URL, release/revision, date, license and required citation.
2. Download the chosen release following that source's instructions into ignored `data/`.
3. Record archive checksum and extraction layout locally; retain non-sensitive provenance in documentation.
4. Audit metadata, label definitions, missing/corrupt videos, class counts,
   patient/eye identifiers, duplicates and any supplied split files.
5. Check patient/eye overlap across partitions and document the selected split policy.
   Do not regenerate or overwrite existing splits without a documented reason.
6. Store only approved non-sensitive lightweight manifests in Git; review paths and identifiers first.

Existing local upstream data remains in the separate read-only reference checkout.
Its existence does not establish completeness, provenance, or suitability of splits.

## First audit inputs

For metadata and split exploration, download only `erdes_metadata.csv` and
`splits.zip` from Zenodo record 18644370 into ignored `data/erdes/`. The notebook
handles these downloads and verifies published checksums. The small
`erdes_metadata.json` is also available for field and label definitions.
The initial local acquisition includes these three files and an ignored
`acquisition.json` with URLs, download time, sizes, and hashes.

Run `notebooks/setup_and_eda.ipynb` for metadata exploration, then use the independent
`notebooks/split_audit.ipynb` to save unchanged candidate-task split CSVs and
reproduce the audit. See [observed counts and limitations](SPLIT_AUDIT.md).
No full video download is needed for this stage. With Colab, these folders are
on the remote runtime under `/content`; the notebook downloads the inputs there
independently of local storage.

## Small training-video sample

`notebooks/video_inspection.ipynb` selects two clips per metadata subtype from
the official `non_rd_vs_rd` training CSV using a fixed SHA256 ranking with seed
42. It downloads eight videos (4,246,987 bytes total) individually from the
authors' Hugging Face dataset at revision
`7867e2504ed3845a46ff7790a2f8c3ffcdc08947` into ignored `data/erdes/clips/`.
Expected file sizes and Git LFS SHA256 hashes are recorded in the notebook and
verified before writing or reusing files. No validation/test videos are acquired
by this workflow. A whole-dataset archive is unnecessary at this stage.

The author dataset card declares CC-BY-4.0; the notebook and output JSON retain
the source, revision, license declaration, and paper citation. See
[source evidence](research/erdes-grouping-and-access.md) and
[inspection instructions/results](VIDEO_INSPECTION.md). Raw videos and generated
frame images remain outside Git; clear video notebook outputs before committing.
