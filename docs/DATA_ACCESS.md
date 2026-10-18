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
