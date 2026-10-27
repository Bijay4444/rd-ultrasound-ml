# ERDES grouping and video access

Checked 2026-10-25 against author sources and official Hugging Face documentation.
This note supports the move from metadata auditing to training-video inspection.

## Patient grouping: documented limitation

The latest arXiv version is **v2, revised 4 March 2026**. Its **Data Record**
section states that splitting occurred at clip level because patient identifiers
were removed. It explicitly allows clips from one patient to occur across
training, validation, and test partitions. The same section describes filenames
as `{random_id}_{counter}.mp4` and clip IDs as unique clip identifiers; it does
not define filename prefixes as patient or eye identifiers. It also notes that
one encounter can produce multiple acquisitions. See [paper v2, Data Record](https://arxiv.org/html/2508.04735v2#S3).

**Interpretation for this project:** absence of shared clip paths establishes
clip separation only. The supplied splits were not constructed to enforce
patient separation. They cannot support a claim of evaluation on unseen patients.
The amount of actual patient overlap cannot be calculated from the released
metadata without a verified grouping map. Preserve the original splits for
benchmark comparison and report this limitation; do not fabricate patient groups
from filename prefixes. These conclusions concern experimental design, not
clinical validation. [Evidence: paper v2, Data Record](https://arxiv.org/html/2508.04735v2#S3).

## Author dataset access and licensing evidence

The authors' [Hugging Face dataset card](https://huggingface.co/datasets/pcvlab/erdes/blob/7867e2504ed3845a46ff7790a2f8c3ffcdc08947/README.md)
declares `license: cc-by-4.0`, provides a paper citation, and describes cropped
MP4 videos organized by diagnostic class. Record this declared dataset license
and attribution with acquired files. This is separate from the upstream code
repository's license.

The [Zenodo v1 record](https://zenodo.org/records/18644370) separately publishes
`clips.zip` (about 1.9 GB), metadata, and split files with MD5 checksums. Its
rendered rights section did not expose a named license during this inspection;
use the explicit author dataset-card declaration for the Hugging Face source
and retain source provenance rather than treating every distribution as
interchangeable. The archive need not be downloaded for a small video inspection.

## Immutable individual-video downloads

The parent task fetched the public [dataset API with file metadata](https://huggingface.co/api/datasets/pcvlab/erdes?blobs=true)
and saved the response locally as ignored `data/erdes/huggingface_source.json`.
I inspected that response: repository revision is
`7867e2504ed3845a46ff7790a2f8c3ffcdc08947`; `cardData.license` is `cc-by-4.0`;
there are 5,404 sibling file entries. Video entries include size and
`lfs.sha256`. Example:

```text
clips/Retinal_Detachment/Macula_Detached/TD/405744_00090.mp4
size: 674444 bytes
SHA-256: 8dba9ff3d435c49e4d17ad40b212aba559098d4015abb91a9fffca656c51d258
```

Use `clips/` followed by a metadata `file_path`, with the full repository
revision in the download URL:

```text
https://huggingface.co/datasets/pcvlab/erdes/resolve/7867e2504ed3845a46ff7790a2f8c3ffcdc08947/clips/<file_path>
```

Hugging Face documents revision-specific single-file downloads and requires the
full commit hash when specifying a commit. Its file API documents LFS SHA-256
and size metadata. Verify acquired bytes against the pinned file entry and
record both revision and checksum. [Download guide](https://huggingface.co/docs/huggingface_hub/guides/download#from-specific-version),
[file API reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree).

## Scope and remaining checks

Select inspection clips only from the verified training CSV. Join by the full
relative path and preserve the supplied labels. Record the deterministic sample,
source revision, expected and observed hashes, decode results, and runtime.
Do not use inspection findings to claim label correctness, population prevalence,
or model accuracy. Paper grouping statements are author-reported; this research
did not recover patient identities or independently measure patient overlap.
The research subtask did not download or decode videos; those checks belong to
the inspection notebook.
