# First baseline and controlled GPU comparisons

Status: the first five-epoch Colab T4 pilot is complete; its selected model
predicted every validation clip as non-RD at 0.5. The next weighted-loss comparison
is implemented and awaits execution. See [the experiment log](EXPERIMENT_LOG.md).
Earlier verification notes below record the status at those stages. This is an
engineering baseline to establish a reference, not the final research question
or a reproduction of the authors' benchmark. The user selected a free Colab GPU.
The course rubric and final research contribution still need confirmation.

## Task and data boundaries

Classify one video clip as non-RD (0, normal or PVD) or RD (1). Preserve the
official `non_rd_vs_rd` partitions. The proposed pilot uses 256 clips selected
only within the 3,873 training rows, approximately retaining its class proportion:
24 RD and 232 non-RD. Rank clip IDs within each binary class by SHA256 of
`42:<clip_id>` and retain the required count; record that exact manifest and hash
before fitting. This is a documented development subset, not a regenerated split.
The training notebook saves the exact manifest and hash before fitting. Preparation
acquires only inspection clips; the pilot notebook acquires its selected videos
after the diagnostic passes.

Use all 431 supplied validation clips (40 RD, 391 non-RD). Keep test membership,
videos, predictions, and performance out of the training notebook. Save separate
train/validation manifests and preserve the full official CSV checksums.

The paper documents clip-level splitting and possible same-patient overlap, so
results describe this clip-level benchmark and cannot establish performance on
unseen patients. Actual patient-overlap counts cannot be measured from the
released identifiers. [Paper v2, Data Record](https://arxiv.org/html/2508.04735v2#S3).

## Fixed initial input representation

| Choice | Initial value | Reason |
| --- | --- | --- |
| Frames per clip | 16, uniform frame indices including endpoints | Bounded input size covering the clip |
| Short clips | Repeat selected source frames | Keep a fixed length without inventing motion |
| Image channels | Arithmetic channel mean to grayscale | Simple explicit treatment of ultrasound imagery |
| Spatial input | Symmetric zero padding, then 128×128 area resize | Preserve released field of view and bound compute |
| Scaling | Float32, divide by 255 | No fitted statistics or dataset-wide normalization |
| Batch layout | `[B,1,16,128,128]` | Channel, temporal, and spatial dimensions explicit |
| Augmentation | None | Establish a simple initial reference |

Channel averaging, square padding, and division by 255 adopt attributed ideas
from [paper Table 6](https://arxiv.org/html/2508.04735v2#S4). The independent
preparation code uses uniform frame selection instead of the paper's temporal
interpolation. No upstream source code was copied. These settings are starting
choices; resizing may lose fine detail and uniform frame positions do not equal
uniform physical timing across clips with different FPS.

## Proposed first model and training settings

The first candidate is a compact 3D CNN from scratch: three convolutional blocks
with 8, 16, and 32 output channels, followed by global average pooling and one
output logit. This is smaller than attempting the paper's full architectures
before checking the pipeline. Each block is Conv3d (kernel 3, stride 2, padding 1,
bias enabled) followed by ReLU. The head averages the temporal/spatial axes
directly and applies Linear(32,1), for 17,585 trainable parameters. A direct mean
implements global average pooling without AdaptiveAvgPool3d's unsupported
deterministic CUDA backward operation.
[PyTorch deterministic-operation support](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

Initial pilot: Adam, learning rate 0.001, batch size 8, five epochs, and unweighted
`BCEWithLogitsLoss`. Feed logits to this loss and apply sigmoid only for reporting
probabilities. Do not add both a sigmoid output layer and a logits-based loss.
[PyTorch loss API](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
Class weighting/resampling is deferred to a documented controlled comparison;
the initial model may collapse to predicting non-RD, which must be reported.

Before the pilot, verify a forward/backward pass and try to overfit four balanced
training clips. Failure to reduce that training loss substantially is a pipeline
diagnostic, not a reason to evaluate the test set. Run these checks before the
larger download/training stage.

Record Python/numpy/PyTorch seeds (42), loader ordering, software versions, GPU
name, determinism settings, configuration hash, manifests, runtime, and training
history. Recreate a fresh model after the overfit check. Exact reproducibility
across hardware and software versions is not guaranteed.
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Validation and model selection

Select the checkpoint by highest validation **average precision**, keeping the
earliest epoch on ties. Average precision is the recall-weighted summary defined
by the metric implementation; do not label it as trapezoidal PR-AUC.
[scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

At the initial threshold of 0.5 report TP, TN, FP, FN, RD sensitivity, specificity,
precision, F1, balanced accuracy, accuracy, average precision, and ROC-AUC.
Record the positive class explicitly. Report undefined metrics as undefined
with a reason rather than silently hiding them. Always include the constant
non-RD reference on the same validation rows: accuracy alone is insufficient.
Any later threshold tuning uses validation only and is documented as a protocol
change. Final test evaluation stays separate until development is complete.

## Current artifacts and checks

Machine-readable proposal: [configs/baseline_001.json](../configs/baseline_001.json).
Run [baseline_preparation.ipynb](../notebooks/baseline_preparation.ipynb) independently
on CPU to verify preprocessing. It uses the eight inspection clips, not the
256-row pilot subset, and reads no validation/test CSV.

All eight real training clips produced a finite float32 batch of shape
`[8,1,16,128,128]`, range [0, 0.9898], consuming about 8.39 MB. Checked the exported
preview image. Six unit checks passed, including synthetic short-video decoding,
repeated-frame behavior, padding/channel averaging/scaling, metadata mismatch,
missing files, and playback compatibility. The embedded notebook configuration
matched the tracked JSON and the saved output JSON.

Commands: temporary uv runs of `/tmp/verify_baseline_preparation.py` and
`python -m unittest discover -s tests -v` with pandas, matplotlib, nbformat,
opencv-python-headless, and ipython; `git diff --check` and `git check-ignore`.
Outputs are ignored under `outputs/baseline_preparation/`: `baseline_001.json`,
`preparation_summary.json`, and `preprocessed_examples.png`.

The user's saved preparation notebook has completed Colab executions 1–8:
Python 3.13.18, numpy 2.1.3, pandas 2.2.3, OpenCV 5.0.0. Its input check passed
with shape `[8,1,16,128,128]`, float32 values from 0 to 0.9897523. That executed
notebook is preserved without changes in this implementation.

## Run the GPU pilot

Open [baseline_training.ipynb](../notebooks/baseline_training.ipynb), select a
Colab GPU runtime, reconnect, and run one section at a time from the top. No
checkout or previous notebook state is needed. The embedded configuration matches
[baseline_001_training.json](../configs/baseline_001_training.json), which extends
the preparation proposal with exact model/diagnostic settings and its source hash.
The original preparation configuration is retained.

The notebook first confirms CUDA, records seeds/determinism/versions, and saves
training/validation manifests. It downloads four training clips and requires a
finite forward/backward pass and an overfit check: loss must fall to at most half
its initial value and training accuracy must reach 1.0, within 300 Adam steps
at 0.003. These diagnostic weights are discarded. A failed diagnostic records
evidence and blocks the larger acquisition; it does not justify changing splits.

After the check, source SHA256 hashes and byte sizes at the pinned author revision
are obtained only for the selected paths via the
[Hugging Face paths-info API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.get_paths_info).
Every cached/downloaded clip is verified, and preprocessing rejects frame-count
mismatches. Memory-mapped float32 inputs use about 0.72 GB of runtime disk.
The pilot starts from a fresh seeded model. It evaluates all 431 validation rows
each epoch and saves checkpoints, predictions, metrics, and history automatically.
The always-non-RD validation reference has accuracy 391/431 (about 90.72%),
sensitivity 0, specificity 1, balanced accuracy 0.5, and undefined precision;
this is a reference calculation, not a trained-model result.

Each run uses a unique ignored `outputs/baseline_training/<run>/` directory.
Set `USE_GOOGLE_DRIVE = True` before the first cell for persistent checkpoints,
or download the results ZIP from the final section before disconnecting. The
ZIP excludes videos/arrays. Preserve the executed notebook too. Interrupted runs
are partial; there is no automatic resume. Rerunning from the top starts a new run.

Colab CUDA execution, the full pilot acquisition/fitting, validation results from
that pilot, and final test evaluation remain pending. No training dependencies
are added to `pyproject.toml` or `uv.lock`; temporary CPU verification uses uv,
and Colab deliberately retains its managed Python/CUDA installation.

## Pilot implementation verification — 2026-10-28

All 16 unit checks passed, covering preprocessing/playback and the new selection,
label/path checks, model gradients, metric definitions/undefined values, checkpoint
ties, validation ordering, and corruption rejection. Notebook schema/code syntax,
configuration agreement, preprocessing agreement, Git whitespace checks, and
ignored artifact paths passed. The executed preparation notebook was not edited.

A temporary CPU harness used the actual official training/validation inputs and
the four cached diagnostic training videos. The 17,585-parameter model passed
forward/backward and overfit checks: initial loss 0.6964316, loss 0.3241498 and
training accuracy 1.0 after 25 steps. These are diagnostic training measurements,
not validation performance. The harness discarded the diagnostic weights.
A separate two-epoch synthetic fixture passed training, checkpoint reload,
prediction agreement, and result ZIP checks. No synthetic metrics are reported
as ERDES results. CPU PyTorch was 2.15.0+cpu on Python 3.14.4; Colab will record
its own environment and must pass its own GPU checks.

The live source API returned matching SHA256/size for the four diagnostic paths.
Offline comparison with the previously acquired pinned source listing covered
all 687 selected videos, totaling 248,833,155 source bytes (about 249 MB); the
remaining videos were not downloaded. Verification evidence is ignored under
`outputs/baseline_training_verification/`.

Commands run: temporary uv execution of `/tmp/verify_gpu_pilot.py` and
`/tmp/verify_pilot_source_api.py`, `python -m unittest discover -s tests -v`,
notebook validation, `git diff --check`, and `git check-ignore`. The temporary
CPU environment uses ignored `outputs/.pilot-uv-cache/`, because `/tmp` lacked
space for the PyTorch wheel. The full verification command was:

```bash
uv --cache-dir outputs/.pilot-uv-cache run --offline --no-project \
  --with torch --with pandas --with nbformat --with opencv-python-headless \
  --with ipython --with scikit-learn --with huggingface-hub \
  --index https://download.pytorch.org/whl/cpu \
  python -m unittest discover -s tests -v
```

Not run: Colab/CUDA execution, full pilot acquisition or five-epoch fitting,
pilot validation performance, full-dataset checks, or final test evaluation.
The course PDF was read from `docs/course-guideline/`; `AGENTS.md` still names
`docs/course-guidance/`. The grading rubric/final contribution remain unconfirmed.

## Colab acquisition interruption and recovery — 2026-10-28

The user's executed notebook confirms Tesla T4/CUDA availability and a passed
four-clip diagnostic (loss 0.6964316 → 0.3242589, accuracy 1.0 at step 25).
The subsequent 687-clip acquisition printed progress through 100 before
`urllib.error.HTTPError: HTTP Error 503: Service Unavailable`. The traceback
points to `urlopen` while acquiring a video, after source manifests were saved
and before preprocessing or pilot fitting began. This is evidence of an HTTP
service failure, not of invalid tensors or GPU failure. Its server-side cause
and duration are unknown. See [HTTP 503 semantics](https://www.rfc-editor.org/rfc/rfc9110.html#name-503-service-unavailable).

Replayed that failure deterministically against the actual acquisition loop,
with 100 cached files and a simulated 503 on the next request. The test failed
with the same exception before the fix. Moved `download_checked` into a standalone
helper cell, with up to five attempts for transient HTTP/connection/read errors,
increasing waits, and Retry-After handling. Permanent errors and hash mismatches
still stop immediately; cached bytes remain checked, and no clip is dropped.
The replay now passes and a second acquisition uses the cache without networking.

In the current connected Colab session, run **Download helper — safe to rerun
after an HTTP error**, then rerun the failed acquisition/preprocessing cell.
Do not rerun the runtime/storage cell: the helper deliberately leaves `run_id`,
manifests, and `diagnostics_passed` intact. Progress recounts cached files from 1.
If retries are exhausted, retain the final clip-specific message and retry later.

Archived the user's executed failure notebook under ignored
`outputs/baseline_training_verification/baseline_training.http503.<timestamp>.ipynb`
before editing. Other saved executions were retained; only the failed cell's
outputs were cleared. Changed the training notebook, its unit tests, added
`tests/test_training_downloads.py`, and updated notebook/test guides and this plan.

Checks: all 22 tests passed via temporary uv `python -m unittest discover -s tests -v`;
the six download-only regressions also passed with
`python3 -m unittest discover -s tests -p test_training_downloads.py -v`.
Notebook schema/code syntax, preservation of other executions, and `git diff --check`
were checked. No dependencies, model settings, or dataset assignments changed.
Not run by the agent: retrying the user's Colab download, remaining video acquisition,
or full pilot fitting/evaluation. Completion of the external download remains to
be confirmed in the user's session.

## Preprocessing range and rerun fixes — 2026-10-28

The user's next saved execution verified all 687 video files; retries recovered
the observed 503 responses. Preprocessing then printed progress through 100/256
training clips before the strict `[0,1]` check failed. A rerun hit FileExistsError
because that attempt had already created the array directory. Full pilot fitting
had not started.

Reproduced a numerical failure locally: OpenCV INTER_AREA on a valid saturated
432×432 uint8 frame produced a float32 maximum of 1.000000119 after scaling.
Other tested dimensions reached 1.000000238. Float64 resize also showed tiny
overshoot, so merely changing dtype did not guarantee the bound. The original
error omitted the triggering clip and actual range; its precise pixel values
were unavailable locally. The reproduction demonstrates a failure mode consistent
with the reported exception, rather than measuring that particular remote clip.

Both notebook preprocessing implementations now require uint8 decoder output,
reject nonfinite values or deviations beyond `1e-6`, and clip only numerical
drift to `[0,1]`. The final strict clip range check remains unchanged in substance.
New error messages include the clip identifier and measured bounds. Section 7
reuses its existing directory and rebuilds complete arrays from the same manifests;
it blocks rebuilding if pilot training history/checkpoints already exist. It
saves `preprocessing_implementation.json` with the numerical handling/version.
Dataset selection, labels, model settings, and recorded configuration remain unchanged.

Before edits, archived both executed notebooks under ignored
`outputs/baseline_training_verification/*.before_range_fix.<timestamp>.ipynb`.
Other executions were retained; the failed preprocessing output was cleared.
Changed `baseline_preparation.ipynb`, `baseline_training.ipynb`, their preprocessing
and training tests, the notebook/test guides, and this plan.

The saturated-frame regression failed before the fix; a separate interrupted-array
regression reproduced FileExistsError. Both now pass. All 26 tests passed using
the temporary uv environment and `python -m unittest discover -s tests -v`.
Saturated synthetic frames at all 49 distinct released image dimensions passed,
and all eight cached real inspection clips passed with unchanged source hashes.
These checks do not represent preprocessing of every pilot/validation clip.
Notebook schema/syntax, helper agreement, preservation of other executions,
and `git diff --check` were verified. Final test evaluation was not run.

Recovery: in the existing Colab runtime rerun only the updated section 4
preprocessing function cell, then the updated section 7 cell. Cached videos are
reused and incomplete arrays are rebuilt; confirm complete shapes before step 8.
The full remote preprocessing rerun and pilot training remain to be confirmed.

## Completed first pilot and next comparison — 2026-10-28

The user's saved execution now confirms all 687 hashes, complete prepared arrays
`[256,1,16,128,128]` and `[431,1,16,128,128]`, five training epochs, and the original
results ZIP export. Epoch 5 had the highest observed validation AP (0.0906834),
below the constant-reference AP (0.0928074). At 0.5 all 40 RD validation clips
were missed; ROC-AUC was approximately 0.449105. These are pilot development
findings, not test or clinical performance. Full details are in the experiment log.

The next sections 10–12 of the same notebook first verify saved predictions,
row/label alignment, manifest hashes, and recomputed metrics. They save classwise
score distributions without changing thresholds. If the audit passes, the
controlled experiment changes only the training loss's positive-class weight:
`pos_weight = n_non_rd_train/n_rd_train = 232/24`. The ratio uses training labels
only. This tests a hypothesis about weighting; the baseline outcome does not
prove that imbalance caused the poor ranking.
[PyTorch positive weighting](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

Keep the same arrays, seed, loader order, model initialization, Adam state/settings,
batch size, five epochs, threshold 0.5, and highest-validation-AP selection rule.
The model and optimizer are freshly initialized with the original seed; the
previous checkpoint is not fine-tuned. Validation loss remains unweighted and
therefore comparable. Weighted training loss has a different scale. No resampling,
threshold selection, or test use is introduced. Weighting can shift scores and
does not establish calibrated probabilities.

Tracked proposal: `configs/baseline_002_weighted.json`. The continuation preserves
the original cells/outputs and writes into a new run directory. It reuses current
session arrays and downloads no data. Download the original ZIP before continuing
on the existing T4 session; automatic recovery after a runtime reset is not implemented.
The original executed baseline notebook was archived under ignored
`outputs/baseline_training_verification/baseline_training.completed.<timestamp>.ipynb`.

Review AP/ROC-AUC and sensitivity/specificity together. An all-RD result would
not demonstrate useful discrimination even if sensitivity increased. Preserve
failed comparisons; do not react by opening the test set or changing several
variables together. The actual weighted Colab results remain pending.

Verification: all 32 tests passed with the temporary uv command documented above
(`python -m unittest discover -s tests -v`). The six new follow-up checks cover
training-only weights, unchanged control settings, positive loss/gradient scaling,
explicit unweighted validation loss, audit mismatch rejection, and a synthetic
execution of the actual audit/training/export cells on CPU. Checkpoint reload
predictions agreed, original baseline files stayed byte-for-byte unchanged,
and attempted retraining into an existing weighted run was rejected.
Synthetic outcomes are not ERDES performance results.

All notebook schemas and code syntax passed validation; every pre-existing
baseline cell and output was compared with the archived completed notebook and
preserved exactly. The embedded proposal matches the new tracked JSON.
`git diff --check` and ignored archive/checkpoint path checks passed. Added the
weighted configuration and tests, appended the follow-up cells, recorded the
first run in the experiment log, and updated README/notebook/test guides. No
dependencies or original configuration files changed. Not run: the actual remote
prediction audit, weighted Colab fitting, or final test evaluation. Runtime
restoration after a Colab reset remains outside this continuation workflow.
