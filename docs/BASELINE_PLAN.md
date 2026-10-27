# First baseline: proposed GPU pilot

Status: preprocessing verified locally; no model has been trained. This is an
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
Its manifest has not yet been produced and no pilot videos beyond the inspection
sample have been acquired by preparation.

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
before checking the pipeline. Exact blocks and parameter count will be recorded
with the training implementation.

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

Not run: preparation on Colab, PyTorch/GPU checks, model fitting, validation
evaluation, full-dataset video checks, or final test evaluation. No training
dependencies were installed into the project. Next implementation: a separate
GPU training notebook using this configuration after preparation review.
