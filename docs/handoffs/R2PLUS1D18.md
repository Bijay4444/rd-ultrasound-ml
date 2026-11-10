# Teammate B: R(2+1)D-18 RD video classification

Status: assignment and implementation brief, 2026-11-10; no new model implemented
or trained by this document. Read the [shared protocol](../MODEL_COMPARISON_PLAN.md)
first; it owns data, optimization, cache integration, evaluation and reporting rules.

## Goal

Train a one-channel torchvision **R(2+1)D-18** classifier from scratch on ERDES
`non_rd_vs_rd`. Compared with teammate A's R3D-18, this architecture factorizes
3D convolutions into spatial and temporal operations with an intervening
nonlinearity. It supplies a second video-learning approach while retaining the
same prepared tensors and training recipe. It is a new comparison, not an ERDES
paper-row reproduction or a guaranteed improvement.
[Official builder](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html),
[original paper](https://arxiv.org/abs/1711.11248),
[source notes](../research/video-model-comparison-sources.md).

## Implement in checkable steps

1. Inspect `AGENTS.md`, the shared protocol, existing training/Kaggle code and
   matching tests. Record your starting Git commit. Use branch
   `feature/r2plus1d18-rd`; proposed experiment ID `r2plus1d18_rd_001_amp`.
2. Add a small model factory in `src/rd_ultrasound_ml/r2plus1d18_baseline.py`.
   Construct `torchvision.models.video.r2plus1d_18(weights=None, num_classes=1)`.
   Its first stem convolution expects three channels: replace **only that first
   convolution** with an otherwise equivalent one-channel `nn.Conv3d`. Preserve
   the factorized stem's later temporal convolution and normalization/activation
   sequence. Initialize the replacement consistently with the builder's Conv3d
   initialization and record it. Leave remaining defaults intact.
3. Confirm input `[B,1,96,128,128]` produces one raw logit per clip, finite loss
   and nonzero finite gradients. Verify that factorized blocks remain present,
   the binary head has one output, and no pretrained weights are downloaded.
   Record the actual parameter count after the grayscale/binary changes.
4. Add `configs/r2plus1d18_rd_001_training.json` and a model-specific notebook,
   suggested `notebooks/kaggle_r2plus1d18_training.ipynb`, plus submission bundle.
   Implement or coordinate the shared cache adapter; preserve baseline notebooks
   and builders. Completion: synthetic integration validates unchanged cached
   inputs under the new run identity and rejects modified provenance/configs.
5. Probe real training clips and complete one full timing epoch. Coordinate the
   same physical batch with teammate A before starting the final run. Factorized
   convolutions add intermediate activations; **do not assume this model is
   faster or uses less memory**. Start fresh weights/seed 42 for the 50-epoch
   experiment and preserve resumable state. Record a partial run honestly if the
   measured budget cannot support completion.
6. Reload selected weights, verify validation predictions/metrics, and return the
   shared artifact package. Freeze all choices before new official-test inference
   and disclose reuse of the official benchmark.

Keep `weights=None` and inherited grayscale scaling. Generic pretrained RGB
video transforms and 16-frame recipes are different protocols. R(2+1)D-18 is
still a video classifier, not a spatial segmentation model.

## Implementation references and expected files

- Reuse `src/rd_ultrasound_ml/unetpp_training.py` where applicable; record the
  installed torchvision version along with the existing runtime fields.
- Follow the shared protocol for the full-config cache digest and identity-bound
  resume. Keep attached prepared arrays read-only.
- Final report: `docs/R2PLUS1D18_RESULTS.md`, including exact epochs, partition,
  BatchNorm/physical batch, recipe limitations, memory and runtime measurements.
- Explainability integration: expose the final feature tensor before pooling
  (`backbone.layer4` if your wrapper names the VideoResNet `backbone`). Return
  the exact module path and tensor shape for Bijay's Grad-CAM code.

## Completion criteria

Focused model/adapter/notebook checks pass, applicable existing checks pass,
and logs support every training claim. Export a resolved config, complete
validation history, selected checkpoint identity and recomputable per-clip
predictions. The full-data result is either 50 epochs or explicitly partial.
All experimental deviations appear in the report. No model/threshold choice
uses a new test score.

## Paste into Claude from the repository root

```text
Read AGENTS.md, docs/MODEL_COMPARISON_PLAN.md and docs/handoffs/R2PLUS1D18.md.
Implement my assigned R(2+1)D-18 RD video classifier and model-specific training
workflow. Inspect existing code/tests, propose a short plan, then implement and
verify small changes. Keep the completed U-Net++ source/configs/results intact.
Use torchvision r2plus1d_18(weights=None, num_classes=1), adapt only the first
stem convolution for grayscale, preserve the factorized temporal stem, and
return one raw logit per clip. Retain the official splits and inherited recipe.
Handle the existing cache config digest explicitly while preserving provenance
checks. Prepare the notebook, config, focused checks and report template; run
real-input memory/timing probes before full training. Coordinate physical batch
with the R3D-18 owner. Train when intended GPU/data access is available within
the recorded budget; complete independent implementation if access is missing.
Return the shared protocol's artifact package and exact blockers/deviations.
Freeze final experiment choices before separate official-test inference, disclose
benchmark reuse, and never invent results or call a partial run complete.
```
