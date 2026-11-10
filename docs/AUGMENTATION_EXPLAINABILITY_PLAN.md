# Bijay: augmentation ablation and explainability

Prepared 2026-11-10. **Proposed work, not implemented or measured here.** Read
[the shared comparison protocol](MODEL_COMPARISON_PLAN.md) for inherited settings,
cache integration, selection, test reuse and team schedule. The minimum is one
training-only augmentation experiment plus checked explanations of the frozen
baseline. Extend explanations to the peer models after their checkpoints arrive.

## Augmentation: one interpretable comparison

Question: **Under the same training recipe, how does mild training-only
augmentation affect U-Net++ RD sensitivity, specificity and average precision?**

Compare existing `unetpp_rd_002_amp` without augmentation against a new
`unetpp_rd_aug_001_amp` trained **from scratch**, with the same architecture,
input tensors, seed, split membership, optimizer, loss, epoch budget and
checkpoint-selection rule. Do not fine-tune the existing epoch-48 checkpoint
and describe that as an augmentation-only comparison. The new run needs its
own config, source identity, resumable state and output directory.

The initial candidate policy below is a conservative engineering proposal.
These ranges are **not** established ultrasound-specific invariances. Inspect
training examples before freezing the policy and record the actual chosen values.

| Transform | Proposed range | Probability per training clip |
| --- | --- | ---: |
| Small spatial affine transform | Rotation within ±7 degrees, translation within ±3% of image width/height, isotropic scale 0.95–1.05 | 0.5 |
| Multiplicative intensity gain | Factor 0.9–1.1, clamp result to `[0,1]` | 0.5 |

Sample each transform **once per clip** and use the same spatial transform and
gain for all 96 temporal positions. Independent frame rotations/gains would
introduce artificial motion/flicker. Preserve temporal order and input shape.
For affine resampling, use bilinear spatial interpolation, zero padding and an
explicit `align_corners=False`; record transform order (affine, then gain).
Torch's `affine_grid`/`grid_sample` can apply a shared 2D grid across frames,
without introducing an augmentation dependency solely for this experiment.
[Official grid_sample documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

Keep vertical/horizontal flips, temporal reversal, heavy cropping, elastic
distortion, MixUp/CutMix and strong noise outside this initial policy. Their
label-preservation and effect on relevant anatomy need separate justification.
Small affine changes can still clip tissue at borders: reject a proposed policy
if training-only preview panels show implausible changes or lost anatomy, then
document the revised ranges **before** the final run.

### Implementation and evidence

1. Add a training dataset wrapper or transform in
   `src/rd_ultrasound_ml/video_augmentation.py`. Apply online transforms to a
   writable tensor copy after loading the deterministic cache. Keep prepared
   `.npy` files and validation/test tensors unchanged. The shared protocol
   describes how to separate preparation identity from new training identity.
2. Save a preview panel of at least eight predetermined training clips spanning
   both labels, each showing original and augmented versions at several temporal
   positions. Record clip IDs, policy, seed and sampled parameters. Verify shape,
   finite `[0,1]` range, unchanged labels, shared temporal transform parameters,
   cache immutability and deterministic replay for a fixed RNG state.
3. Document how augmentation RNG resumes. The existing engine saves Python,
   NumPy, torch and CUDA RNG states; ensure transforms actually use those states
   with the current `num_workers=0`, or derive randomness reproducibly from seed,
   epoch and clip identity. Check a synthetic interruption/resume across an
   optimizer boundary including the augmentation draws. Do not promise bitwise
   reproducibility on CUDA from a seed alone; record its relevant settings.
4. Freeze `configs/unetpp_rd_aug_001_training.json` with the resolved policy and
   start fresh training after its resource check. Preserve all epoch-level metrics
   and predictions. Validation and final inference always use clean inputs.
5. Compare selected validation results and eventually frozen reused-test results
   with the corresponding reference partition. Report change in sensitivity,
   specificity, AP and FN/FP counts, as well as accuracy. Retain an adverse or
   inconclusive result; one seed does not establish a reliable augmentation gain.

Deliver `docs/UNETPP_AUGMENTATION_RESULTS.md`, original/augmented previews,
learning curves, the resolved config, recomputable validation predictions and
private checkpoint/artifact references. Report exact epochs if incomplete.
Peers train their initial models without this policy; applying augmentation to
all models would require additional matched runs and is a stretch goal.

## Explainability: explain the clip score

The output is a binary video score. A Grad-CAM overlay is a coarse attribution
to that score; it is **not a retinal-detachment mask, anatomical annotation or
clinical explanation**. No segmentation localization score is justified without
verified ground-truth masks. Ground-truth clip labels, predicted class and heatmap
interpretation must be labeled separately in every figure.

### First implementation: 3D Grad-CAM

Use the already selected epoch-48 U-Net++ checkpoint first; explanations do not
require another training run. Suggested module:
`src/rd_ultrasound_ml/video_explainability.py`; suggested notebook:
`notebooks/video_explainability.ipynb`. Implement against a model/layer interface
so peer checkpoints can later use it. Source basis:
[Grad-CAM, Section 3](https://arxiv.org/html/1610.02391v4#S3) and
[local source notes](research/video-model-comparison-sources.md).

1. Verify checkpoint hash/config, instantiate its exact architecture and load its
   weights. Keep a clean model in `eval()` so dropout and normalization use
   evaluation behavior. Run the attribution forward pass with autograd enabled,
   outside both `torch.no_grad()` and `torch.inference_mode()`. Use fresh tensor
   copies, with `requires_grad=True` if necessary when model weights are frozen.
   No optimizer update occurs. Start with batch one; probe memory and record
   precision. FP32 is a reasonable first explanation setting if it fits; verify
   predictions against the saved inference configuration within a recorded
   tolerance. [PyTorch autograd guidance](https://docs.pytorch.org/docs/stable/notes/autograd.html).
2. Capture the final convolutional feature tensor before global pooling:
   U-Net++ `model.encoder[-1]`; peer VideoResNet `layer4` through its actual
   wrapper path. Record `[C,D',H',W']` activation shape. A shallower layer may
   offer finer maps, but choose layers using training/validation development,
   documenting changes before any new test explanation analysis.
3. Target the **single raw RD logit** `z`, not output index 1 (there is only one
   output). For each channel `k`, compute
   `alpha[k] = mean(dz/dA[k], over D',H',W')`, then
   `CAM = relu(sum_k(alpha[k] * A[k]))`.
   Document this as our volumetric adaptation of Grad-CAM. An explanation of
   non-RD evidence can target `-z`; give it a distinct label.
4. Interpolate the coarse volume to `[96,128,128]` with trilinear interpolation
   and `align_corners=False`. Normalize across the **whole clip**, recording the
   rule; avoid separate normalization that makes every frame look equally
   salient. Handle a zero/constant CAM explicitly as no positive attribution.
   Resizing a heatmap does not recover missing spatial or temporal resolution.
5. Show original grayscale frames alongside overlays with the dataset label,
   predicted class, RD score, target logit sign, checkpoint and chosen layer.
   Positions are resampled temporal indices, not original frame timestamps.
   The score is a model output, not calibrated clinical risk.
6. Remove hooks and release graphs after each clip. Prefer gradients of captured
   activations over fragile module backward hooks around in-place activations.
   Verify explanation runs leave model parameters and normalization buffers
   unchanged and that a subsequent ordinary prediction is unchanged within the
   recorded tolerance.

### Fixed examples and sanity checks

Preselect validation examples using a reproducible rule: from the frozen
baseline's validation predictions, choose up to two clips in each of TP, TN, FP
and FN, ordered by a fixed seed or clip-ID rule. Record the rule and IDs before
looking at heatmaps. Use the same clips for every model, retaining each model's
own outcome label; include all selected examples, even visually unhelpful maps.
This is an outcome-stratified case panel, not a representative performance sample.

Include at least these checks:

- **Weight dependence:** run the same inputs through separately copied,
  progressively randomized model layers, including the encoder. Save the
  heatmap and prediction changes, and flag maps that remain suspiciously similar.
  This diagnoses possible implementation/method limitations; it does not certify
  attribution correctness. Preserve the original checkpoint/model state.
  Full data-label randomization requires retraining and can be reported as
  outside this week's scope. [Sanity Checks for Saliency Maps](https://arxiv.org/html/1810.03292v3).
- **Occlusion:** on the fixed validation panel, define equal-sized cuboids in
  space/time, then compare the RD-logit change after masking high-attribution
  versus low-attribution/random regions. Freeze cuboid dimensions, count, seed
  and fill value before comparing. Use the same settings on each model and report
  outcomes, including failures of the expected pattern. Zero-fill occlusion is
  an artificial input intervention, so this is a sensitivity check rather than
  causal or anatomical validation. Do not select masks by which causes the
  biggest score drop after seeing results.
- **Implementation behavior:** finite gradients/maps, explicit zero-map handling,
  correct positive/negative logit targeting, restored hooks and unchanged model
  state; identical inputs/settings should reproduce outputs within the declared
  tolerance.

The minimum presentation panel can be a few matched frames per clip. A short
overlay animation and a mean-attribution-over-time curve are useful exports if
time permits. Avoid SHAP/LIME across every pixel and frame as the first-week
requirement; a verified Grad-CAM path and an occlusion check provide a bounded
deliverable. Explanations should describe model sensitivity, not claim that the
model has learned correct clinical reasoning.

Deliver `docs/EXPLAINABILITY_RESULTS.md` with chosen layer/target/normalization,
checkpoint hash, example-selection manifest, original/overlay panels, control
results and limitations. Store per-clip arrays/videos in ignored outputs. Main
explanation development uses validation. If existing test errors later appear
in slides, label that analysis post hoc and keep it out of experiment selection.

## Presentation evidence to aim for

Use two connected experiment questions: model architecture under a shared recipe,
and augmentation on the fixed U-Net++ architecture. Show separate validation and
reused-test comparison tables, matched confusion matrices/ROC/PR curves, one
augmentation preview, and original-versus-attribution success/failure panels.
Retain actual epochs, timing and memory so progress is visible even if a model
does not improve. Conclude with the known clip-level split, exposed benchmark,
single-seed and attribution limitations without making clinical-validity claims.

## Paste into your implementation assistant

```text
Read AGENTS.md, docs/MODEL_COMPARISON_PLAN.md and
docs/AUGMENTATION_EXPLAINABILITY_PLAN.md. Implement my portion in small checked
steps: training-only temporally consistent augmentation with a separate U-Net++
from-scratch run, and 3D Grad-CAM plus sanity/occlusion checks using the frozen
baseline. Inspect existing data/training/resume/evaluation code first and state
a short plan. Preserve original cache arrays, splits, frozen configs and results.
Implement and inspect training-only augmentation previews before freezing the
policy. Keep augmentation RNG reproducible across checkpoint resume. Make the
explanation interface accept the later R3D-18 and R(2+1)D-18 checkpoints too.
Verify finite maps/gradients, scalar RD-logit targeting and unchanged model state.
Use a recorded fixed validation case panel, not handpicked attractive maps.
Train only when the intended GPU/data access and measured budget allow it;
complete independent code and checks if access is missing. Export auditable
artifacts, report commands/checks actually run, preserve negative findings, and
keep final experiment choices independent of any new official-test scores.
```
