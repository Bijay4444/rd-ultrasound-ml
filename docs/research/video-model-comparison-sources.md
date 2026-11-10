# Sources for the next video-classification comparison

Checked 2026-11-10. This is a research recommendation, not a measured training
result or a claim that either model will outperform the completed baseline.
No training, installation, or GPU benchmark was performed for this note.

## Starting point and recommendation

The completed local “UNet++” is a 3D encoder classifier with a scalar RD logit,
not a pixel-mask segmentation experiment. Its recorded input is
`[B,1,96,128,128]`, with 3,873 training and 431 validation clips, no augmentation,
and training from scratch. Preserve that completed reference. See
[the local results](../UNETPP_RESULTS.md) and
[frozen training configuration](../../configs/unetpp_rd_002_training.json).

Recommended peer assignments are **R3D-18** and **R(2+1)D-18**, subject to a
real-input forward/backward timing and memory probe before committing the week.
They offer a controlled contrast between joint spatiotemporal convolution and
factorized spatial/temporal convolution within residual networks. Factorization
adds an intervening nonlinearity; the original paper's action-recognition results
do not predict performance on ERDES. This pair is a feasibility/design judgment,
not an experimentally established ranking. [Tran et al., CVPR 2018](https://arxiv.org/abs/1711.11248)

| Candidate | Role | Official stock parameter count | Stock reported GFLOPS |
| --- | --- | ---: | ---: |
| [R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html) | Peer 2; joint 3D residual baseline | 33,371,472 | 40.70 |
| [R(2+1)D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html) | Peer 1; factorized residual baseline | 31,505,325 | 40.52 |
| [MC3-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mc3_18.html) | Optional mixed-convolution alternative | 11,695,440 | 43.34 |

These are official stock RGB/400-class metadata, **not measurements at our
96-frame grayscale shape**; the classification adaptation changes parameter
counts. The published weight evaluations use 16-frame clips. Smaller parameter
count does not establish faster training or lower activation memory: MC3's
reported operations are slightly higher than the other two. R(2+1)D's intermediate
feature maps also make a real backward-pass memory probe essential. The existing
3.6M-parameter classifier's runtime cannot be projected onto these models.
[R3D documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html),
[R(2+1)D documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html),
[MC3 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.mc3_18.html)

If the pair fails the measured time/memory budget, a stronger deadline fallback
is a **2D ResNet-18 frame encoder with mean-pooled frame features and one clip
logit**. Use a documented fixed, uniformly spaced frame subset (for example, 16
of the cached 96) and train with clip labels. This changes temporal sampling and
removes learned temporal convolution, so report it as a separate protocol. The
official ResNet-18 is an available small image backbone; the pooling/sampling
design here is our proposed adaptation. MC3 is an architectural alternative,
not an assured compute rescue. [Official ResNet-18 builder](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## Exact torchvision adaptation

The inspected implementation uses `stem[0]` for the first convolution in all
three video models. Replace only its input-channel count with 1:

| Builder, with `weights=None, num_classes=1` | Replacement `stem[0]` |
| --- | --- |
| `torchvision.models.video.r3d_18` / `mc3_18` | `Conv3d(1,64,(3,7,7),stride=(1,2,2),padding=(1,3,3),bias=False)` |
| `torchvision.models.video.r2plus1d_18` | `Conv3d(1,45,(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=False)` |

Keep the remaining stem. The head is `fc`; `num_classes=1` constructs
`Linear(512,1)` after global 3D average pooling. Initialize the replacement
convolution consistently with the builder's Kaiming-normal rule. R3D uses 3D
convolutions throughout; R(2+1)D separates spatial and temporal operators; MC3
uses temporal convolutions early and spatial-only convolutions in later stages,
retaining temporal resolution there. All use BatchNorm3d. These details are
version-sensitive: record and verify the installed torchvision version rather
than assuming current stable documentation matches the saved Kaggle runtime.
[Official video ResNet source](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/video/resnet.html)

For the first comparison, retain cached grayscale `[0,1]` inputs and explicit
`weights=None`. Do not apply the stock Kinetics RGB transforms, their three-channel
normalization, crop, or resize on top of the cache. Those are preprocessing
prescriptions associated with pretrained weights, while the builders support
random initialization. Pretraining would need a separate documented adaptation.
[Official builder and weight-transform documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html)

BatchNorm computes training statistics on each physical forward batch over
`N,D,H,W`, with running statistics used at evaluation. Gradient accumulation
therefore does **not** turn physical batch 1–4 into a BatchNorm batch of 48.
Record physical batch, accumulation, and normalization; inspect finite training
and validation behavior. A normalization replacement is an additional model
adaptation, not an invisible fix. [PyTorch BatchNorm3d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm3d.html)

## Relationship to ERDES benchmarks

ERDES provides clip-level labels, not per-frame annotations, and acknowledges
possible patient overlap in its clip-level split. Its Table 8 reports a **3D
ResNet** Non-RD/RD row: sensitivity 0.939, specificity 0.978, precision 0.817,
F1 0.874, accuracy 0.974. R(2+1)D-18 and MC3-18 are not named benchmark rows.
The paper's ResNet uses a different cited implementation family; do not call a
torchvision R3D-18 run an exact reproduction without a separate source/config
audit. Preserve local split membership and validation-only selection. The
already exposed official test is a reused benchmark, not a new untouched
holdout. [ERDES v2, labeling, split limitations and Table 8](https://arxiv.org/html/2508.04735v2)

## Explainability source checks

Grad-CAM uses gradients of a class score before the output nonlinearity and
forms a coarse positive-evidence activation map. For this binary classifier,
target the **scalar RD logit** `z`, not a nonexistent output index 1. Targeting
`-z` provides a separately labeled non-RD explanation. A proposed 3D extension
averages gradients over `D,H,W`, weights the selected `[C,D,H,W]` feature maps,
applies ReLU, and interpolates the volume to input size. Document the chosen
layer and extension; interpolated resolution is not localization accuracy.
[Original Grad-CAM, Section 3](https://arxiv.org/html/1610.02391v4#S3)

Use `model.eval()` **with autograd enabled**. `eval()` changes normalization and
dropout behavior but does not disable gradients; `no_grad()` and
`inference_mode()` prevent the recorded graph required for this explanation.
Ensure the chosen activation has gradients (or set input `requires_grad=True`
when parameters are frozen), use fresh inputs outside inference mode, and make
no optimizer update. [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html)

For the week, preselect a small validation TP/TN/FP/FN panel, show original
frames beside overlays, and include a copied-model parameter-randomization
check. The saliency sanity-check paper warns that visual appeal alone is
insufficient and proposes model-parameter and data-label randomization tests;
record if the latter retraining test is outside scope. Heatmaps are model
attributions, not segmentation ground truth or evidence of clinical validity.
[Adebayo et al., Sanity Checks for Saliency Maps](https://arxiv.org/html/1810.03292v3)

Unresolved before implementation: actual GPU availability, per-model physical
batch, epoch time, peak training memory, convergence under the shared starting
recipe, and whether any fallback is necessary. None can be established from
parameter counts or paper metrics alone.
