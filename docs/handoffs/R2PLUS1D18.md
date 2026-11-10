# Teammate 1: R(2+1)D-18 RD video classification

Status: assignment and implementation brief, 2026-11-10; no new model implemented
or trained by this document. Read the [shared protocol](../MODEL_COMPARISON_PLAN.md)
first; it owns data, optimization, cache integration, evaluation and reporting rules.

## Why this assignment

Bijay requested the more involved model for the first teammate. R(2+1)D-18 has
a factorized spatial/temporal stem and intermediate activations that require
careful adaptation and resource checks. R3D-18 has a simpler joint-convolution
stem. This makes R(2+1)D-18 the more involved implementation in our judgment;
actual relative training time, memory and convergence remain unmeasured.
Both have ready-made torchvision builders, so neither needs building from scratch.

## Goal

Train a one-channel torchvision **R(2+1)D-18** classifier from scratch on ERDES
`non_rd_vs_rd`. Compared with teammate 2's R3D-18, this architecture factorizes
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
   same physical batch with teammate 2 before starting the final run. Factorized
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

## What Bijay should share

Send the repository URL, this handoff and the Claude prompt below. The teammate
must have the repository files, not just the prompt:

- Repository branch: <https://github.com/Bijay4444/rd-ultrasound-ml/tree/dev>.
  Start from `dev`, not the older default `main` branch. Fresh checkout:

  ```bash
  git clone --branch dev https://github.com/Bijay4444/rd-ultrasound-ml.git
  cd rd-ultrasound-ml
  git switch -c feature/r2plus1d18-rd
  ```

- Required context: this file, `AGENTS.md`, `docs/MODEL_COMPARISON_PLAN.md`,
  `docs/UNETPP_RESULTS.md`, `docs/research/video-model-comparison-sources.md`,
  `configs/unetpp_rd_002_training.json` and `docs/KAGGLE_GUIDE.md`.
- Implementation context: the repository's `src/`, `scripts/`, `tests/` and
  `notebooks/` directories. Claude should inspect these in the checkout.
- Data access: the two exact private Kaggle preparation-output source/version
  identifiers and appropriate access, if sharing existing inputs. Otherwise
  follow `docs/DATA_ACCESS.md` to prepare the identical release independently.
  Local documentation alone does not grant access to private Kaggle outputs.

The [dev readiness audit](DEV_READINESS_AUDIT.md) distinguishes verified repository
contents from external requirements and the model-specific files you will create.

Share the updated files as attachments or make them available through the team's
normal Git workflow before asking him to read them. This handoff update does not
itself commit or push anything. Use each person's own credentials; exchange no
API tokens. The first useful check-in is a passed model/input check plus a real
GPU memory probe and one full-epoch timing, preferably within the first day.

## WhatsApp message to copy

```text
Hey, please take R(2+1)D-18 for our final milestone model comparison. It is the
more involved of the two implementations; the other teammate will take R3D-18.
I'll handle augmentation and explainability.

Repo: https://github.com/Bijay4444/rd-ultrasound-ml/tree/dev
Please start from the dev branch; main is older.

I'm sending the model handoff and a Claude prompt. Open the repo in Claude Code
and give it the prompt. Keep the same ERDES splits and video preprocessing as
our U-Net++ run, and train your first model without augmentation so we can compare.

Please aim to send me the GPU memory check and one full-epoch timing within a
day, then continue toward 50 epochs if the budget fits. For the final handoff,
send the notebook/code, config, best checkpoint, training curves, validation
predictions and metrics, plus runtime details. Keep test evaluation separate
until we've frozen the experiments together. I'll coordinate the prepared-data
access. Send me any access or compute blockers early.
```

## Claude implementation prompt to copy

Paste this into Claude Code with the repository open at its root. In a chat-only
session, attach the required context and relevant code files; a chat without
repository or execution access cannot perform training itself.

```text
I am teammate 1 on the CSCI 555 ERDES retinal-detachment video-classification
project. Implement and, when GPU/data access permits, train our R(2+1)D-18
comparison model. We have roughly one week. Bijay owns augmentation and
explainability; another teammate owns R3D-18.

Start from the repository's dev branch, then create feature/r2plus1d18-rd.
If the checkout has local changes, preserve them before switching branches.

First read AGENTS.md, docs/MODEL_COMPARISON_PLAN.md,
docs/handoffs/R2PLUS1D18.md, docs/UNETPP_RESULTS.md,
docs/research/video-model-comparison-sources.md, and
configs/unetpp_rd_002_training.json. Inspect existing source, tests, Kaggle
notebooks and submission scripts before making changes. Propose a short plan,
then implement in small checkable steps on feature/r2plus1d18-rd.

Our completed U-Net++ is a 3D encoder video classifier, not segmentation. Use
torchvision.models.video.r2plus1d_18(weights=None, num_classes=1), changing only
the first stem convolution's input channels from 3 to 1 and initializing that
replacement consistently with the builder. Preserve its later temporal stem.
Input is [B,1,96,128,128], grayscale in [0,1]; output one raw RD logit per clip.
Use original official splits: 3,873 training and 431 validation clips. RD=1.

For the first comparison inherit the recorded recipe: seed 42, 50 epochs,
Adam at 1.5e-5, unweighted BCEWithLogitsLoss, ReduceLROnPlateau and effective
batch 48. No pretrained weights, augmentation, class resampling or weighting.
Retain threshold 0.5 and highest-validation-accuracy checkpoint selection,
earliest ties. Follow the shared protocol for full optimizer/scheduler defaults.

Reuse existing training code where appropriate. The cache loader currently
checks the entire U-Net++ config digest: handle preparation identity separately
from the new training identity while retaining source/tensor hash, label,
partition/order and preprocessing checks. Keep original data, source snapshots,
configs, notebooks and result artifacts intact. Use new config/output names.
Use uv locally; record the GPU runtime including torch and torchvision versions.

Verify model shape, finite gradients, cache validation and checkpoint reload.
Run a disposable real-training-input GPU memory/precision probe, coordinate
physical batch with the R3D-18 owner, and time one full epoch before committing
to the full run. Discard probe weights and start final training from the fixed
seed. Record BatchNorm/physical-batch limitations and preserve complete resume
state. If access is missing, finish independent code/checks and report exactly
what is missing. Never invent execution results or silently shorten the recipe.

Return model-specific code, config, standalone training notebook/submission
workflow, tests/checks actually run, docs/R2PLUS1D18_RESULTS.md, per-epoch curves,
selected-checkpoint identity, per-clip validation predictions and metrics,
parameter count, runtime/memory measurements, and private artifact locations.
Expose the exact final feature-layer path for Bijay's Grad-CAM integration.
Report actual epochs, failures and deviations. Keep new test inference separate
until the team freezes all final choices; the official test was already exposed
and any later evaluation must be disclosed as benchmark reuse.
```

## Reference links to send

1. [Torchvision R(2+1)D-18 builder](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html): supported constructor and pretrained-weight distinction.
2. [Official VideoResNet source](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/video/resnet.html): factorized stem, blocks, initialization and classifier head.
3. [Tran et al., A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248): model rationale; results on other datasets are not ERDES guarantees.
4. [ERDES paper v2](https://arxiv.org/html/2508.04735v2): dataset/task and benchmark context; this new model is not an exact reproduction of a named Table 8 row.
