# ERDES UNet++ protocol and source audit

Checked 2026-10-30. This is a source audit and proposed protocol, not a completed
reproduction. Target **one model, UNet++ on `non_rd_vs_rd`**, subject to a measured
Colab memory/runtime preflight. The completed custom 17,585-parameter, 16-frame,
256-training-clip engineering pilot and its weighted comparison are separate
experiments; neither reproduces this benchmark.

The research sequence was to inspect project instructions/existing notes, compare
the versioned paper with executable source/configuration, count supplied split
rows, and record a checkable single-model protocol. No upstream files were edited.
The prescribed `docs/course-guidance/` directory is absent; the existing guide is
under `docs/course-guideline/`. Previously inspected guide themes are workflow,
verification, reproducibility, and leakage; the course grading rubric is not
available. Do not infer additional assessment requirements from the paper.

## Source identities

- Paper: Ozkut et al., [arXiv:2508.04735v2](https://arxiv.org/html/2508.04735v2).
- Read-only authors' checkout: `/home/bijay/first_sem_proj/dm_ml/ERDES`, revision
  `4af67e0b9f4667edbae75a64a819884ee14d7a3b`. All ERDES GitHub links below pin that revision.
- Dataset provenance and patient-grouping limitations are already recorded in
  [the access note](erdes-grouping-and-access.md). Preserve official clip splits;
  do not describe them as unseen-patient evaluation.
- The authors distribute [benchmark checkpoints](https://zenodo.org/records/18821031)
  and [a ResNet model card](https://huggingface.co/pcvlab/resnet3d_non_rd_vs_rd).
  Loading an author-trained checkpoint can reproduce an evaluation, but does not
  reproduce training from scratch. No checkpoint was downloaded for this audit.

## Paper claims and the comparison target

The paper specifies scratch initialization, AdamW, initial learning rate
`1.5e-5`, 50 epochs, unweighted BCE with logits, and no resampling. Training used
three 48-GB A6000s, with total batches typically 12–96. Table 6 specifies grayscale,
square padding, trilinear resizing to `96×128×128`, and division by 255.
[Technical Validation](https://arxiv.org/html/2508.04735v2#S4).

The **Non-RD vs. RD test columns of Table 8**, in sensitivity / specificity /
precision / F1 / accuracy order, report:

| Backbone | Sensitivity | Specificity | Precision | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT | 0.689 | 0.969 | 0.696 | 0.693 | 0.943 |
| UNETR | 0.569 | 0.972 | 0.678 | 0.619 | 0.935 |
| SENet154 | 0.689 | 0.992 | 0.907 | 0.784 | 0.964 |
| Swin-UNETR | 0.680 | 0.980 | 0.781 | 0.727 | 0.952 |
| V-Net | 0.829 | 0.994 | 0.943 | 0.882 | 0.979 |
| 3D ResNet | 0.939 | 0.978 | 0.817 | 0.874 | 0.974 |
| **UNet++ — selected target** | **0.879** | **0.986** | **0.871** | **0.875** | **0.976** |
| 3D U-Net | 0.920 | 0.988 | 0.893 | 0.906 | 0.982 |

Source: [Table 8](https://arxiv.org/html/2508.04735v2#S4.T8). These are author
test results, not our validation results or promised reproduction outcomes.
Normal-vs-RD and PVD-vs-RD occupy different columns; do not mix them.

## Exact input pipeline recovered from code

Use the released **already cropped** clips. The training dataset class contains
no ROI detector. Its complete processing path is:

1. Decode the full video with `torchvision.io.read_video(..., pts_unit='sec')`.
2. Convert uint8 pixel values to float and reorder `[T,H,W,C]` to `[C,T,H,W]`.
3. For three-channel input, take the arithmetic channel mean, retaining one channel.
4. Pad the shorter spatial axis with constant zero: floor-half on one side,
   remainder on the other. Time is not padded separately.
5. Add a temporary batch dimension and use `torch.nn.functional.interpolate`
   in `trilinear` mode to **jointly resize time and space** to `(96,128,128)`.
   The call omits `align_corners`; record the selected PyTorch default explicitly
   in our implementation/configuration rather than assuming unspecified libraries
   behave identically.
6. Divide by 255. Output per clip is float `[1,96,128,128]`, then batch to
   `[B,1,96,128,128]`.

Evidence: [dataset lines 31–59][dataset], [padding/interpolation lines 9–66][resize],
[data configuration][data-config]. This is not uniform frame-index sampling,
16 selected frames, an OpenCV area resize, ImageNet normalization, FPS
standardization, or a fixed-duration crop. The same dataset class is used for
train/validation/test; no random augmentation is present in this path.
[Data module lines 42–84][datamodule].

For release generation, the paper describes confidence-based ROI selection,
whereas the later model-card example chooses the largest detected box. Do not
re-run that example on the released cropped clips.
[Paper preprocessing](https://arxiv.org/html/2508.04735v2#S2),
[author inference example](https://huggingface.co/pcvlab/resnet3d_non_rd_vs_rd#usage).

## Preserve the supplied partition and labels

The source task definition maps label 0 to Normal plus PVD; label 1 combines all
RD macula/anatomical subdirectories. Counts below were independently recomputed
with Python's `csv.DictReader` and `collections.Counter` from the pinned checkout.
Preserve row membership and order; record CSV SHA-256 checksums when preparing
the run. [Task mapping lines 30–41][splits-code], [official CSV directory][splits].

| Partition | Label 0: Non-RD | Label 1: RD | Total |
| --- | ---: | ---: | ---: |
| Train | 3,511 | 362 | 3,873 |
| Validation | 391 | 40 | 431 |
| Test | 977 | 100 | 1,077 |

All 5,381 clips belong to this task. The split generator uses stratification,
seed 42, a 20% test split, then 10% of the remainder for validation, producing
72/8/20 proportions. It uses unsorted filesystem glob input, so regenerating
the split from the same seed is not proof of identical membership. Use the
released CSVs. [Split generation lines 69–100][splits-code].

Other supplied task totals, for orientation only, were also checked directly:
Normal/PVD `3512/391/976`, Normal/RD `3413/377/945`, PVD/RD `826/92/230`, and
macula detached/intact `360/41/101` (train/validation/test). They are not the
selected reproduction target. Macula-intact, rather than detached, is class 1
in that task. [Task definitions][splits-code], [CSV directories][all-splits].

## What “UNet++” means in this benchmark

Use the authors' classification architecture, not an arbitrary MONAI segmentation
UNet++ with a newly attached head. The factory passes one input channel, one
output logit, average pooling, and the following classifier defaults:

| Component | Executed setting recovered from source |
| --- | --- |
| Encoder | Five sequential blocks: `conv_0_0`, then `conv_1_0` through `conv_4_0` |
| Widths | `(32,32,64,128,256)` along the down path; configuration tuple `(32,32,64,128,256,32)` has an unused final decoder width |
| Blocks | MONAI `TwoConv`, then four MONAI `Down` blocks; 3D operations |
| Convolutions / pooling | Two 3×3×3 stride-1, padding-1 convolutions per block; each down block begins with 2×2×2 stride-2 max pooling |
| Activation | LeakyReLU, negative slope 0.1, inplace |
| Normalization | Instance normalization, affine parameters enabled |
| Observed normalization defaults | `eps=1e-5`, `momentum=0.1`, `track_running_stats=False` under MONAI 1.5.2 |
| Convolution bias | Enabled |
| Dropout | 0.0 |
| Features used | Only deepest 256-channel features; no nested decoder or segmentation output |
| Head | Adaptive global average pool over all 3 axes, flatten, Linear(256,128), ReLU, Linear(128,1) |
| Output | One raw logit; sigmoid is applied by the training/evaluation module |
| Initialization | No checkpoint/pretraining load in factory/classifier; constructor defaults initialize weights |
| Observed trainable parameter count | **3,603,553** under local MONAI 1.5.2 |

Evidence: [factory lines 76–85][factory], [classifier lines 188–224][classifier],
[encoder lines 24–94][unetpp-encoder], [head lines 15–69][classifier]. The code
does not call a custom weight initialization routine for this classifier. The
exact initialized tensors therefore also depend on PyTorch/MONAI versions and
the seed; do not substitute pretrained weights or a different initializer.
The parameter count and module settings above were checked by constructing the
model on CPU with `PYTHONDONTWRITEBYTECODE=1` and the existing interpreter
`/home/bijay/first_sem_proj/dm_ml/ERDES/.venv/bin/python`; no forward pass or
training was run. MONAI's [TwoConv/Down implementation](https://github.com/Project-MONAI/MONAI/blob/1.5.2/monai/networks/nets/basic_unet.py#L26-L91)
defines the convolution and pooling defaults. The checkpoint's state dict was
not inspected; its inventory size alone does not prove matching tensor shapes.
In the inspected PyTorch **2.5.1+cpu** environment, Conv3d and Linear reset
weights with `kaiming_uniform_(a=sqrt(5))` and biases uniformly within
`±1/sqrt(fan_in)`; affine instance-normalization starts with scale 1 and bias 0.
These are library constructor defaults, not a new initializer to add on top.
[PyTorch convolution initialization](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/conv.py#L178-L188),
[linear initialization](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/linear.py#L114-L123),
[normalization initialization](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/batchnorm.py#L93-L97).

For other names, the factory/code resolve to: ResNet basic blocks with
`layers=[4,4,4,4]`, widths `[64,128,256,512]`; MONAI SENet154; 3D U-Net encoder
widths `[64,128,256,512,768]`; V-Net down path 16→32→64→128→256; Swin-UNETR
feature size 24 with depths `(2,2,2,2)` and heads `(3,6,12,24)`; UNETR hidden
size 768, MLP 3072, 12 attention heads and 12 transformer layers; and a distinct
ViT using patch size 7, hidden 768, MLP 3072, 4 layers/4 heads, and a CLS-token
linear classifier. These details explain why generic architecture names alone
cannot specify a reproduction. [Factory][factory], [classifier][classifier],
[UNETR encoder][unetr-encoder]. Only UNet++ is proposed for implementation now.

## Compose the experiment overrides, not the template defaults

| Setting | Pinned upstream UNet++ `non_rd_vs_rd` experiment |
| --- | --- |
| Seed | 42; training invokes `seed_everything(seed, workers=True)` before instantiation |
| Epochs | 50 maximum, minimum 0 |
| Initial learning rate | `1.5e-5`, overriding model YAML's `1.5e-4` |
| Optimizer in code | `torch.optim.Adam`, **not AdamW**; `weight_decay=0.0` |
| Other optimizer arguments | Not explicitly set; inherited from installed PyTorch |
| Scheduler | ReduceLROnPlateau, `mode=min`, factor 0.1, patience 10; other arguments inherit library defaults |
| Scheduler observation | `val/loss`, once per epoch |
| Training/validation loss | `BCEWithLogitsLoss()`; no `weight` or `pos_weight` |
| Resampling | No weighted sampler; training shuffle enabled |
| Configured total batch | 48; data module divides it by world size, so 16 per GPU for 3 GPUs |
| Checkpoint selection | Highest validation accuracy; save top 1 plus last |
| Validation cadence | Each training epoch |
| Early stopping | Still inherited from default callbacks with patience 100 on validation accuracy; commented experiment block does not remove it |
| Gradient clipping | 0.0 |
| Compile | False |
| Precision | No explicit mixed-precision override in experiment/default trainer |
| Determinism | Default trainer sets `deterministic=False`; a seed does not promise bitwise determinism |
| Loader | Workers 0, pinned memory enabled; shuffle train only; no explicit `drop_last` |

Evidence: [experiment lines 6–55][experiment], [model YAML][model-config],
[training seed and fit/test logic][train], [optimizer/scheduler wiring lines 294–305][module],
[loss/metrics lines 64–94][module], [data module][datamodule],
[default callbacks][callbacks], [default trainer][trainer].

Observed PyTorch 2.5.1 defaults, which should be made explicit for a portable
code-faithful attempt: Adam `betas=(0.9,0.999)`, `eps=1e-8`, `amsgrad=False`;
ReduceLROnPlateau `threshold=1e-4`, `threshold_mode=rel`, `cooldown=0`,
`min_lr=0`, `eps=1e-8`. The paper does not authenticate this library version.
These signatures were inspected read-only rather than inferred from a tutorial.
[Adam implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adam.py#L33-L48),
[scheduler implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/lr_scheduler.py#L1234-L1250).

The bare training configuration instead defaults to UNet++, batch 4, CPU, 10
epochs, and no seed. Its `test=True` automatically evaluates after training,
using the selected checkpoint. The authors' launch script overrides the trainer
to DDP with three devices; the DDP YAML enables synchronized batch normalization.
For this instance-normalized UNet++ architecture, do not add batch normalization
to imitate that flag. [Base configuration][train-config], [launch script][launch],
[DDP YAML][ddp].

In our development/preflight runs, explicitly disable automatic test execution.
Keep the official test set closed until the protocol/model/checkpoint are frozen.
This is project experimental discipline; it does not change checkpoint selection
from validation accuracy to validation AP.

## Metric implementation and unresolved reproducibility details

The module thresholds `sigmoid(logit) >= 0.5`, then updates binary TorchMetrics
accuracy, precision, recall/sensitivity, specificity, and F1. Positive means RD
for this task. Save one prediction per official test clip plus a global confusion
matrix, denominator counts, threshold, selected epoch and checkpoint hash.
AP, ROC-AUC and balanced accuracy can be added as clearly marked supplementary
metrics; they are not Table 8's five columns. [Metrics and threshold][module].

Resolve or explicitly disclose these gaps before claiming an exact reproduction:

1. **Optimizer conflict:** the paper says AdamW; pinned executable YAML says Adam.
   The experiment only overrides learning rate. Run provenance/checkpoint metadata
   or author clarification is needed to know which generated Table 8. A single
   honest first attempt can follow pinned code (Adam, zero decay) and label this
   discrepancy. Do not silently choose AdamW's default weight decay.
2. **Reported denominator inconsistency:** on the supplied 100-positive test set,
   a single global recall must be an integer multiple of 0.01. Reported UNet++
   0.879 (and ResNet 0.939) cannot result from that denominator rounded to three
   decimals. The cause is unresolved. Do not infer another split or quietly
   adjust our metric to match. Distributed evaluation details, predictions or
   original run artifacts would be needed to reconcile it.
3. **Exact software environment:** broad dependency ranges in
   [requirements.txt][requirements] do not identify the training versions.
   The local installed MONAI is 1.5.2, an observation about this checkout's
   environment, not evidence that the authors used it for published runs.
   Record all resolved versions and constructor defaults in our run.
4. **Unstated run controls:** the paper does not settle a scheduler, checkpoint
   criterion, training seed, precision, per-model batch, augmentation policy,
   exact initializer or repeated-run aggregation. The values above come from
   pinned source, not an authenticated resolved config of the published run.
   No original run logs were found under the local `logs/` tree.
5. **Training-resource equivalence:** using one GPU, gradient accumulation or AMP
   needs explicit disclosure. For BatchNorm architectures, accumulation cannot
   reproduce larger synchronized-batch statistics. UNet++ uses instance norm,
   but floating-point/order differences still prevent an identity guarantee.

## Why start with this model, and what remains to measure

**Recommendation, inferred from source:** UNet++ is a reasonable first
compute-conscious target because it is a short encoder-only implementation,
starts at 32 channels, uses instance normalization, and has a directly specified
published comparison. Its released checkpoint is 43.3 MB, versus 849.6 MB for
ResNet. Checkpoint size is not GPU peak memory or a speed benchmark.
[Encoder][unetpp-encoder], [checkpoint inventory](https://zenodo.org/records/18821031).
ResNet is an alternative when prioritizing the authors' strongest reported RD
sensitivity; a standard torchvision ResNet18 is not that model.
[ResNet definition lines 335–396][classifier].

A free Colab T4 session has not been measured for this architecture/input. Google
says hardware availability and limits vary; free sessions can last at most 12
hours depending on availability/usage. Do not promise a particular GPU, batch,
fit, or 50-epoch completion time. [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits).

UNet++ baseline implementation checklist:

- [ ] Build only this architecture in our own project, with attribution and
      [upstream license][license] obligations considered for any adapted source;
      leave the upstream checkout untouched.
- [ ] Record source/dataset revisions, checksums, task labels, shape, parameter
      count, seed, dependency versions and all resolved optimizer/scheduler defaults.
- [ ] Verify released split membership against existing project audits and verify
      every acquired video hash. Never derive patient groups from filename prefixes.
- [ ] Check a real official-training-only batch for decode, exact preprocessing,
      forward/backward, finite loss/gradients, and a diagnostic overfit; discard
      diagnostic weights before initializing the actual run.
- [ ] Measure peak GPU allocation and wall time for actual training steps and
      a full validation pass. Time includes decoding/interpolation and checkpoint IO.
- [ ] Try the preserved effective batch 48 with a measured physical microbatch
      and explicit gradient accumulation if needed; correctly weight an incomplete
      final accumulation window. If changing batch, precision, frames, dimensions
      or epochs, record the deviation and retain an appropriately qualified name.
- [ ] Estimate `50 × (training-epoch time + validation time + save overhead)`
      from measurements. Resumable checkpoints must include optimizer, scheduler,
      epoch, random states and configuration; Colab resets cannot erase progress.
- [ ] Freeze the protocol, train with all 3,873 official training clips and all
      431 validation clips, unweighted loss, seed 42, 50 epochs, and validation
      accuracy checkpoint selection. A partial run is reproduction in progress.
- [ ] Evaluate the frozen selected checkpoint once on the 1,077 official test
      clips; compare against the UNet++ row with the optimizer/denominator gaps
      stated. Validation numbers are never presented as paper test reproduction.

For the upcoming progress talk after the weekend, the defensible evidence includes the completed
engineering pilot/comparison, this paper-to-code audit, a measured preflight,
and whatever full-training progress is actually recorded. Finishing all 40 models
is unnecessary for the stated single-baseline objective. A later controlled
contribution should be defined after this reference attempt, with development
choices made on validation data and final-test comparisons planned in advance.

## Audit checks and limits

Commands used: `rg --files`, `rg -n`, `nl -ba`/`sed` for read-only source
inspection; `git rev-parse HEAD`, `git remote -v`, `git status --short`; and Python
standard-library CSV counting across all 15 split files. Paper, author checkpoint
inventory/model card, MONAI source and Colab FAQ were opened as primary sources.
The read-only upstream interpreter also constructed the chosen model to count
parameters and inspect modules; bytecode writing was disabled.
An offline `uv --with pymupdf` guide extraction failed because that package was
not cached; no dependency or environment files were changed for it.
The new-file diff was reviewed and whitespace checked. Upstream status remained
its pre-existing untracked `uv.lock`; the audit made no upstream changes.

No training, video acquisition, author-checkpoint inference, Colab/CUDA benchmark,
or final-test evaluation was performed by this research subtask. These findings
specify an experiment; they do not validate a medical diagnostic system.

[factory]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/models/components/factory.py#L25-L143
[classifier]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/models/components/cls_model.py#L15-L494
[unetpp-encoder]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/models/components/encoders/unetplusplus.py#L24-L94
[unetr-encoder]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/models/components/encoders/unetr.py#L72-L167
[dataset]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/data/components/erdes_dataset.py#L31-L59
[resize]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/data/components/utils.py#L9-L66
[data-config]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/data/erdes.yaml
[datamodule]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/data/erdes_datamodule.py#L42-L84
[splits-code]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/data/create_data_splits.py#L20-L100
[splits]: https://github.com/OSUPCVLab/ERDES/tree/4af67e0b9f4667edbae75a64a819884ee14d7a3b/data/splits/non_rd_vs_rd
[all-splits]: https://github.com/OSUPCVLab/ERDES/tree/4af67e0b9f4667edbae75a64a819884ee14d7a3b/data/splits
[experiment]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/experiment/non_rd_vs_rd/unetplusplus.yaml#L6-L55
[model-config]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/model/unetplusplus.yaml#L1-L23
[train]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/train.py#L47-L92
[module]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/erdes/models/model_module.py#L64-L305
[callbacks]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/callbacks/default.yaml
[trainer]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/trainer/default.yaml
[train-config]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/train.yaml
[launch]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/script/run_exp3_nonrd_rd.sh#L13-L20
[ddp]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/configs/trainer/ddp.yaml
[requirements]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/requirements.txt
[license]: https://github.com/OSUPCVLab/ERDES/blob/4af67e0b9f4667edbae75a64a819884ee14d7a3b/LICENSE
