# Erdes Paper Notes

The grouping/release review is supplemented by the 2026-10-30
[UNet++ protocol and source audit](research/erdes-unetpp-protocol.md).
That note compares paper v2 with the pinned executable configurations, identifies
open disagreements, and specifies our first task/model baseline. Unfilled topics
below still need separate review; code claims are not substituted for paper evidence.

## Citation

Ozkut, Y., Navard, P., Adhikari, S., Situ-LaCasse, E., Acuña, J., Yarnish, A. A.,
and Yilmaz, A. *ERDES: A Benchmark Video Dataset for Retinal Detachment and
Macular Status Classification in Ocular Ultrasound*. arXiv:2508.04735v2,
4 March 2026. [Versioned paper](https://arxiv.org/html/2508.04735v2).

## Dataset purpose

TODO — evidence pending.

## Clinical problem

TODO — evidence pending.

## Data source

TODO — evidence pending.

## Dataset composition

TODO — evidence pending.

## Labels/classes

TODO — evidence pending.

## Preprocessing

For benchmark training the paper specifies channel-mean grayscale, symmetric
square padding, trilinear time/spatial resizing to `[1,96,128,128]`, and division
by 255. Released clips are already cropped; the benchmark loader does not run
the ROI detector again. This differs from our 16-frame engineering pilot.
[Table 6 / source evidence](research/erdes-unetpp-protocol.md).

## Train/validation/test strategy

Data Record, Table 4(a), reports the non-RD/RD counts observed in our audit.
The paragraph after Table 5 states that splits are clip-level because patient
identifiers were removed, so clips from one patient may appear in multiple
partitions. Actual patient-overlap counts cannot be recovered from the released
metadata. [Data Record](https://arxiv.org/html/2508.04735v2#S3).

## Architectures

The paper benchmarks eight named architectures over five tasks. Our selected
scope is one UNet++ classifier on non-RD versus RD. In the released code this
classifier uses a five-block down path with widths 32/32/64/128/256, instance
normalization, global average pooling and a 256→128→1 head; it does not retain
the nested segmentation decoder. Our independently written equivalent has
3,603,553 parameters and passed CPU parameter/forward/backward parity checks
against the pinned MONAI reference. [Exact definitions and citations](research/erdes-unetpp-protocol.md).

## Training setup

The intended reference training uses the full official training/validation sets
and 50 epochs from scratch. The paper names AdamW, but the released UNet++
configuration uses Adam, lr 1.5e-5, weight decay 0, seed 42, effective batch 48,
a plateau scheduler on validation loss and checkpoint selection by validation
accuracy. No weighting or resampling is used. This disagreement must be disclosed;
our first configuration explicitly follows the released code. Hardware fit and
completion time require the Colab resource check. [Protocol audit](research/erdes-unetpp-protocol.md).

## Evaluation metrics

TODO — evidence pending.

## Main results

The UNet++ non-RD/RD row of Table 8 reports **test** sensitivity 0.879, specificity
0.986, precision 0.871, F1 0.875 and accuracy 0.976. These are the authors' results,
not ours. The reported recall does not reconcile with a simple global count of
100 RD test examples; its aggregation is an unresolved source issue.
[Results and denominator caveat](research/erdes-unetpp-protocol.md).

## Error analysis

TODO — evidence pending.

## Limitations

Supplied partitions do not enforce patient independence. Benchmark results using
them must not be described as evaluation on unseen patients. This is a protocol
limitation; we have not independently measured patient overlap.
[Data Record](https://arxiv.org/html/2508.04735v2#S3).

## Reproducibility resources

TODO — evidence pending.

## Questions for our professor

TODO — evidence pending.

## Potential research gaps

TODO — evidence pending.

## What we should reproduce

One manageable non-RD/RD UNet++ reference, followed by a controlled extension
using the same model/data protocol. File and variable names identify our model
and task; methods retain attribution and disclose deviations. The immediate
artifact is [the UNet++ resource check](../notebooks/unetpp_resource_check.ipynb),
not a completed 50-epoch result. [Execution plan](UNETPP_BASELINE_PLAN.md).

## What we should NOT simply repeat

TODO — evidence pending.
