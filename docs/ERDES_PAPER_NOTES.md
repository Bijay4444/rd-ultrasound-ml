# Erdes Paper Notes

Partial review: grouping and release details were checked against paper v2.
Other sections below remain open; upstream README claims are not substituted for
paper evidence. See [the focused research note](research/erdes-grouping-and-access.md).

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

TODO — evidence pending.

## Train/validation/test strategy

Data Record, Table 4(a), reports the non-RD/RD counts observed in our audit.
The paragraph after Table 5 states that splits are clip-level because patient
identifiers were removed, so clips from one patient may appear in multiple
partitions. Actual patient-overlap counts cannot be recovered from the released
metadata. [Data Record](https://arxiv.org/html/2508.04735v2#S3).

## Architectures

TODO — evidence pending.

## Training setup

TODO — evidence pending.

## Evaluation metrics

TODO — evidence pending.

## Main results

TODO — evidence pending.

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

TODO — evidence pending.

## What we should NOT simply repeat

TODO — evidence pending.
