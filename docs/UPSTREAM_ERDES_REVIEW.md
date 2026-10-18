# Upstream ERDES inspection

Read-only checkout: `/home/bijay/first_sem_proj/dm_ml/ERDES`.
Inspected revision: `4af67e0b9f4667edbae75a64a819884ee14d7a3b`.
Source: [authors' repository](https://github.com/OSUPCVLab/ERDES).
These are static code observations, not verified runtime behavior or paper findings.
No source code was copied or changed. Upstream contains a GPL license file;
review reuse obligations before adapting code.

| Area | Evidence relative to upstream root |
| --- | --- |
| Structure | `erdes/`, `configs/`, `tests/`, `script/`, `pipe/`, `docs/`, local data/logs |
| Environment | `requirements.txt` uses broad torch/lightning minima and Hydra pins; `environment.yaml` specifies Python 3.10 and older major ranges. This does not determine our environment. |
| Loading | `erdes/data/components/erdes_dataset.py`: CSV `path`/`label`, optional data root, `torchvision.io.read_video`, float tensor `[C,D,H,W]`, RGB mean to grayscale, resize, divide by 255. |
| Resize | `erdes/data/components/utils.py`: square padding followed by trilinear interpolation across time and space. |
| Data configuration | `configs/data/erdes.yaml`: size `[96,128,128]`, batch 4, workers 0; CSV splits. These are upstream defaults, not our choices. |
| Splits | `erdes/data/create_data_splits.py`: stratified video rows, 72/8/20 proportions, seed 42. No explicit patient grouping; glob ordering is not sorted. Audit metadata and existing splits before adoption. |
| Entry points | `erdes/train.py` and `erdes/eval.py`; Hydra configurations and Lightning training. |
| Architectures | `configs/model/`: resnet3d, senet, unet3d, unetplusplus, vnet, vit, unetr, swinunetr. |
| Tasks | normal/RD, non-RD/RD, PVD/RD, detached/intact macula, normal/PVD. Positive class varies by task. |
| Metrics | `erdes/models/model_module.py`: binary accuracy, precision, sensitivity, specificity, F1, loss. Inspect aggregation before comparisons. |
| Logging/checkpoints | Hydra paths under logs/output; multiple optional logger configs; checkpoint callback defaults include null monitor, overridden settings need inspection. |
| Evaluation discipline | `configs/train.yaml` enables test after training and defaults seed to null. Our development must reserve final test evaluation and set explicit seeds. |
| Tests | data modules, configs, training, evaluation and sweeps. Not executed; may require data and upstream environment. |

Reuse the task definitions, reference configurations, and published comparison
protocol as attributed ideas after review. Implement a small independent data
audit and baseline instead of importing the full framework. Investigate video
codec support, current library API compatibility, path portability, split
grouping, experiment overrides, metrics aggregation and hardware needs before
reproduction. Historical constraints are compatibility questions, not established bugs.
