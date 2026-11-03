# UNet++ progress presentation — 12-minute speaker guide

[Open the new Google Slides deck](https://docs.google.com/presentation/d/1kBcOG2HnWa4MnhlQ5gq0C-e4ZFujRQbsttXLXTiSXd0/edit?usp=drivesdk).
It is saved in My Drive → ChatGPT. The previous deck is unchanged.

13 main slides; 12 minutes of speaking. Questions afterward. Two appendix slides have no allocated talk time.
Team names follow the previous deck: Bijay, Naman and Ritu. Rehearse handoffs and divide speaking roles together.
Figures are derived from saved artifacts, without retraining or test threshold tuning.

## 01. Retinal Detachment Analysis
from Ocular Ultrasound
Time: 0:00–0:25

Our project investigates retinal-detachment classification from ocular ultrasound videos. In the previous presentation we introduced the medical motivation, the ERDES dataset, and our plan to build a reproducible baseline. Today we can report a completed single-model benchmark, including a separate frozen test evaluation. We will explain what worked, what the early pilots taught us, and the limits of the findings.

Source: Local verified artifacts · UNet++ RD 002

## 02. A quick recap: our task and motivation
Time: 0:25–1:15

Last time, we explained that ultrasound provides a video rather than one isolated image. We selected ERDES as a labeled research dataset and narrowed our first task to retinal detachment versus non-retinal detachment. Non-RD includes both normal clips and posterior vitreous detachment, or PVD. That distinction matters because a model must learn more than simply whether a membrane is visible. We also highlighted that the dataset contains far more non-RD than RD clips. Our original plan was to verify the data, establish a reproducible reference, and then investigate errors or a controlled improvement. This update follows that sequence. Macular-status classification is outside this completed experiment.

Source: Previous team presentation · ERDES v2, dataset/task definitions

## 03. The planned baseline is now a measured result
Time: 1:15–2:00

Our previous presentation ended with two technical gates: a verified data audit and a reproducible baseline. Both are now complete. We checked metadata and official split membership, inspected training videos, and tested input preprocessing. Next we ran two small CNN pilots to establish the training and evaluation pipeline. Finally, we trained the released UNet++ encoder-classifier equivalent for fifty epochs on the full official training set. We selected the checkpoint using validation only, froze it, and evaluated the official test set separately. The next contribution experiment has not yet been run, so we keep it separate from this completed reference.

Source: Local verified artifacts · UNet++ RD 002

## 04. The official split is strongly imbalanced
Time: 2:00–2:55

The released binary task contains 5,381 clips: 4,879 non-RD and 502 RD. We retained the supplied assignments: 3,873 training clips, 431 validation clips, and 1,077 test clips. RD is roughly nine percent of each partition. Our integrity checks found no shared clip paths between partitions and labels agreed with metadata. However, this does not prove patient independence. The paper says identifiers were removed and the split is at clip level, so clips from one patient may occur in multiple partitions. We therefore describe this as performance on the official clip-level benchmark. The imbalance also explains why accuracy alone is insufficient: an always-non-RD predictor already reaches about ninety-one percent accuracy.

Source: Official split audit · ERDES v2, Data Records

## 05. The pilot exposed a misleading accuracy score
Time: 2:55–3:55

Our first model was a compact custom 3D CNN, trained for only five epochs on 256 clips using sixteen sampled frames. It reported about ninety-one percent validation accuracy. The confusion counts revealed the problem: it predicted every validation clip as non-RD and missed all forty RD clips. We then tested positive-class weighting in the same small pilot. Average precision improved from about 0.091 to 0.145, but sensitivity at the fixed threshold remained zero. This was useful engineering evidence, not successful RD detection. The later full-data UNet++ run changes the model, training data amount, input duration and epoch budget together. We cannot attribute the resulting improvement to architecture alone. The pilots established that we must report sensitivity, specificity and ranking metrics alongside accuracy.

Source: Local verified artifacts · UNet++ RD 002

## 06. One feasible reference: the UNet++ encoder classifier
Time: 3:55–4:50

We chose one feasible model rather than attempting all forty paper experiments. The released UNet++ model for this task is an encoder-based classifier, not a segmentation output. It has approximately 3.6 million parameters. We use the released pre-cropped clips, convert them to grayscale, square-pad, and interpolate the complete video to ninety-six frames of 128 by 128 pixels. The classifier uses 3D convolutional blocks, global average pooling and a binary head. Training follows the released Adam configuration, unweighted binary cross-entropy and fifty epochs. Our effective batch is forty-eight, implemented by accumulating twelve physical batches of four. We disclose differences from the reference, including mixed precision and decoding. Checkpoint selection remains highest validation accuracy with the earliest tie.

Source: Released ERDES code/config audit · configs/unetpp_rd_002_training.json

## 07. A resumable workflow made free GPU compute workable
Time: 4:50–5:45

The full experiment exceeded our remaining Colab allocation, so we moved to Kaggle. To avoid spending GPU quota downloading and preparing videos, we prepared the training and validation inputs in two CPU jobs. Each output was saved, and the GPU jobs verified the cached tensor hashes before training. Training was resumed across three saved jobs, preserving model, optimizer, scheduler, scaler and random states. The allocation was T4 times two, but our implementation used one GPU. For test evaluation, we froze epoch forty-eight and threshold point-five before the evaluation workflow accessed test inputs. CPU preparation then saved the test cache, and a separate inference-only GPU job used the original training container. This makes the run traceable and avoids confusing resumed training with a new experiment.

Source: Local verified artifacts · UNet++ RD 002

## 08. Learning curves show when RD detection emerged
Time: 5:45–6:50

The left panel shows training and validation binary cross-entropy over all fifty epochs. Both decrease substantially. The right panel explains why looking only at accuracy would have hidden early failure. Validation accuracy initially stays near ninety-one percent while sensitivity is zero. Average precision improves earlier, showing better score ordering before many RD scores cross the fixed threshold. Sensitivity starts rising later. The dotted vertical line is epoch forty-eight, selected by our pre-specified validation-accuracy rule. Its validation accuracy is 98.84 percent and sensitivity is 92.5 percent. Epoch forty-nine ties on accuracy and does not replace the earlier checkpoint. Epoch fifty is the final training state, but it is not the selected model. These are validation findings; the next slides show the separate official test evaluation.

Source: Local verified artifacts · UNet++ RD 002

## 09. Official test: 86 RD detected, 14 missed
Time: 6:50–7:50

This confusion matrix is the main test result. At the frozen threshold, the model correctly identified eighty-six RD clips and missed fourteen, out of one hundred RD clips. Among 977 non-RD clips, it correctly rejected 964 and raised thirteen false alarms. This gives 97.49 percent accuracy, 86 percent sensitivity and 98.67 percent specificity. The cell colors are normalized within each dataset class, while the large numbers show raw counts. This prevents the much larger negative class from visually hiding the RD errors. All 1,077 saved predictions were checked against the test manifest and all twelve exported artifact hashes were verified. These outcomes describe agreement with dataset labels. They do not establish clinical performance or performance on an independent hospital population.

Source: Local verified artifacts · UNet++ RD 002

## 10. Ranking is strong; the operating point remains fixed
Time: 7:50–8:55

The ROC curve summarizes sensitivity against the false-positive rate. Its area is approximately 0.987. Because RD is uncommon, we also show precision against recall. Average precision is approximately 0.927, compared with an RD prevalence of about 0.093. Here average precision uses the standard non-interpolated statistic, rather than being relabeled as trapezoidal PR area. The marked point on both curves is threshold point-five, which we fixed before test inference. We are not selecting a more favorable test threshold. These plots describe ranking across this test partition; the scores have not been calibrated as clinical risk probabilities. A strong ROC curve also does not erase the fourteen missed RD clips at the reported operating point. We therefore show curves and confusion counts together.

Source: Local verified artifacts · UNet++ RD 002

## 11. Our test result is close to the published UNet++ row
Time: 8:55–9:55

We compare our official test result to the correct row: UNet++ for non-RD versus RD in Table eight. We do not compare our selected validation score to the paper's test score. Our accuracy is 97.49 percent, against the reported 97.6 percent. Sensitivity is 86 percent, against 87.9 percent; specificity is 98.67 percent, against 98.6 percent. Precision and F1 are also shown. This establishes one feasible adapted benchmark close to the published result. It does not establish exact numerical replication or statistical equivalence. Important differences remain: the paper text says AdamW while the released configuration uses Adam, and our implementation uses a different decoder, gradient accumulation and mixed precision. The paper's sensitivity denominator or aggregation is also unresolved, so we retain its reported value without inventing an explanation.

Source: Ozkut et al., ERDES v2, Table 8 · https://arxiv.org/html/2508.04735v2

## 12. The remaining errors and limits shape our next study
Time: 9:55–11:05

The result is encouraging as a course benchmark, but its limitations determine what we can claim. First, sensitivity depends on only one hundred RD test clips. A change of one detected RD clip moves sensitivity by one percentage point, and fourteen were missed. Second, the official split is clip-level, so patient overlap may inflate apparent generalization. Third, we have one seed, one official split and no external dataset. Fourth, execution differs numerically from the reference, and a single mixed-precision optimizer update was skipped in epoch thirty-five and recorded. We do not interpret this result as clinical validation or a robust improvement over the paper. Finally, the test set is now exposed. Future experiment choices must use training and validation data, and any later comparison on the same official test must be disclosed as reuse. The useful next contribution is a pre-specified controlled study of errors or imbalance, not repeated test-driven tuning.

Source: Local verified artifacts · UNet++ RD 002

## 13. A completed reference, then one controlled extension
Time: 11:05–12:00

We now have the technical reference that was missing at our previous progress update: a verified pipeline, full fifty-epoch training, a frozen checkpoint, official-test predictions and a documented comparison with the paper. The immediate deliverable is this progress report with the results notebook and reusable figures. For the next experiment, we propose a single controlled question about imbalance handling, such as whether training-only class weighting improves validation sensitivity while tracking specificity. That proposal needs a written protocol before execution and should keep the model, inputs, splits and compute budget comparable. It is not yet a completed contribution or a promised improvement. We will preserve the current reference and disclose that the official test has already been evaluated. This gives the project a clear baseline and a defensible path to its own analysis.

Source: Local verified artifacts · UNet++ RD 002

## 14. Protocol and reproducibility details
Backup slide — outside speaking budget

Backup only. The input uses full-video temporal/spatial interpolation, rather than sixteen evenly sampled frames. We use the released encoder-classifier equivalent, unweighted BCE, Adam and fixed fifty-epoch budget. Data and checkpoint identities are retained. Validation was used for checkpoint selection; no optimizer or threshold search was used during test evaluation.

Source: Local verified artifacts · UNet++ RD 002

## 15. Sources and reusable evidence
Backup slide — outside speaking budget

Backup only. The original team presentation supplies the recap. The ERDES v2 article and released code/config audit provide the published comparison and protocol. The local verified archives provide our measurements. The results notebook exports every figure and table without retraining. See the repository report for complete SHA256 identities and limitations.

Source: Sources and artifacts; links are included in notes and the local speaker guide
