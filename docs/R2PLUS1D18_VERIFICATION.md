# R(2+1)D-18 independent saved-artifact verification

Completed 2026-11-16. **Result: passed for the downloaded training evidence,
selected checkpoint and test-result archive.** This audit recomputed metrics from
saved predictions; it did not rerun training or model inference.

## Sources and local evidence

The following saved outputs are now accessible using the project's configured
Kaggle account:

- [Test evaluation](https://www.kaggle.com/code/namanbabariya/r2plus1d18-rd-001-test-evaluation),
  downloaded explicitly as `namanbabariya/r2plus1d18-rd-001-test-evaluation/1`.
- [Final training](https://www.kaggle.com/code/namanbabariya/r2plus1d18-rd-001-train-04),
  downloaded explicitly as `namanbabariya/r2plus1d18-rd-001-train-04/1`.

All downloaded outputs and audit scripts are retained under ignored
`outputs/kaggle_results/r2plus1d18_final_verification/`:

| Path under that directory | Evidence |
| --- | --- |
| `download/r2plus1d18_rd_001_test_results.zip` | Original 69,878-byte test archive |
| `test_artifacts/` | Safely checked and extracted test files |
| `training_small/r2plus1d18_rd_001_amp/` | Final training configs, manifests, predictions, histories, source and curve |
| `checkpoint/r2plus1d18_rd_001_amp/best.pt` | Downloaded selected weights, 125,361,855 bytes |
| `verification.json` | Detailed saved-artifact audit and check results |
| `checkpoint_payload_verification.json` | Safe checkpoint payload inspection |
| `verified_comparison.csv` | Independently recomputed test metrics for both models |
| `verify_saved_artifacts.py` | Independent standard-library verifier |
| `verify_checkpoint_payload.py` | PyTorch `weights_only=True` checkpoint inspection |

The test ZIP SHA256 is:

```text
240c4213cc57c636356a4fac1bc05c056c5576ac2223511f44b8a24b3979bf36
```

The selected epoch-15 checkpoint SHA256 matches the tracked pre-evaluation freeze:

```text
c041f7e2f635e9f3203e76045fbad247c3243788d527425d27cc73b27b93ec16
```

## Checks passed

- **Integrity:** all 16 listed test artifact hashes match. The final training
  manifest contains 79 entries; 77 match their downloaded files, including all
  76 listed small artifacts and `best.pt`. `last_a.pt` and `last_b.pt` remain
  on Kaggle and were not verified. The artifact hash manifest itself is not
  self-authenticating; identity is also checked against the committed freeze,
  repository source and official release inputs.
- **Source/config identity:** archived test source equals the repository source;
  all seven frozen training implementation hashes match; the resolved training
  config equals the tracked template plus implementation identities and hashes
  to the pinned digest. Saved test freeze equals
  `configs/r2plus1d18_rd_001_evaluation.json` exactly. Training and test runtime
  records agree with that freeze.
- **Official partitions:** the locally retained original `splits.zip` and
  `erdes_metadata.csv` match their published MD5s. Training, validation and test
  manifests match official order, paths, clip IDs and labels. The test CSV SHA256
  matches the saved provenance. No clip is shared across partitions; this does
  not establish patient independence.
- **Training evidence:** histories contain epochs 1–50. All 50 sets of 431
  validation probabilities, totaling **21,550 values**, produce the recorded
  confusion counts, sensitivity, specificity, precision, F1, accuracy, balanced
  accuracy, AP and ROC-AUC within absolute tolerance `1e-12`. CSV history agrees
  with JSON history. Maximum validation accuracy selects **epoch 15** with the
  original earliest-tie rule. The best-prediction exports match that epoch.
- **Runtime and integration:** the run fingerprint recomputes from the saved
  training/cache identities. Memory reports cover all 50 epochs; their segment
  times sum to **22.62 hours**. One skipped AMP update is recorded, in epoch 17.
  The final disposable probe uses only official training clips. The saved
  Grad-CAM target is `backbone.layer4`, shape `[1,512,12,8,8]` on the fixed input.
- **Checkpoint:** SHA256 matches the freeze and final training manifest.
  `torch.load(..., map_location='cpu', weights_only=True)` confirms epoch 15,
  config digest, run fingerprint and `test_used=False`. All **224 state tensors**
  are finite; the 113 parameter tensors contain **31,296,228 FP32 parameters**.
  Grayscale stem, temporal stem and binary-head tensor shapes are correct.
- **Test predictions:** all **1,077** rows are unique and match official
  membership, order and labels (977 non-RD, 100 RD). All probabilities are finite
  and in `[0,1]`; predicted labels use the frozen threshold 0.5. Classification
  and ranking metrics were independently recomputed. The archived log records
  training-freeze verification before test inference and successful completion.
- **Reference comparison:** the existing U-Net++ test archive passes all 12
  listed hashes and matches the same ordered test clips. Paired comparisons
  reproduce the original report.

The independent verifier uses grouped-score average precision and rank-based
ROC-AUC, including ties; it does not import the model's metric function.

## Verified results

The selected checkpoint's **validation** result is accuracy **99.07%**,
sensitivity **95.00%**, specificity **99.49%**, TP 38 / TN 389 / FP 2 / FN 2.
These validation measurements were used for checkpoint selection.

The following table is the separate **reused official test benchmark**:

| Metric | R(2+1)D-18, epoch 15 | U-Net++, epoch 48 |
| --- | ---: | ---: |
| TP / TN / FP / FN | 92 / 962 / 15 / 8 | 86 / 964 / 13 / 14 |
| Sensitivity | 92.00% | 86.00% |
| Specificity | 98.46% | 98.67% |
| Precision | 85.98% | 86.87% |
| F1 | 88.89% | 86.43% |
| Accuracy | 97.86% | 97.49% |
| Balanced accuracy | 95.23% | 92.33% |
| Average precision | 0.958854 | 0.926845 |
| ROC-AUC | 0.993321 | 0.986858 |

R(2+1)D-18 is correct alone on 15 clips and U-Net++ alone on 11; both are wrong
on 12. The exact two-sided McNemar value is **0.557197**, so this comparison
does not establish an overall accuracy improvement. On RD clips, the discordant
counts are 6 versus 0 (exact two-sided value 0.03125). That subgroup test is
post hoc and unadjusted; possible related clips and one seed further limit
interpretation. The descriptive result is six fewer missed RD clips and two
more false positives on this benchmark.

## Commands and limits

The downloads used the existing read-only Kaggle CLI workflow with explicit
`/1` source versions and filename filters. Small training artifacts were
downloaded separately from the selected checkpoint to avoid the large resume
checkpoints and full training ZIP.

```bash
uv --cache-dir /tmp/dmml-uv-cache run --locked --offline python outputs/kaggle_results/r2plus1d18_final_verification/verify_saved_artifacts.py
PYTHONDONTWRITEBYTECODE=1 ../ERDES/.venv/bin/python outputs/kaggle_results/r2plus1d18_final_verification/verify_checkpoint_payload.py
```

The payload inspection reused the existing read-only CPU PyTorch 2.5.1
environment without installing packages or changing upstream files. It does
not set this project's Python version. The standard-library artifact audit uses
the project's uv environment.

Not performed: a fresh inference/reload probability reproduction on CUDA, full
ML test-suite execution, raw-video decoding, rehashing the remote cached input
tensors, or downloading/validating the two resume checkpoints. BCE loss cannot
be reproduced exactly from exported sigmoid probabilities without original
logits; its recorded value is preserved rather than independently certified.
The saved training-run reload report records zero probability difference, but
that computation was not rerun by this audit.

The existing freeze confirms this model's choices, not a team-wide freeze.
Preserve disclosure that the official test was already exposed. These checks
support the saved-artifact result and checkpoint identity; they do not establish
clinical validation, unseen-patient performance or repeated-seed robustness.
