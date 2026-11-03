# Notebooks

Use one notebook for each focused stage:

| Order | Notebook | Purpose | Results folder |
| --- | --- | --- | --- |
| 1 | [setup_and_eda.ipynb](setup_and_eda.ipynb) | Metadata checks, labels, class imbalance, and clip lengths | `outputs/metadata_eda/` |
| 2 | [split_audit.ipynb](split_audit.ipynb) | Supplied non-RD/RD partitions, label agreement, coverage, clip overlap, and grouping limitations | `outputs/split_audit/` |
| 3 | [video_inspection.ipynb](video_inspection.ipynb) | Eight training clips: source hashes, decoding, frame previews, playback, and visual notes | `outputs/video_inspection/` |
| 4 | [baseline_preparation.ipynb](baseline_preparation.ipynb) | Verify fixed-size inputs and save the proposed first GPU-pilot configuration | `outputs/baseline_preparation/` |
| 5 | [baseline_training.ipynb](baseline_training.ipynb) | GPU/overfit checks, first pilot, prediction audit, and controlled weighted-loss comparison | `outputs/baseline_training/<run>/` |
| 6 | [unetpp_resource_check.ipynb](unetpp_resource_check.ipynb) | One released UNet++ classifier at the paper's 96-frame input: GPU memory/runtime preflight | `outputs/unetpp_resource_check/<run>/` or the corresponding mounted Drive folder |
| 7 | [unetpp_training.ipynb](unetpp_training.ipynb) | Full official train/validation inputs, mixed-precision check, fixed 50-epoch training with resume | `MyDrive/rd-ultrasound-ml/outputs/unetpp_training/unetpp_rd_001_amp/` |

Each notebook runs independently from a fresh kernel and downloads its own
inputs when needed. The UNet++ training stage prepares the full train/validation
inputs; earlier stages use metadata or small video subsets. The first four use a CPU runtime, with no PyTorch or upstream
clone required. The first two need no videos; the third downloads about 4.3 MB
of selected training clips. The fifth requires a CUDA GPU and Colab's managed
PyTorch environment. It downloads only its selected training/validation videos.

## Run with Colab in VS Code

1. Install Google's [Colab extension](https://marketplace.visualstudio.com/items?itemName=Google.colab)
   (`Google.colab`).
2. Open the notebook, then choose **Select Kernel → Colab → Auto Connect** and
   sign in. A CPU runtime is sufficient for the first four notebooks. For
   `baseline_training.ipynb`, select a Colab GPU runtime and reconnect before
   running from section 1. In browser Colab use **Runtime → Change runtime type →
   GPU** (T4 if available). Free GPU availability is not guaranteed.
3. Run section 1 first. Confirm that imports work and read the printed storage paths.
4. Continue one cell at a time with **Shift+Enter**, reading each result.

See Google's [extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).
You can also upload the notebook to [Colab in a browser](https://colab.research.google.com/).

The notebook is edited locally in VS Code, but a Colab kernel runs Python on a
remote machine. Local files and `.venv` are separate from that runtime. This
notebook downloads its inputs directly, so no project upload or clone is needed.
The metadata notebook uses pandas and matplotlib; the split audit needs only
pandas. Video inspection adds OpenCV for decoding and notebook video display.
Training adds torch, numpy, scikit-learn, and huggingface_hub; it checks CUDA,
records installed versions, and stops if there is no GPU. It deliberately uses
Colab's existing Python/CUDA environment instead of installing the local project
or replacing the managed torch wheel. Local environment changes do not configure
the remote kernel. Each notebook records versions in section 1.
If an import fails, stop and inspect the selected runtime before adding packages.

### Recover an interrupted training-video download

If acquisition stops after some clips with HTTP 503 or another temporary download
error, keep the current Colab GPU session connected. In the updated training
notebook run **Download helper — safe to rerun after an HTTP error**, then rerun
the failed acquisition/preprocessing cell. Do not rerun the runtime/storage cell
or Run All: the helper preserves the existing run and passed diagnostic state.
The progress counter starts at 1 again while cached clips are hash-checked and
reused; only missing videos need downloading. The helper makes up to five attempts
for transient errors and honors Retry-After. A persistent failure identifies the
clip; no clip is silently dropped or substituted. Keep its final message for review.

### Recover a preprocessing range error or existing-array-folder error

Keep the same runtime. Run the updated section 4 cell defining `prepare_frame`
and `prepare_clip`, then the updated section 7 acquisition/preprocessing cell.
Section 7 now uses `array_dir.mkdir(parents=True, exist_ok=True)` and rebuilds
incomplete arrays from the unchanged manifests, reusing hash-verified videos.
It refuses to rebuild inputs if pilot training artifacts already exist.

The preprocessing helper corrects float32 area-resize drift of at most `1e-6`
at the range boundaries. Nonfinite values and larger errors still stop the run;
the final clip check still requires values strictly within `[0,1]`. The numerical
handling is recorded in `preprocessing_implementation.json`. Run step 8 only
after section 7 prints both complete `Prepared:` array shapes. No Run All,
runtime reconnection, video deletion, or dataset substitution is needed for recovery.

## Optional: run on your own computer

Use uv from the project root to declare the analysis dependencies:

```bash
uv add --group analysis pandas matplotlib ipykernel
```

For local video inspection, also declare the decoder dependency:

```bash
uv add --group analysis opencv-python-headless
```

Colab normally already provides `cv2`; do not install a second OpenCV distribution
there just to match the local package name. Check the actual selected kernel.

Select the project's `.venv` as the notebook kernel. Review and commit the
resulting `pyproject.toml` and `uv.lock` changes with the notebook. Teammates can
then recreate that environment with `uv sync --locked --group analysis`.
The existing Python requirement is provisional; check compatibility when
resolving dependencies. A local uv environment does not configure Colab.

## Save and review

- Metadata and the split archive go under `data/erdes/`; unchanged CSV copies go
  under `data/erdes/splits/non_rd_vs_rd/`. Result folders are listed in the table
  above. Inputs and outputs are ignored by Git. Colab uses `/content` when no
  project checkout is present.
- The selected videos go under `data/erdes/clips/`, preserving the metadata paths.
- Colab storage is temporary. Preserve `metadata_summary.json` and
  `metadata_overview.png` from the first notebook, and `split_summary.csv` and
  `split_audit.json` from the second, by downloading them or copying them to Drive.
  Saving a notebook does not save its data files.
- From video inspection, preserve `inspection_summary.json`, `decode_report.csv`,
  `training_frame_examples.png`, and your completed `visual_review_notes.csv`.
- Agent inspection notes and measurements are recorded in local
  `outputs/video_inspection/visual_review_notes.csv` and `observations_summary.json`.
  The user's executed notebook was archived in that ignored folder before clearing
  embedded video/image outputs. The tracked notebook includes a concise review summary.
- From baseline preparation, preserve `baseline_001.json`, `preparation_summary.json`,
  and `preprocessed_examples.png`. The notebook embeds the tracked config so Colab
  does not need a project upload. It does not train a model or evaluate validation/test data.
- From baseline training, preserve the exported results ZIP and executed notebook.
  The ZIP contains configuration, manifests, runtime, diagnostics, training history,
  validation predictions/metrics, `best.pt`, and `last.pt`; it excludes raw videos
  and prepared arrays. Set `USE_GOOGLE_DRIVE = True` in its first cell for persistent
  per-epoch checkpoints, or download the ZIP before the runtime ends. Run folders
  are unique; partial runs are preserved but automatic training resume is not implemented.
- When a kernel restarts, run the selected notebook from its first code cell.
  No execution state is shared between notebooks; cached inputs are checked and reused.
- Save the notebook after editing. Review outputs and clear sensitive details
  before committing. Move reusable logic into `src/` when later work needs it.
- Clear **all outputs** from `video_inspection.ipynb` before committing it:
  playback embeds video bytes and the frame overview contains dataset images.
  Retain those artifacts under ignored output folders instead.

## What comes next

1. Preparation passed in the user's saved Colab execution. Review the transformed inputs.
2. Review the [baseline plan](../docs/BASELINE_PLAN.md), including the
   documented possible patient overlap and distinction between inspection clips
   and the future pilot training subset.
3. Open `baseline_training.ipynb` on a GPU runtime and run from the top. First
   confirm CUDA is available, then pass the forward/backward and four-training-clip
   overfit checks. Only then acquire the 256 training and 431 validation clips
   and run five epochs. Stop and retain evidence if any check fails.
4. Review `summary.json`, `history.csv`, and `diagnostics.json` before designing
   the next comparison. A diagnostic pass is not a validation performance result.
5. The user's first pilot is now complete: all-non-RD validation predictions at
   threshold 0.5, AP 0.090683. Preserve its original ZIP and continue directly to
   sections **10–12** in the same T4 session. Section 10 audits saved predictions
   and score distributions; section 11 changes only positive-class weighting in
   the training loss; section 12 exports a separate run and comparison table.
   No videos are downloaded or preprocessed again. Original baseline artifacts
   and cells are preserved. Avoid Run All or runtime reconnection for this continuation.
   If the session has already reset, stop and restore its inputs/state before following
   these continuation cells; automatic session restoration is not implemented.

Reserve final test evaluation until model development is complete.

## Next: one doable paper-baseline reproduction

Both custom-CNN pilots are now complete. The weighted run improved validation
AP to 0.145320 and ROC-AUC to 0.611061, while still missing all RD at threshold
0.5. They do not reproduce the paper: their model, input length, training size,
optimizer and epoch budget differ.

The professor's clarified target is one manageable published reference before
the new contribution. Open `unetpp_resource_check.ipynb` on the available
Colab CUDA GPU and run it from the top. It is standalone; do not rerun the old
baseline notebook to use it. It checks the released UNet++ classifier's full
96×128×128 inputs with two hash-verified training clips and physical batches
1/2/4. It saves to mounted Drive by default and records candidate accumulation
for effective batch 48. No full research training or test evaluation occurs.

The saved T4 check now passes batch 4 (6.54 GB allocated) with a 16.92-hour
FP32 compute-only projection. Next open **`unetpp_training.ipynb`** on the same
T4 and run from the top. Drive persistence is required and enabled. It checks
mixed precision on two real training clips, then prepares all 3,873 training and
431 validation inputs on runtime disk (about 27.08 GB). No 100-request cap or
256-clip subsampling is imposed. No test CSV is opened.

In section 6 leave `UNETPP_EPOCHS_THIS_SESSION = 1` for the first complete epoch.
Then set it to `None` and rerun **section 6 only** to continue toward the fixed
50-epoch target. Checkpoints save during each epoch and after validation, with
optimizer/scheduler/scaler/RNG/cursor state. After a runtime reset, run this
notebook from the top with the **same run name**: inputs are rebuilt on runtime
disk and the run resumes from Drive. The old custom-pilot notebook has no such
resume support; those earlier instructions apply only to that notebook.

For a transient download error, rerun section 5 in the same runtime to reuse
verified inputs. Do not delete videos, change official splits or reduce input
size. An integrity/config/runtime mismatch stops explicitly and needs inspection.
The optional export ZIP is saved beside the run folder in Google Drive, not on
your Windows computer. Browser location: **My Drive → rd-ultrasound-ml → outputs
→ unetpp_training**. A run stays labeled partial until 50 epochs complete.
[Plan, source attribution and limitations](../docs/UNETPP_BASELINE_PLAN.md).


## Kaggle alternative after Colab quota exhaustion

Follow [the Kaggle setup guide](../docs/KAGGLE_GUIDE.md). From the VS Code WSL
terminal run `bash scripts/setup_kaggle.sh` to configure your account/token
locally and generate private submission bundles. This uses the official CLI
for saved remote jobs; it does not attach an interactive Kaggle kernel in VS Code.

`kaggle_unetpp_preparation.ipynb` runs as two separate **CPU** jobs (indices 0/1).
Each saves half of the unchanged float32 inputs below the 20 GB per-notebook
output limit. `kaggle_unetpp_training.ipynb` attaches both saved outputs, checks
them and trains on GPU without repeating video acquisition/preprocessing.
Start with one full epoch. Additional bounded jobs attach the preceding saved
training output to resume the same fixed 50-epoch protocol. No Drive mount or
final-test access is required. Saved output must finish successfully before it
can be used to continue; an unsaved interrupted job can lose its current interval.


## After 50 epochs: frozen official-test evaluation

The Kaggle UNet++ run and official-test evaluation are complete. Epoch 48 and
threshold 0.5 were frozen before evaluation-workflow test access; earlier
split-integrity audits had read membership/counts.
[Results and protocol](../docs/UNETPP_RESULTS.md).

| Notebook | Accelerator | Purpose |
| --- | --- | --- |
| [kaggle_unetpp_test_preparation.ipynb](kaggle_unetpp_test_preparation.ipynb) | None; Internet enabled | Verify the final training artifact/freeze, then cache all 1,077 test clips |
| [kaggle_unetpp_evaluation.ipynb](kaggle_unetpp_evaluation.ipynb) | GPU; Internet disabled | Verify frozen weights/test inputs, infer once and export official-test metrics |

CPU test preparation and GPU evaluation version 1 both completed all 1,077 clips.
The saved results have been downloaded and verified; do not launch another copy.
The GPU evaluation bundle attaches the completed CPU test output and final
training output by version. It has no training, threshold search, calibration
fit or checkpoint selection. Its verified test accuracy is 97.49%, sensitivity
86.00% and specificity 98.67%; the linked report contains the paper comparison.


## Completed results: CPU tables and figures

Open [unetpp_results_analysis.ipynb](unetpp_results_analysis.ipynb) for the verified
split tables, learning curves, confusion counts, ROC/precision–recall curves,
score distributions, paper comparison and explicitly qualified uncertainty
intervals. It reads saved archives and needs no GPU or raw videos. A ready-to-view
executed copy is under `outputs/unetpp_analysis/`; figures export as PNG/PDF and
tables as CSV. See [the reporting command](../docs/UNETPP_RESULTS.md#results-notebook-and-presentation)
and [12-minute speaker guide](../docs/UNETPP_PRESENTATION_GUIDE.md).
