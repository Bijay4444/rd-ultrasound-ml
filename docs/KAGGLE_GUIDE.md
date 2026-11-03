# Run UNet++ on Kaggle from VS Code

This is the free-compute alternative after Colab access was exhausted during
full-data preparation. The user reached `READY: 3873 train, 431 validation` but
had not started the full UNet++ training cell. The displayed constant non-RD
metrics were a reference, not trained UNet++ results. No Colab model checkpoint
needs migration if training never started.

The supported route here is **edit notebooks in VS Code, submit saved jobs with
the official Kaggle CLI, inspect logs/output in Kaggle, and retrieve results into
this workspace**. This does not attach Kaggle as an interactive VS Code kernel
like Google's Colab extension. No third-party extension or SSH tunnel is needed.
See [Kaggle's CLI notebook commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

## Storage and execution

| Stage | Notebook | Accelerator | Saved output |
| --- | --- | --- | --- |
| Input shard 00 | `notebooks/kaggle_unetpp_preparation.ipynb`, index 0 | None | First 2,152 train/validation clips, about 13.54 GB |
| Input shard 01 | Same notebook, index 1, separate job | None | Remaining 2,152 clips, about 13.54 GB |
| Training | `notebooks/kaggle_unetpp_training.ipynb` | Available CUDA GPU; request T4 | Checkpoints, history, predictions, results ZIP |

Kaggle documents 20 GB of saved output per notebook and supports attaching a
completed notebook's outputs to another notebook. The two-shard design keeps
all **float32** values without quantization or reducing the dataset. Raw clips
are verified, decoded into the cache and removed from the preparation job's
own temporary directory. The GPU job reads both saved input shards directly;
it does not redownload/decode all videos. GPU hash checks still take some time.
[Notebook environment and output chaining](https://www.kaggle.com/docs/notebooks).

The source-aligned 96-frame encoder classifier, official partitions, Adam,
unweighted loss, effective batch 48, accuracy checkpoint selection and fixed
50-epoch target remain. AMP and one-device accumulation are recorded adaptations.
Only GPU 0 is used even if T4 x2 is allocated. Kaggle runs use ID
`unetpp_rd_002_amp`, separate from the unstarted Colab run. No test CSV is read.
See [the protocol and its limitations](UNETPP_BASELINE_PLAN.md).

## 1. Set up the account locally

In the **VS Code terminal for this WSL project**, run:

```bash
bash scripts/setup_kaggle.sh
```

The wizard opens Kaggle settings, records your public username, guides you
through any account verification needed for GPU/Internet access, then accepts
an API token through a hidden terminal prompt. It stores only
`KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` in ignored `.env.kaggle` with mode 0600.
Do not paste credentials into chat, notebook cells, or submission metadata.

Kaggle's current authentication flow is **Settings → API tokens → Generate New
Token**; the direct page is <https://www.kaggle.com/settings/api>.
[Official authentication instructions](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).

The wizard generates three private submission bundles in
`outputs/kaggle_jobs/`: two CPU jobs and the first GPU job. It does not start them.
The wrapper uses an isolated uv environment with Kaggle CLI 2.2.5 plus `six`:
that CLI imports `six` without declaring it, as reproduced during setup. Project
Python/dependency metadata and the system installation are unchanged.

Check credentials from the same terminal:

```bash
python3 scripts/kaggle_cli.py --version
python3 scripts/kaggle_cli.py kernels list --mine
```

`--version` should print `Kaggle CLI 2.2.5`. The list command verifies account
access; an empty notebook list is valid. If GPU or Internet controls are locked,
complete Kaggle's requested verification before submitting. Quota is shown in
your Kaggle account/session settings; free access is not guaranteed.
[Kaggle GPU usage guide](https://www.kaggle.com/docs/efficient-gpu-usage).

## 2. Submit both CPU jobs

```bash
python3 scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/unetpp-rd-002-inputs-00
python3 scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/unetpp-rd-002-inputs-01
```

Both bundles explicitly disable GPU and enable Internet. If Kaggle limits
concurrent CPU runs, run them sequentially. Keep these notebooks private.
Each submission prints its Kaggle URL. In the browser inspect the running
version/log and wait for **both jobs to complete successfully and save outputs**.
The last preparation message says `CPU preparation COMPLETE`.

The bundle generator also prints each exact status command. Substitute your
public username for `YOUR_USERNAME` here:

```bash
python3 scripts/kaggle_cli.py kernels status YOUR_USERNAME/unetpp-rd-002-inputs-00
python3 scripts/kaggle_cli.py kernels status YOUR_USERNAME/unetpp-rd-002-inputs-01
```

Do not download the 27 GB input arrays onto your laptop. They become attached
Kaggle inputs automatically through the GPU bundle's `kernel_sources` metadata.
The `prepared_manifest.json` in each output must report `complete: true`.
If an interactive acquisition fails transiently, rerun its failed cell to reuse
verified arrays. A newly submitted CPU version starts a new execution, so do
not assume it retains an unsuccessful earlier run's working directory.

## 3. Submit the first full-data GPU epoch

Only after both CPU outputs are successfully saved:

```bash
python3 scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/unetpp-rd-002-train-01 --accelerator NvidiaTeslaT4
```

The VS Code wrapper now checks every required notebook dependency before
submitting GPU training. A dependency that is running, failed, unavailable or
unverifiable blocks submission locally, before a GPU run starts. This also
checks the preceding saved GPU job for continuation. `--no-run` remains a
source-only save and does not start training. Browser submissions need the
same manual completion check.

The GPU bundle attaches both CPU outputs, disables Internet, verifies every
prepared tensor and performs a disposable mixed-precision check. It trains
**one full epoch** by default, exports and finishes normally. Read the actual
epoch time, sensitivity, confusion counts, AP and skipped AMP updates.
If the requested GPU is unavailable, check Kaggle's available accelerators;
do not guess that changing a local VS Code kernel will allocate one.

A model or runtime mismatch must be inspected rather than bypassed. Existing
training requires the same recorded libraries/GPU to resume. Keep the Kaggle
container/environment stable across the preparation shards and training jobs;
the code records versions and rejects inconsistent inputs or training resume.

## 4. Continue from the saved GPU output

After job 01 finishes and its outputs are saved, generate a **new** GPU job.
For example, after confirming that five epochs fit comfortably:

```bash
python3 scripts/prepare_kaggle_job.py train --session 2 --epochs 5
python3 scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/unetpp-rd-002-train-02 --accelerator NvidiaTeslaT4
```

Job 02 automatically attaches both CPU outputs and job 01's saved output. It
requires the preceding checkpoint, copies only training artifacts into its own
working directory and restores model/Adam/scheduler/scaler/RNG state. It runs
up to five additional epochs; the experiment target stays **50 total**. Repeat
with session 3 after 02 succeeds, and so on. Each predecessor has a distinct
notebook slug: do not overwrite/rerun a successful predecessor while dependents
are being used. Dependency metadata uses its latest saved version; the input
fingerprint guards the attached preparation content.

The loop stops between epochs if its estimated next epoch would overrun a
six-hour job budget. This estimate is a precaution, not a guaranteed runtime.
It does not reduce the target or declare partial work complete. After 50 epochs,
freeze the selected checkpoint and plan separate official-test evaluation.

**Durability:** checkpoints in `/kaggle/working` alone are not guaranteed to
survive termination. Only successfully saved outputs are a durable handoff.
If a GPU job fails before output saving, retry a new job from its preceding
successful saved output; its unsaved interval may be lost. Do not use Colab's
Drive-resume instructions for Kaggle. No Google Drive mount is required.

## 5. Retrieve and inspect results in VS Code

The GPU output includes `/kaggle/working/unetpp_rd_002_amp_results.zip`. Download
only that ZIP from a successful job, rather than CPU input arrays:

```bash
python3 scripts/kaggle_cli.py kernels output YOUR_USERNAME/unetpp-rd-002-train-01 --file-pattern '.*_results\.zip$' -p outputs/kaggle_results/session_01
```

This writes the ZIP into this local WSL project, visible in VS Code Explorer
under `outputs/kaggle_results/session_01/`. Kaggle's browser **Output** tab is an
alternative download route. Preserve each successful job's output and record
its version/URL. The ZIP contains a `summary.json` labeled `partial` or `completed`,
configuration, input manifests, history, validation predictions, best weights,
both recovery checkpoints and hashes. No raw clips or full input arrays are zipped.

## Local verification and limits

The official CLI starts locally and its push/status/output options were checked.
The setup script passes `bash -n`. Tests cover shard coverage/space budgeting,
private submission metadata/dependencies, safe credential parsing, source hashes,
missing/reordered input rejection and read-only checkpoint handoff. Notebook
schema and source/config embedding are checked. A synthetic integration executes
the actual CPU preparation and training/export cells, then continues from a copied
saved output. These CPU checks are not Kaggle or ERDES performance results.

No authenticated Kaggle submission, remote CPU preparation or GPU training has
been performed by the agent yet; the account token/eligibility are user setup
steps. Course guidance was checked in `docs/course-guideline/`; the
`docs/course-guidance/` path named by AGENTS.md remains absent.


Verification completed: **55 unit tests passed**, both notebook schemas validated,
actual cached official metadata/splits joined to 3,873/431 rows, and the synthetic
CPU-shard → training → export → continuation integration passed. Official CLI
2.2.5 parsed all generated bundles and built expected private requests with its
network client mocked; this is not a remote submission. `git diff --check` and
`bash -n scripts/setup_kaggle.sh` passed. Shellcheck was unavailable. The interactive
wizard was intentionally not run by the agent because it requires the user's
browser and hidden credential entry.

Commands used for checks:

```bash
uv --cache-dir outputs/.pilot-uv-cache run --offline --no-project --with torch --with pandas --with nbformat --with opencv-python-headless --with ipython --with scikit-learn --with huggingface-hub --index https://download.pytorch.org/whl/cpu python -m unittest discover -s tests -v
# In the same temporary CPU environment:
python /tmp/verify_kaggle_workflow.py
# Isolated official CLI environment, network client mocked by the script:
uv --cache-dir outputs/.kaggle-uv-cache run --offline --no-project --with kaggle --with six python /tmp/verify_kaggle_submission.py
bash -n scripts/setup_kaggle.sh
git diff --check
```

Synthetic integration evidence is ignored under
`outputs/unetpp_verification/kaggle_workflow_check.json`. Submission examples were
built under ignored `outputs/kaggle_verification/`; no credential is embedded.


## 2026-11-02 — Missing preparation inputs during first GPU attempt

The user's GPU job now reported a Tesla T4 and CUDA 12.8, then stopped with
`Attach exactly two completed CPU preparation outputs; found 0 manifests`.
Authenticated inspection confirmed both CPU jobs had failed at the first Zenodo
metadata download (`URLError` after five attempts), before preparing any videos.
Their metadata requested Internet access; the captured error omits its underlying
network reason, so the exact connection cause is not established. Account phone
verification was completed after the initial submissions.

Both CPU jobs were resubmitted as version 2 with Internet requested and GPU
disabled. One was observed running; download progress has not yet been confirmed.
Kaggle subsequently returned HTTP 429 for output/log retrieval; automated polling
was stopped. Do not resubmit the GPU job until both CPU retries complete and save
their manifests. If preparation repeats the network failure, inspect the latest
CPU log and Internet setting before another retry.

The wrapper now blocks a GPU `kernels push` unless all notebook inputs report
`KernelWorkerStatus.COMPLETE`. Two regression tests first failed on the original
wrapper, then passed with the guard. This prevents the observed failed-dependency
submission pattern; it does not claim to fix or bypass a remote network outage.

Validation after the submission guard: all **57 unit tests passed** using the
existing temporary uv CPU environment (`python -m unittest discover -s tests -v`),
and `git diff --check` passed. Remote CPU retry completion and GPU training
remain unverified; no GPU retry was submitted during this diagnosis.


## After completed training: official-test evaluation

All 50 epochs and official-test evaluation are now complete. The selected
epoch-48 checkpoint and threshold 0.5 were frozen before the evaluation workflow
read the test CSV; earlier integrity audits had inspected test membership/counts.
Verified test accuracy is 97.49%, sensitivity 86.00% and specificity 98.67%.
The saved result ZIP is already downloaded under
`outputs/kaggle_results/test_evaluation/`; all 12 artifact hashes and 1,077
prediction rows/metrics passed independent checks. See
[UNet++ results and the paper comparison](UNETPP_RESULTS.md).

`unetpp-rd-002-test-inputs/1` completed CPU preparation with Internet enabled,
verifying the final checkpoint and preparing all 1,077 official test clips
(about 6.78 GB float32). `unetpp-rd-002-test-evaluation/1` then completed GPU
inference. Do not submit these jobs again for the completed result. The commands
below are retained as execution documentation:

```bash
python3 scripts/prepare_unetpp_evaluation_job.py evaluate
python3 scripts/kaggle_cli.py kernels push -p outputs/kaggle_jobs/unetpp-rd-002-test-evaluation --accelerator NvidiaTeslaT4
```

The GPU notebook attaches final training version 1 and test preparation version 1,
checks the exact frozen model SHA and input order/hashes, and performs inference
without an optimizer. Do not rerun training to change checkpoint selection after
test access. Preserve the one test-evaluation output and compare test to test.
The existing CLI guard checks version-pinned dependencies before GPU submission.


For frozen GPU inference, the bundle generator also reads
`configs/unetpp_rd_002_runtime.json` and requests the exact training container
with `docker_image_pinning_type=original`. This avoids silently upgrading the
GPU runtime after training; the notebook retains its runtime equality check.
The CPU test-preparation job already completed and saved all 1,077 clips.
