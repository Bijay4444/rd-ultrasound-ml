# Final model integration workflow

Prepared 2026-11-22. Work from `feature/final-model-integration`, targeting `dev`.
This branch preserves the completed local augmentation/explanation work and
integrates the verified R3D-18 implementation without updating remote `main` or
`dev` directly.

Local integration is complete: `3f7f809` saves the prior work and merge commit
`dd114fb` imports the peer branch with its history. All 22 imported tests pass.
GitHub browser sign-in is now complete and the integration branch has been
pushed. [PR #19](https://github.com/Bijay4444/rd-ultrasound-ml/pull/19) targets
`dev`. The R3D explanation job, saved-output audit and real three-model offline
browser checks are complete. The final model's adapter/notebook is committed
separately in `dc52291`; all 22 explanation/viewer tests passed. The default
demo and portable ZIP now contain all three models, with the previous viewer
preserved. See [the execution report](R3D18_EXPLAINABILITY_RESULTS.md).
`main` is reserved for Bijay's later promotion of the checked `dev` branch.

## Order of work

1. Save the existing work on a feature branch. Commit `3f7f809` records the
   Grad-CAM implementation/demo, completed augmentation experiment, frozen
   evaluation and four-model comparison. Only small source, configs, cleared
   notebooks, tests and reports were staged. Data, weights, logs, credentials
   and generated outputs remain ignored.
2. Push that branch as a remote backup. The first attempt failed with
   `could not read Username for 'https://github.com': No such device or address`.
   No GitHub credentials/helper, token environment or SSH setup was found in this
   WSL environment. A commit is saved locally; it is not yet a GitHub backup.
3. Merge `origin/feature/r3d18-training-eval` into the integration branch. This
   contains the earlier `feature/r3d18-rd` branch; do not merge both separately.
   The automatic merge was clean, including both additions to `tests/README.md`.
   Peer commit history and contributions are preserved.
4. Run the imported model/evaluation tests and source/freeze checks before
   committing the merge. Keep model code, selected epoch 18, threshold 0.5 and
   frozen source hashes unchanged.
5. Push the checked integration branch and open a pull request **into `dev`**.
   Review the combined source and reports there. A PR merge into `dev` and any
   later release merge into `main` are distinct from this local integration.
6. Add R3D-18 to the validation Grad-CAM delivery on the integration branch,
   using the same eight fixed validation examples and its frozen checkpoint.
   Commit/test the adapter and notebook separately from the model import.
7. Execute and verify the new explanation output, then update the presentation
   demo with its third model. Keep the completed two-model output as provenance.
8. Return to augmentation interpretation and final reporting. Its 50-epoch run
   and single test evaluation are already complete; no retraining or new test
   evaluation is required to resume that discussion.

## Authentication and the exact push

The `user.name` / `user.email` Git settings identify commit authors; they do not
authenticate pushes. This session initially had no `gh` executable. An official
GitHub CLI 2.105.0 binary has now been downloaded under ignored
`outputs/github-auth-tools/2.105.0/gh` and checked against the release checksums.
No system package was installed. Its release URL and checksum are retained in
`provenance.json` beside it.

An ephemeral [wizard](../../outputs/github_auth_wizard.sh) guides the user through
three actions: browser login, configuring Git's credential helper, and checking
the signed-in account's repository permission:

```bash
bash outputs/github_auth_wizard.sh
```

The user runs this interactively; the assistant did not run the browser/auth
flow or capture a token. GitHub CLI manages credential storage in the user's
normal credential store/config. No credential is written into the repository's
source or project dotenv files. Keep the prepared CLI binary available while it
is configured as Git's helper. The wizard does not push or merge branches.
`bash -n` passed; `shellcheck` is unavailable. Commands follow GitHub's
[browser login](https://cli.github.com/manual/gh_auth_login) and
[credential helper](https://cli.github.com/manual/gh_auth_setup-git) documentation.

After authentication is available, from the repository:

```bash
git status
git branch --show-current
git push -u origin feature/final-model-integration
```

No force push is needed. If push is rejected because the remote branch changed,
fetch and inspect that history before reconciling it; do not overwrite it.
Create the PR with base `dev` and compare `feature/final-model-integration`:
[Open GitHub comparison](https://github.com/Bijay4444/rd-ultrasound-ml/compare/dev...feature/final-model-integration).
This link becomes usable after the first successful push.

## R3D-18 explanation integration requirements

- Reuse `R3D18RDClassifier` and the exact checked
  `configs/r3d18_rd_001_evaluation.json`; do not select a different epoch.
- Feature layer `backbone.layer4`, head `backbone.fc`; explain scalar RD logit
  `z` and non-RD evidence `-z` in FP32 at the existing full input shape.
- Attach `namanbabariya/r3d18-rd-001-train-03/1` plus the original training/
  validation cache. Test-preparation inputs are unnecessary.
- Retain validation prediction parity, cache/source/checkpoint checks, state
  preservation, occlusion/randomization controls and artifact hash verification.
- Use the same fixed eight-case manifest; do not pick cases after viewing maps.
  A separate output/job identity preserves `gradcam_validation_001` and prevents
  mixing historical two-model artifacts with the new result.
- Update the demo deliberately for three models at presentation resolutions;
  preserve the working clip controls and fullscreen behavior.

Augmented U-Net++ explanations, if desired later, are a separate expansion;
they are not needed to import the last architecture or finish the three-model
Grad-CAM comparison.

## Evidence and checks

Remote refs were refreshed with `git fetch origin`; local work was saved with
`git switch -c` / explicit `git add` / `git commit`. The attempted feature push
failed before authentication. Integration used
`git merge --no-ff --no-commit origin/feature/r3d18-training-eval` for review.

The existing 14 explanation tests passed before the merge. Imported R3D-18
training/evaluation tests run with the existing CPU uv environment and the
pattern `*r3d18*.py`. The first run passed 21 checks but failed while writing a
checkpoint in the full-model resume check: this environment's `/tmp` is a 1.9 GB
tmpfs, smaller than the two runs' simultaneous optimizer checkpoints. The tests
were rerun unchanged with disk-backed temporary storage:

```bash
mkdir -p outputs/integration_test_tmp
TMPDIR="$PWD/outputs/integration_test_tmp" <documented-uv-prefix> python -m unittest discover -s tests -p '*r3d18*.py' -v
```

The unchanged rerun passed **all 22 R3D-18 tests** in 325.57 seconds. The uv prefix
is recorded in [the augmentation report](UNETPP_AUGMENTATION_RESULTS.md).
Temporary test artifacts stay ignored; no training implementation was changed
to accommodate the local storage limit. Exact merged freeze and frozen training hashes match the
previously downloaded and audited Kaggle snapshot. `git diff --cached --check`
passed, as did compilation of the imported Python files. The full unrelated
repository suite was not rerun. The subsequent Grad-CAM extension completed
eight fixed validation cases; no new training or official-test inference ran.
The default viewer passed real offline Chromium checks on all 24 model/case
pairs and four views. Promotion uses `dev`; `main` remains the user's later step.
