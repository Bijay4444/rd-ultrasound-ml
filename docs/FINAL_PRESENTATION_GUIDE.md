# Final presentation: comparison, augmentation and explanations

Revised 2026-11-24 following the user’s slide review. **20 slides, no appendix**, planned for
**15 minutes excluding questions**. The reference notes already embedded in the deck remain
available; Bijay will prepare the final personal speaker notes after this revision.

[Open the revised Google Slides presentation](https://docs.google.com/presentation/d/1dG5MA0oW5V0nbI7xk472jOouVE-XNWIBIJn2Q30wurg/edit).

## Requested changes

Numbers in this list refer to the original 25-slide version.

- Removed original slides 17–18, as requested. There is no longer a paper-comparison or contribution-summary segment.
- Integrated the useful settings from original slide 21 into current slides 4–5: shared input shape,
  splits, optimizer, loss, seed, training budget, physical/effective batch and precision. Runtime
  and normalization limitations remain on slide 19. Removed the separate protocol appendix slide.
- Promoted original slide 22 to **current slide 13**, after augmentation findings. It explicitly
  reports all four **frozen checkpoints on clean test inputs** at threshold 0.5.
- Promoted original slide 23 to **current slide 15**, between the Grad-CAM method and its examples.
  It explains occlusion, progressive weight randomization and model-state preservation.
- Removed original slides 24–25. Renumbered all remaining slides and section labels.
- Corrected stale appendix references in the reference notes. Retained existing note bodies
  where possible and reassigned the speaking budgets to keep 900 seconds.

## Validation versus frozen test results

| Current slide | Partition and purpose | Inputs / choices |
| --- | --- | --- |
| 7 | 431 validation clips; used to select checkpoints | Clean inputs; maximum validation accuracy, earliest tie |
| 8 | 1,077 official test clips; three-architecture error counts | Selected frozen checkpoints; threshold 0.5 |
| 12 | Augmentation ablation; validation and reused-test rows explicitly labeled | Baseline versus augmented U-Net++ |
| 13 | 1,077 official test clips; full four-model metrics | Frozen weights; clean inputs, including augmented-training model |

“Frozen” means the chosen weights and threshold were fixed for evaluation. The official test
benchmark was already exposed earlier in the project; keep the term **reused official test
benchmark**. The augmentation model receives no augmentation during evaluation. No new
training, inference or threshold search was performed for this slide revision.

## Speaking outline

This remains a planned time budget; a human timed rehearsal has not been performed.

| Slide | Time | Topic |
| --- | --- | --- |
| 01 | 00:00–00:20 | Retinal Detachment Analysis from Ocular Ultrasound |
| 02 | 00:20–01:00 | A brief recap: classify the whole ultrasound clip |
| 03 | 01:00–01:45 | From one baseline to four completed experiments |
| 04 | 01:45–02:35 | Shared data, training and checkpoint rules |
| 05 | 02:35–03:25 | Three video classifiers, different ways to learn features |
| 06 | 03:25–04:10 | Resumable GPU runs, independently checked results |
| 07 | 04:10–04:55 | Validation selected a checkpoint for every experiment |
| 08 | 04:55–05:55 | Test tradeoff: fewer missed RD versus fewer false alarms |
| 09 | 05:55–06:35 | Six fewer missed RD came with two more false alarms |
| 10 | 06:35–07:20 | Augmentation changes training views, not patient count |
| 11 | 07:20–07:55 | Training previews shaped the policy before the run |
| 12 | 07:55–08:55 | Augmentation produced mixed evidence |
| 13 | 08:55–09:45 | Frozen checkpoints: all four test results |
| 14 | 09:45–10:35 | 3D Grad-CAM attributes a clip score across space and time |
| 15 | 10:35–11:25 | How we checked the Grad-CAM maps |
| 16 | 11:25–12:55 | One RD case: the residual models detect what baseline missed |
| 17 | 12:55–13:35 | A second RD case reverses that model disagreement |
| 18 | 13:35–14:25 | We checked the heatmaps instead of trusting their appearance |
| 19 | 14:25–15:00 | The limits define what we can responsibly claim |
| 20 | Questions | Thank You |

## Grad-CAM demo cues

1. Before presenting, open `outputs/presentation/gradcam_demo/index.html` separately.
   Keep `assets/` alongside it. Preselect `405744_00014`, RD evidence animation and Full screen.
2. Current slide **16** has a 90-second allowance including the optional browser switch.
   The dataset label is RD. Scores are U-Net++ 0.64%, R(2+1)D-18 61.11%, R3D-18 86.80%.
3. Current slide **17** shows the counterexample, `102769_00005`: U-Net++ 82.77%,
   R(2+1)D-18 7.92%, R3D-18 39.13%. Explain that aggregate improvements do not eliminate
   cases detected only by the baseline.
4. Current slide **18** presents the controls. Their counts describe eight fixed validation
   cases; they do not establish general explanation fidelity or anatomical correctness.
5. If switching is slow, use the stills on slides 16–17. Both show resampled position 95
   with independently normalized maps; the talk does not require a live browser demonstration.

“3D” here means time × height × width, not a reconstructed anatomical eye volume.
Scores are not calibrated clinical risks. Animated playback samples resampled positions
and does not reproduce original acquisition timing.

## Files and verification

Latest local backups, all under ignored `outputs/presentation/final_revision_20261124/`:

- `rd_final_presentation_15min.pdf`: final 20-page Google export.
- `rd_final_presentation_15min.pptx`: editable PowerPoint export with reference notes.
- `slide-plan.json`: final ordering, original slide numbers and 900-second allocation.
- `revision-plan.json`: requested removals and integrations.
- `delivery-checks.json`: **102 export/content checks plus 44 revision checks passed**.
- `renders/final/`: final rendered pages and review contact sheets.

Checks included slide order and numbering, all table cells and reference notes, saved test
CSV agreement, PDF pages/titles, five unchanged native tables, four figures with unchanged
geometry/cropping, requested removals and the 15-minute allocation. The local slide checker
reported zero issues. All 20 rendered pages were visually inspected; no further layout
repair was needed for this revision. The previous 25-slide exports remain historical copies.

Commands used the skill’s `parse_template_design_system.mjs`, `check_output_issues.mjs`
and export/render helper, plus isolated uv-managed Node/Python tools and
`check_delivery.py`. The live read used the complete slide resource with explicit fields
after the unrestricted full-presentation read repeatedly reset the tool runtime. Existing
rendered exemplars were reused. PyMuPDF supplied local PDF rendering.

Not run: new ML training/inference, the full repository ML test suite, a separate
PowerPoint-application render, a physical projector test or a human timed rehearsal.
The next step is final personal speaker notes, speaker handoffs and rehearsal.

## Supporting research records

- [Frozen four-model comparison](FINAL_MODEL_COMPARISON.md)
- [Augmentation experiment](UNETPP_AUGMENTATION_RESULTS.md)
- [Original explanation evidence](EXPLAINABILITY_RESULTS.md)
- [R3D-18 explanation extension](R3D18_EXPLAINABILITY_RESULTS.md)
- [Offline demo guide](EXPLAINABILITY_DEMO.md)
