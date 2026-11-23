# Show the Grad-CAM results in class

The Kaggle run completed for both models and all eight fixed validation clips.
You can present saved results without running Python, Kaggle or inference live.

The presentation layout was refreshed on 2026-11-18. On desktop viewports at
least 1000 CSS pixels wide and 600 high, the clip/view selectors, navigation,
both complete model images, and their predictions fit together in one screen.
Images scale to the available height without cropping. Smaller or heavily zoomed
windows use a normal scrolling layout so controls and text remain usable.

Offline headless Chromium checks passed for all eight clips, four views per clip,
and 64 media loads, with zero JavaScript errors. All four views fit without page
scrolling at 1024×600, 1024×768, 1280×720, 1366×768, 1440×900 and 1920×1080.
Previous/Next, navigation boundaries, keyboard shortcuts, full-screen toggling,
Notes/checks panels and missing-image recovery also passed. Five smaller/shorter
viewports, including 320- and 390-pixel widths, had no horizontal overflow.
Screenshots, the browser check and JSON evidence are under ignored
`outputs/presentation/demo_checks/layout_20261118/`. These are emulated Chromium
checks; physical projectors, touch gestures, Firefox and Safari were not tested.

The rebuilt portable ZIP was checked against all 67 generated demo files. All
64 media files, embedded result/control data and source provenance are unchanged.

## Open the demo

1. Find `outputs/presentation/gradcam_demo.zip` in your project folder.
2. Copy it to the presentation laptop and **extract the entire ZIP**.
3. Open `gradcam_demo/index.html` in Chrome, Edge or Firefox. Keep `assets/`
   and `speaker-notes.md` beside the HTML file.
4. Test it with Wi-Fi off. Choose **Full screen**, press **F** outside a selector,
   or use the browser's shortcut. The button changes to **Exit full screen**.
5. Use the clip selector and View selector. Animations, original/overlay panels
   and temporal curves are available for the same clip in both models.
6. Use **Previous / Next** or the **Left / Right arrow keys** outside selectors
   to change clips. The case counter shows your position in the fixed panel.
7. Open **Notes** or **Explanation checks** when needed. On desktop they open
   above their controls without moving the comparison; click the same heading
   again to close the panel before showing the clips. Only one panel opens at
   a time. On smaller screens the expanded content remains in the page flow.

This is a viewer for saved evidence. It does not perform a fresh prediction when
you change a selection. Show the method rather than code during your talk.

## A three-minute demonstration

**Introduce the question:** “We compared classification performance, then added
3D Grad-CAM to inspect which spatial and temporal regions contribute to each
model's output.”

**Show clip `405744_00014` first.** Its dataset label is RD. At the fixed 0.5
threshold, U-Net++ misses it (RD score **0.64%**); R(2+1)D-18 detects it (**61.11%**).
Switch to “Original + both targets” to show the original frames and separate
positive RD/non-RD attribution maps. Say: “We can compare the predictions and
inspect their attribution patterns. The maps do not confirm anatomical reasoning.”

**Then select `102769_00005`.** The dataset label is also RD, but the outcome
reverses: U-Net++ detects it (**82.77%**) and R(2+1)D-18 misses it (**7.92%**).
Say: “Improved overall sensitivity does not guarantee improvement on every clip.”

**Open the explanation checks on that second clip.** Masking high-attribution
regions reduces R(2+1)D-18's RD logit more than masking low-attribution regions;
U-Net++ does not show that expected pattern. Say: “We tested the visualizations
with input occlusion and weight randomization. The checks were mixed, so we
report their limitations rather than relying on attractive heatmaps.”

**Close:** “The checkpoints stayed frozen. We used fixed validation examples,
included successes and failures, and kept explainability separate from final
test selection. Anatomical localization needs additional expert evidence.”

## Explain the display simply

- **Dataset label:** the clip's reference label, RD or non-RD.
- **Prediction:** the model's decision at threshold 0.5.
- **RD score:** the sigmoid model output; not calibrated clinical risk.
- **RD +z:** relative positive attribution to the RD logit.
- **Non-RD −z:** relative positive attribution to the opposite target.
- **Brighter yellow:** stronger relative attribution within that particular map.
  Separate normalization prevents comparing color brightness across models.
- **Blank/constant map:** no useful positive localization for that target; it
  does not mean the input is medically normal.
- **Occlusion drop:** how the RD logit changes when a specified region is masked.
  Positive means masking reduced the score. This is an artificial intervention.

## Slide backup and likely questions

Keep one overall benchmark-results slide, one Grad-CAM comparison image and one
limitations/control slide. Panels in the demo's `assets/` folder are suitable as
backups; the original matched panels are in the verified results folder.

**Did you retrain for this?** No. U-Net++ epoch 48 and R(2+1)D-18 epoch 15 were
already selected. Explanations use gradients without weight updates.

**Are these segmentation masks?** No. Grad-CAM provides coarse score attribution.
There are no verified anatomical masks supporting localization accuracy here.

**Why only eight examples?** Two per U-Net++ outcome, sorted by clip ID, were
fixed before seeing maps. This illustrates successes and errors; it does not
estimate population performance.

**Does a better heatmap prove a better model?** No. Performance comes from the
separate benchmark evaluation. Attribution and sensitivity controls provide
additional diagnostics, with mixed findings.

**Is the animation the original video?** It shows sampled, resampled inputs
with attribution overlays. Playback speed does not reproduce acquisition timing.

## Rebuild the portable viewer

```bash
uv --cache-dir /tmp/dmml-uv-cache run python scripts/build_explainability_demo.py --archive outputs/kaggle_results/gradcam_validation_001/gradcam_validation_001.zip --output outputs/presentation/gradcam_demo
```

Use a new output directory if it already exists. The builder verifies all source
artifact hashes, copies the presentation media, embeds result/control records,
and writes the portable ZIP. Large generated files stay under ignored outputs.

For this refresh the builder ran with output
`/tmp/rd-gradcam-layout-v2/gradcam_demo`. After browser checks and byte-level
verification, the updated HTML, speaking script and ZIP replaced the corresponding
files under `outputs/presentation/`. The previous viewer and ZIP are retained
locally under `/tmp/rd-gradcam-before-layout/` for this session.

Additional commands run: `uv --cache-dir /tmp/dmml-uv-cache run python -m
py_compile scripts/build_explainability_demo.py`, `git diff --check`, and the
Impeccable design detector (no findings). The ad hoc offline browser command used
`uv run --offline --with playwright --with pillow --with pymupdf python
/tmp/rd_gradcam_layout_check.py`, with the existing browser-library directory in
`LD_LIBRARY_PATH`; the executed script is retained with the screenshots. No
training, inference, model changes or ML test-suite run was needed for this
viewer-only change.
