"""Build a portable offline classroom viewer from a verified explanation ZIP.

No training, inference, network requests or third-party Python packages.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retinal ultrasound · Model explanations</title>
<style>
:root{color-scheme:light;--ink:#182e36;--muted:#49626b;--line:#c7d4d7;--accent:#075b70}
*{box-sizing:border-box}body{margin:0;background:#f4f7f7;color:var(--ink);font:16px/1.4 system-ui,sans-serif}
main{max-width:1760px;margin:auto;padding:16px 24px}header{border-bottom:1px solid var(--line);padding-bottom:10px}
h1{font-size:clamp(22px,2.2vw,30px);line-height:1.2;margin:0 0 4px;text-wrap:balance}h2{font-size:clamp(18px,1.6vw,24px);line-height:1.25;margin:0}p{max-width:74ch;margin:8px 0}
.muted{color:var(--muted);font-size:14px;max-width:none;margin:0}.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:14px 0}
label{display:grid;gap:4px;font-size:14px;font-weight:650;min-width:0}select,button{font:inherit;color:var(--ink);background:white;border:1px solid #829ba3;border-radius:6px;padding:8px 10px;min-height:44px}select{width:100%;min-width:0}button{white-space:nowrap;cursor:pointer}
button:hover{background:#dbe9ed}button:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible,.details-body:focus-visible{outline:3px solid var(--accent);outline-offset:3px}button:disabled{color:var(--muted);background:#e8eeee;cursor:default}
#fullscreen{background:#dbe9ed}.case-info{margin:0 0 12px}.case-heading{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:4px 16px}#truth{font-size:17px;font-weight:650;margin:0}.case-meta{font-size:14px;color:var(--muted);margin:0;font-variant-numeric:tabular-nums}
#talk{max-width:none;margin:4px 0 0;color:var(--muted);font-size:14px}
.comparison{display:grid;grid-template-columns:repeat(var(--model-count,2),minmax(0,1fr));gap:16px;min-height:0}.model{background:white;padding:12px;min-width:0;min-height:0}.prediction{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:2px 12px;margin-top:5px}.outcome{font-size:16px;font-weight:650;margin:0}.meta{font-size:15px;margin:0;font-variant-numeric:tabular-nums}.media{min-width:0;min-height:0}.model img{display:block;width:100%;height:auto}.caption{font-size:14px;color:var(--muted);max-width:none;margin:10px 0}
.support{position:relative;display:flex;flex-wrap:wrap;align-items:start;gap:8px 24px;border-top:1px solid var(--line);padding-top:4px}details{min-width:0}details[open]{flex-basis:100%}summary{cursor:pointer;font-size:14px;font-weight:650;min-height:44px;align-content:center}.details-body{padding:0 4px 8px;font-size:15px;scrollbar-color:#829ba3 #e8eeee;scrollbar-width:thin}.details-body p{margin:6px 0 10px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:left;padding:6px;border-bottom:1px solid var(--line)}
a{color:var(--accent);text-underline-offset:3px}::selection{background:#c1dce4;color:#122d35}footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 16px;font-size:14px;color:var(--muted);margin-top:8px}#error{color:#9c2020;max-width:none}#error:empty{display:none}
@media(min-width:1000px) and (min-height:600px){
 main{height:100vh;height:100dvh;display:grid;grid-template-rows:auto auto auto minmax(0,1fr) auto auto auto;gap:8px;padding:12px 20px}
 .toolbar{display:grid;grid-template-columns:minmax(220px,1fr) minmax(270px,1.2fr) auto auto auto;margin:0;gap:10px}
 .case-info{margin:0}.model{display:grid;grid-template-rows:auto auto minmax(0,1fr);gap:4px}.prediction{margin:0}
 .model img{height:100%;object-fit:contain}.caption{margin:0}.support details[open]{flex-basis:auto}
 .details-body{position:absolute;bottom:calc(100% + 8px);left:0;right:0;z-index:2;max-height:55vh;max-height:55dvh;overflow:auto;overscroll-behavior:contain;background:white;padding:16px 20px;border:1px solid var(--line);box-shadow:0 8px 24px rgb(24 46 54 / 15%)}footer{margin:0}
}
@media(max-width:999px){main{padding:16px}.toolbar label{flex:1 1 260px}.support{margin-top:12px}footer .shortcuts{display:none}}
@media(max-width:640px){main{padding:14px 12px}.comparison{grid-template-columns:1fr}.toolbar{gap:8px}.toolbar label{flex-basis:100%}.toolbar button{flex:1}.prediction{display:block}.meta{margin-top:2px}.case-heading{display:block}.case-meta{margin-top:4px}table{font-size:13px}th,td{padding:6px 3px}}
@media print{.toolbar,.shortcuts{display:none}body{background:white}main{display:block;height:auto;padding:0}.comparison{gap:12px}.model{display:block}.model img{height:auto}.details-body{max-height:none;overflow:visible}details{display:block}}
</style>
<main><header><h1>What contributes to the model’s RD score?</h1>
<p class="muted">Frozen checkpoints · Eight fixed validation clips · Educational research</p></header>
<div class="toolbar"><label>Validation clip<select id="case"></select></label>
<label>View<select id="view"><option value="rd_overlay.gif">RD evidence · animation (+z)</option><option value="non_rd_overlay.gif">Non-RD evidence · animation (−z)</option><option value="panel.png">Original + both targets · still frames</option><option value="temporal.png">Attribution over time</option></select></label>
<button id="previous" type="button" aria-label="Previous case" aria-keyshortcuts="ArrowLeft">Previous</button><button id="next" type="button" aria-label="Next case" aria-keyshortcuts="ArrowRight">Next</button><button id="fullscreen" type="button" aria-keyshortcuts="f" aria-pressed="false">Full screen</button></div>
<section class="case-info" aria-label="Selected validation clip"><div class="case-heading"><p id="truth" aria-live="polite"></p><p class="case-meta"><span id="position"></span> · Decision threshold 50%</p></div><p id="talk"></p></section>
<div class="comparison" id="comparison" aria-label="Model explanations for the same clip"></div>
<p class="caption">Yellow means stronger relative positive attribution within each map. Maps are normalized separately; colors cannot rank models. Playback uses sampled, resampled indices, not original timing.</p>
<aside class="support" aria-label="Supporting information"><details id="checks" name="support"><summary>Explanation checks</summary><div class="details-body" tabindex="0" role="region" aria-label="Occlusion and randomization checks">
<p>Mean change in the RD logit after masking one region at a time. Positive means the RD score decreased. High-attribution regions do not always produce the largest decrease.</p>
<table><thead><tr><th>Model</th><th>High</th><th>Low</th><th>Random</th></tr></thead><tbody id="occlusion"></tbody></table>
<p id="constant" class="caption"></p>
<p>Map correlation after cumulative weight randomization (head; head + final block; all learned layers). Correlation is undefined for constant maps. These are diagnostic controls, not a certificate of explanation correctness.</p>
<table><thead><tr><th>Model</th><th>Head</th><th>Head + block</th><th>All layers</th></tr></thead><tbody id="randomization"></tbody></table></div></details>
<details id="notes" name="support"><summary>Notes</summary><div class="details-body" tabindex="0" role="region" aria-label="Notes">
<p>Start with clip 405744_00014: one model misses an RD case and the other detects it. Then show 102769_00005: the reverse occurs. Open the checks and discuss mixed results. Finish with the limitations below.</p>
<p>Grad-CAM is a coarse attribution map for a chosen model score. It is not a retinal mask, clinical diagnosis, calibrated risk estimate or proof of correct anatomical reasoning. The eight examples were selected by baseline outcome and clip ID, not by heatmap appearance; they do not estimate overall performance.</p>
<p>Animations are saved results. Selecting a clip changes the displayed evidence; it does not run inference. Use the separate benchmark table when discussing model performance.</p></div></details></aside>
<footer><span>Offline viewer · <a href="speaker-notes.md">Speaking script and setup</a></span><span class="shortcuts">Left / Right: change case · F: full screen</span><p id="error" role="alert"></p></footer></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const select=document.getElementById('case'),view=document.getElementById('view');
const names=data.model_labels;
const meanings={TP:'Correct RD detection',TN:'Correct non-RD prediction',FP:'False alarm',FN:'Missed RD'};
for(const c of data.cases){const o=document.createElement('option');o.value=c.clip_id;o.textContent=c.clip_id+' · baseline '+c.baseline_outcome;select.append(o)}
if(data.cases.some(c=>c.clip_id==='405744_00014'))select.value='405744_00014';
function update(){const c=data.cases.find(c=>c.clip_id===select.value),rows=data.models.map(m=>data.results.find(r=>r.clip_id===c.clip_id&&r.model===m));
document.getElementById('truth').textContent='Dataset label: '+(c.label?'RD':'non-RD')+' · Clip '+c.clip_id;
document.getElementById('position').textContent='Case '+(select.selectedIndex+1)+' of '+select.options.length;
document.getElementById('talk').textContent=data.notes[c.clip_id]||'Compare the same clip across models. Describe the prediction first, then the regions contributing to the selected score. Avoid inferring anatomical correctness from the heatmap.';
document.getElementById('error').textContent='';const container=document.getElementById('comparison');container.replaceChildren();container.style.setProperty('--model-count',rows.length);
rows.forEach(r=>{const box=document.createElement('section');box.className='model';const heading=document.createElement('h2');heading.id='heading-'+r.model;heading.textContent=names[r.model];box.setAttribute('aria-labelledby',heading.id);const prediction=document.createElement('div');prediction.className='prediction';const desc=document.createElement('p');desc.className='outcome';desc.textContent=meanings[r.outcome];const meta=document.createElement('p');meta.className='meta';meta.textContent='RD score '+(100*r.fp32_probability).toFixed(2)+'%';prediction.append(desc,meta);const media=document.createElement('div');media.className='media';const img=document.createElement('img');img.src='assets/'+r.model+'/'+c.clip_id+'/'+view.value;img.alt=names[r.model]+' '+view.options[view.selectedIndex].text+' for '+c.clip_id;img.onerror=()=>{document.getElementById('error').textContent='An image is missing. Extract the entire demo ZIP and keep index.html beside the assets folder.'};media.append(img);box.append(heading,prediction,media);container.append(box)});
const oc=document.getElementById('occlusion'),rand=document.getElementById('randomization');oc.replaceChildren();rand.replaceChildren();const constants=[];
data.models.forEach(m=>{const record=data.controls[m+'/'+c.clip_id],r=document.createElement('tr');for(const value of [names[m],...['high','low','random'].map(g=>record.occlusion.groups[g].mean_drop.toFixed(4))]){const td=document.createElement('td');td.textContent=value;r.append(td)}oc.append(r);
const rr=document.createElement('tr');for(const value of [names[m],...record.randomization.map(v=>v.similarity.pearson===null?'Undefined':v.similarity.pearson.toFixed(3))]){const td=document.createElement('td');td.textContent=value;rr.append(td)}rand.append(rr);if(record.occlusion.constant_map)constants.push(names[m])});
document.getElementById('constant').textContent=constants.length?'Constant RD map: '+constants.join('; ')+'. High/low ranking is uninformative for these maps.':'';
document.getElementById('previous').disabled=select.selectedIndex===0;document.getElementById('next').disabled=select.selectedIndex===select.options.length-1;
}select.addEventListener('change',update);view.addEventListener('change',update);
function moveCase(step){const index=select.selectedIndex+step;if(index>=0&&index<select.options.length){select.selectedIndex=index;update()}}
document.getElementById('previous').onclick=()=>moveCase(-1);document.getElementById('next').onclick=()=>moveCase(1);
const fullscreen=document.getElementById('fullscreen');
fullscreen.onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else if(document.documentElement.requestFullscreen)await document.documentElement.requestFullscreen();else throw new Error('Full screen unavailable')}catch{document.getElementById('error').textContent='Full screen is unavailable here. Use your browser’s full-screen shortcut.'}};
document.addEventListener('fullscreenchange',()=>{fullscreen.textContent=document.fullscreenElement?'Exit full screen':'Full screen';fullscreen.setAttribute('aria-pressed',String(Boolean(document.fullscreenElement)))});
document.addEventListener('keydown',event=>{if(event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||event.target.closest('select,input,textarea,summary,a,[contenteditable]'))return;if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();moveCase(event.key==='ArrowLeft'?-1:1)}else if(event.key.toLowerCase()==='f'&&!event.repeat){event.preventDefault();fullscreen.click()}});
update();
</script></html>'''


NOTES = '''# Classroom demonstration — about three minutes

Extract the whole demo ZIP. Open index.html in Chrome, Edge or Firefox.
Keep its assets folder beside it. Test once with Wi-Fi off. Use browser full screen.
On a desktop or projector, the controls and both model views fit in one screen.
Use Left/Right to change cases and F to toggle full screen outside selectors.
Notes and Explanation checks open above their controls without moving the
comparison. Close the panel before showing the clips. On phones and short
windows, the page scrolls normally.
The page displays saved results; you do not need Kaggle, Python or a GPU in class.

1. Introduce the method (20 seconds): "We added 3D Grad-CAM to investigate which
   spatial and temporal regions contribute to each frozen classifier's score.
   These are attribution maps, not segmentation masks."
2. Start on 405744_00014 (45 seconds). The dataset label is RD. U-Net++ scores
   it about 0.64% and misses it; R(2+1)D-18 scores it about 61.11% and detects it
   at the fixed 50% threshold. Switch to the still-frame view to show the original
   input above the positive and negative target maps. Say: "These maps help us
   inspect the different predictions; they do not prove why one model is better."
3. Select 102769_00005 (30 seconds). The label is also RD. Here U-Net++ detects
   it (82.77%), while R(2+1)D-18 misses it (7.92%). Say: "Better aggregate
   sensitivity does not mean the model is correct on every individual clip."
4. Open the controls (45 seconds). On 102769_00005, masking high-attribution
   regions decreases the R(2+1)D-18 RD logit more than masking low-attribution
   regions. U-Net++ does not show this expected pattern on the same clip.
   Say: "Our sensitivity checks are mixed; visual appeal alone is insufficient."
5. Close (20 seconds): "We used fixed validation examples, kept checkpoints
   unchanged, and included successes and failures. Clinical localization and
   anatomical correctness require further evidence and expert annotations."

Use the existing benchmark results slide for overall performance. The eight-case
panel is outcome-stratified and cannot estimate performance. RD scores are model
outputs rather than calibrated clinical probabilities. Do not compare color
brightness across models: maps use separate normalization. Sampled GIF timing
is not the original ultrasound recording speed.

Backup: put the PNG panels from assets into your slide deck. If the HTML cannot
open, show those images and use the same script. No live inference is needed.
'''


def verified_archive(archive):
    """Stream-check every member before reading records or copying display assets."""
    names = archive.namelist()
    if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
        raise ValueError('Unsafe or duplicate archive members')
    manifests = [n for n in names if n.count('/') == 1 and n.endswith('/artifact_sha256.json')]
    if len(manifests) != 1:
        raise ValueError('Require one explanation artifact manifest')
    prefix = manifests[0].rsplit('/', 1)[0] + '/'
    hashes = json.loads(archive.read(manifests[0]))
    if set(names) != {prefix+n for n in hashes} | {prefix+'artifact_sha256.json'}:
        raise ValueError('Archive membership differs from manifest')
    for name, digest in hashes.items():
        h = hashlib.sha256()
        with archive.open(prefix+name) as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
        if h.hexdigest() != digest:
            raise ValueError('Checksum differs: '+name)
    return prefix, hashes


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def build(archive_path, output, additional_archives=None):
    output = Path(output)
    sources = [Path(archive_path), *map(Path, additional_archives or [])]
    cases = None
    results, models, controls, evidence, labels, locations = [], [], {}, [], {}, {}
    known = {'unetpp':'U-Net++', 'r2plus1d18':'R(2+1)D-18', 'r3d18':'R3D-18'}
    settings = algorithm = None
    with ExitStack() as stack:
        for path in sources:
            archive = stack.enter_context(zipfile.ZipFile(path))
            prefix, hashes = verified_archive(archive)
            def read(name): return json.loads(archive.read(prefix+name))
            selection, status = read('selection_manifest.json'), read('status.json')
            selected = selection['cases']
            current = list(status['models'])
            if (status['status'] != 'completed' or selection['partition'] != 'validation'
                    or len(selected) != 8 or len({c['clip_id'] for c in selected}) != 8
                    or not current or set(current) - set(known) or set(current) & set(models)
                    or any(status['models'][m]['status'] != 'completed' for m in current)):
                raise ValueError('Require completed, distinct models on eight validation cases')
            pairs = {(m,c['clip_id']) for m in current for c in selected}
            rows = read('case_results.json')
            if (len(rows) != len(pairs) or {(r['model'],r['clip_id']) for r in rows} != pairs
                    or len(status['cases']) != len(pairs)
                    or {(r['model'],r['clip_id']) for r in status['cases'] if r['status']=='completed'} != pairs):
                raise ValueError('Incomplete or duplicate model/case records')
            method = read('settings.json')
            code = archive.read(prefix+'sources/video_explainability.py')
            if cases is None:
                cases, settings, algorithm = selected, method, code
                if current != ['unetpp','r2plus1d18']:
                    raise ValueError('Start with the historical two-model comparison')
            elif selected != cases or method != settings or code != algorithm:
                raise ValueError('Extension changes fixed cases, attribution method or cache identity')
            for model in current:
                freeze = read(model+'/freeze.json')
                if freeze['threshold'] != .5:
                    raise ValueError('Model threshold differs from the fixed protocol')
                labels[model] = known[model]+' · epoch '+str(freeze['selected_epoch'])
                locations[model] = (archive,prefix)
                for case in selected:
                    key = model+'/'+case['clip_id']
                    controls[key] = {'occlusion':read(key+'/occlusion.json'),
                                     'randomization':read(key+'/randomization.json')}
            results.extend(rows)
            models.extend(current)
            evidence.append({'archive_sha256':file_sha(path),'verified_artifacts':len(hashes),
                             'models':current,'runtime':read('runtime.json')})
        if additional_archives:
            plan = json.loads((Path(__file__).resolve().parents[1]/'configs/r3d18_gradcam_validation_001.json').read_text())
            if (evidence[0]['archive_sha256'] != plan['reference_archive_sha256']
                    or cases != plan['selection']['cases'] or models != ['unetpp','r2plus1d18','r3d18']):
                raise ValueError('Historical archive or extension identity differs from the fixed plan')
        output.mkdir(parents=True, exist_ok=False)
        for model in models:
            archive,prefix = locations[model]
            for case in cases:
                key = model+'/'+case['clip_id']
                for filename in ('panel.png','temporal.png','rd_overlay.gif','non_rd_overlay.gif'):
                    target = output/'assets'/key/filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(prefix+key+'/'+filename))
        data = {'cases':cases,'results':results,'models':models,'controls':controls,'model_labels':labels,
                'notes':{'405744_00014':'U-Net++ misses this RD clip while R(2+1)D-18 detects it. Compare the saved predictions and maps for every displayed model.',
                         '102769_00005':'U-Net++ detects this RD clip while R(2+1)D-18 misses it. Compare each model on the same input; aggregate performance does not guarantee a correct individual prediction.'}}
        encoded = json.dumps(data).replace('<', chr(92)+'u003c')
        (output/'index.html').write_text(TEMPLATE.replace('__DATA__',encoded))
        notes = NOTES.replace('both model views','all model views')
        if 'r3d18' in models:
            notes += '\n## R3D-18 extension: frozen epoch 18\n\n'
            notes += 'The same eight cases were verified before producing the new maps. Earlier model explanations are reused unchanged.\n\n'
            for row in results:
                if row['model']=='r3d18':
                    notes += f"- {row['clip_id']}: RD score {100*row['fp32_probability']:.2f}%, outcome {row['outcome']}.\n"
        (output/'speaker-notes.md').write_text(notes)
        (output/'provenance.json').write_text(json.dumps({'sources':evidence,
            'verified_artifacts':sum(e['verified_artifacts'] for e in evidence),
            'models':models,'cases':len(cases),'reused_historical_explanations':bool(additional_archives)},indent=2)+'\n')
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file(): bundle.write(path,str(path.relative_to(output.parent)))
    return output/'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--additional-archive',type=Path,action='append',default=[])
    args = parser.parse_args()
    print(build(args.archive, args.output, args.additional_archive))
