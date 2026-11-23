"""Verified validation-only explanation orchestration and presentation exports."""
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from rd_ultrasound_ml.unetpp_training import autocast_for, file_digest, json_digest
from rd_ultrasound_ml.unetpp_evaluation import find_frozen_training
from rd_ultrasound_ml.comparison_evaluation import find_comparison_training
from rd_ultrasound_ml.unetpp_kaggle import load_prepared_shards
from rd_ultrasound_ml.video_cache import load_cache_for_run, read_preparation_config
from rd_ultrasound_ml.video_explainability import (
    assert_state, gradcam, occlusion, outcome, randomized_maps, select_cases, snapshot,
)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def render_case(directory, inputs, maps, title):
    # Plotting is optional for core attribution tests, required for the export job.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    gray = inputs.detach().cpu().numpy()[0, 0]
    positions = [0, 19, 38, 57, 76, 95]
    fig, axes = plt.subplots(3, 6, figsize=(16, 9))
    for col, position in enumerate(positions):
        for row in range(3):
            axes[row, col].imshow(gray[position], cmap='gray', vmin=0, vmax=1)
            if row:
                axes[row, col].imshow(maps[row-1]['normalized'][position], cmap='inferno', vmin=0, vmax=1, alpha=.45)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'{["Original", "RD +z", "Non-RD -z"][row]} | index {position}')
    fig.suptitle(title + '\nClip-normalized attribution; not a segmentation mask', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(directory/'panel.png', dpi=140); plt.close(fig)
    for name, cam in zip(('rd', 'non_rd'), maps):
        images = []
        for position in np.linspace(0, 95, 12).round().astype(int):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(gray[position], cmap='gray', vmin=0, vmax=1)
            ax.imshow(cam['normalized'][position], cmap='inferno', vmin=0, vmax=1, alpha=.45)
            ax.set_title(title + f'\n{name} | resampled index {position}\nSampled playback, not original timing', fontsize=7)
            ax.axis('off'); fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
            plt.close(fig)
        images[0].save(directory/f'{name}_overlay.gif', save_all=True, append_images=images[1:], duration=167, loop=0)
    fig, ax = plt.subplots()
    for name, cam in zip(('RD +z', 'Non-RD -z'), maps):
        ax.plot(cam['normalized'].mean((1, 2)), label=name)
    ax.set(xlabel='Resampled temporal index', ylabel='Mean normalized attribution', title=title)
    ax.legend(); fig.tight_layout(); fig.savefig(directory/'temporal.png'); plt.close(fig)


def prediction_record(directory, validation):
    manifest = pd.read_csv(directory/'validation_manifest.csv')
    if not manifest.equals(validation):
        raise ValueError('Saved validation manifest differs from verified official order')
    record = json.loads((directory/'best_validation_predictions.json').read_text())
    labels, probabilities = np.asarray(record['dataset_labels']), np.asarray(record['rd_probabilities'])
    if (not np.array_equal(labels, validation.label.to_numpy()) or len(probabilities) != len(validation)
            or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Invalid saved validation predictions')
    return probabilities


def run_explanations(input_root, output_root, adapters, sources, device='cuda', runtime=None):
    """Adapters are ordered baseline first; fresh output path prevents mixed artifacts."""
    started = time.perf_counter()
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'cases': [], 'models': {}}
    save_json(output/'status.json', status)
    try:
        if not adapters or adapters[0].name != 'unetpp' or len({a.name for a in adapters}) != len(adapters):
            raise ValueError('Require U-Net++ first and unique model adapter names')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Attach a CUDA GPU; no automatic precision/resolution fallback')
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        save_json(output/'runtime.json', {**(runtime or {}), 'python': platform.python_version(), 'torch': torch.__version__,
                  'numpy': np.__version__, 'device': device,
                  'gpu': torch.cuda.get_device_name() if device == 'cuda' else None})
        preparation, _ = read_preparation_config(input_root)
        train, validation, paths, fingerprint, identity = load_prepared_shards(input_root, preparation)
        directories, configs, predictions = {}, {}, {}
        for adapter in adapters:
            freeze = adapter.freeze
            finder = find_frozen_training if adapter.name == 'unetpp' else find_comparison_training
            directory = finder(input_root, freeze)
            config = json.loads((directory/'config.json').read_text())
            for module, digest in freeze['training_implementation_sha256'].items():
                if hashlib.sha256(sources[module].encode()).hexdigest() != digest:
                    raise ValueError('Embedded frozen source differs: ' + module)
            if adapter.name == 'unetpp':
                expected = freeze['training_data_fingerprint']
                if fingerprint != expected or json_digest(preparation) != json_digest(config):
                    raise ValueError('Baseline preparation identity differs')
            else:
                _, val2, _, run_fingerprint, record = load_cache_for_run(input_root, config)
                if (not val2.equals(validation) or run_fingerprint != freeze['training_run_fingerprint']
                        or record['preprocessing_identity'] != identity):
                    raise ValueError('Comparison cache/run identity differs')
            directories[adapter.name], configs[adapter.name] = directory, config
            predictions[adapter.name] = prediction_record(directory, validation)
        cases = select_cases(validation, predictions[adapters[0].name])
        save_json(output/'selection_manifest.json', {'partition': 'validation', 'rule': 'two per baseline outcome, clip ID ascending', 'cases': cases})
        validation.to_csv(output/'validation_manifest.csv', index=False)
        save_json(output/'settings.json', {'precision': 'float32', 'parity_tolerance': .001,
                  'threshold': .5, 'normalization': 'whole volume min-max, constant maps zero',
                  'cuboid': [12, 16, 16], 'occlusions_per_group': 8, 'seed': 42,
                  'preparation_fingerprint': fingerprint, 'preprocessing_identity': identity})
        source_dir = output/'sources'; source_dir.mkdir()
        for name, source in sources.items():
            (source_dir/(name+'.py')).write_text(source)
        rows, control_rows = [], []
        for adapter in adapters:
            directory, freeze = directories[adapter.name], adapter.freeze
            model_dir = output/adapter.name; model_dir.mkdir()
            save_json(model_dir/'freeze.json', freeze)
            save_json(model_dir/'training_config.json', configs[adapter.name])
            payload = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
            expected = freeze.get('training_run_fingerprint', freeze.get('training_data_fingerprint'))
            if (payload['epoch'] != freeze['selected_epoch'] or payload['data_fingerprint'] != expected
                    or json_digest(payload['config']) != freeze['training_config_digest'] or payload['test_used']):
                raise ValueError('Checkpoint payload differs from freeze')
            model = adapter.factory().to(device).eval()
            model.load_state_dict(payload['model_state']); del payload
            before = snapshot(model)
            status['models'][adapter.name] = {'status': 'running', 'layer': adapter.layer}
            # Gate every selected probability before producing this model's explanations.
            parity = []
            for case in cases:
                index = case['validation_index']
                x = torch.from_numpy(np.load(paths[len(train)+index], allow_pickle=False).copy()).unsqueeze(0).to(device)
                with torch.no_grad():
                    with autocast_for(device, freeze['precision']):
                        original = float(model(x).float().sigmoid().item())
                    fp32 = float(model(x).sigmoid().item())
                saved = float(predictions[adapter.name][index])
                parity.append({**case, 'saved_probability': saved, 'original_precision_probability': original,
                               'fp32_probability': fp32, 'original_vs_saved': abs(original-saved),
                               'fp32_vs_original': abs(fp32-original)})
            save_json(model_dir/'prediction_parity.json', parity)
            if max(p['original_vs_saved'] for p in parity) > .001:
                raise ValueError('Saved-prediction parity gate failed for ' + adapter.name)
            for case, parity_row in zip(cases, parity):
                case_start = time.perf_counter()
                if device == 'cuda': torch.cuda.reset_peak_memory_stats()
                index = case['validation_index']
                x = torch.from_numpy(np.load(paths[len(train)+index], allow_pickle=False).copy()).unsqueeze(0).to(device)
                case_dir = model_dir/case['clip_id']; case_dir.mkdir()
                maps = [gradcam(model, x, adapter.layer, sign) for sign in (1, -1)]
                for name, cam in zip(('rd', 'non_rd'), maps):
                    np.savez_compressed(case_dir/(name+'.npz'), raw=cam['raw'], normalized=cam['normalized'])
                probability = parity_row['fp32_probability']
                title = (f'{adapter.name} epoch {freeze["selected_epoch"]} | {case["clip_id"]} | '
                         f'label={case["label"]}, prediction={int(probability >= .5)}, p(RD)={probability:.4f}\n'
                         f'layer={adapter.layer}')
                render_case(case_dir, x, maps, title)
                occluded = occlusion(model, x, maps[0]['normalized'])
                save_json(case_dir/'occlusion.json', occluded)
                control_rows.append({'model': adapter.name, 'clip_id': case['clip_id'],
                    'constant_rd_map': maps[0]['constant_map'],
                    **{group+'_mean_logit_drop': values['mean_drop'] for group, values in occluded['groups'].items()}})
                randomization = []
                for result in randomized_maps(model, x, adapter, maps[0]['normalized']):
                    np.savez_compressed(case_dir/(result['stage']+'.npz'), raw=result['raw'], normalized=result['normalized'])
                    randomization.append({k: v for k, v in result.items() if k not in ('raw', 'normalized')})
                save_json(case_dir/'randomization.json', randomization)
                assert_state(model, before)
                with torch.no_grad():
                    after = float(model(x).sigmoid().item())
                if abs(after-probability) > 1e-6:
                    raise ValueError('Prediction changed after attribution')
                row = {**case, 'model': adapter.name, 'outcome': outcome(case['label'], probability),
                       'fp32_probability': probability, 'seconds': time.perf_counter()-case_start,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                       'maps': [{k: v for k, v in cam.items() if k not in ('raw', 'normalized')} for cam in maps]}
                save_json(case_dir/'result.json', row); rows.append(row)
                status['cases'].append({'model': adapter.name, 'clip_id': case['clip_id'], 'status': 'completed'})
                save_json(output/'status.json', status)
                print('Completed', adapter.name, case['clip_id'], flush=True)
            status['models'][adapter.name]['status'] = 'completed'
            del model, before
            if device == 'cuda': torch.cuda.empty_cache()
        save_json(output/'case_results.json', rows)
        pd.DataFrame([{k: v for k, v in row.items() if k != 'maps'} for row in rows]).to_csv(output/'comparison.csv', index=False)
        pd.DataFrame(control_rows).to_csv(output/'occlusion_comparison.csv', index=False)
        from PIL import Image
        matched = output/'matched_panels'; matched.mkdir()
        for case in cases:
            panels = [Image.open(output/a.name/case['clip_id']/'panel.png').convert('RGB') for a in adapters]
            canvas = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), 'white')
            offset = 0
            for panel in panels:
                canvas.paste(panel, (0, offset)); offset += panel.height; panel.close()
            canvas.save(matched/(case['clip_id']+'.png'))
        status.update(status='completed', seconds=time.perf_counter()-started)
    except Exception as error:
        status.update(status='failed', error=str(error), seconds=time.perf_counter()-started)
        raise
    finally:
        save_json(output/'status.json', status)
        hashes = {str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_sha256.json'}
        save_json(output/'artifact_sha256.json', hashes)
        with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file(): archive.write(path, str(path.relative_to(output.parent)))
    return status
