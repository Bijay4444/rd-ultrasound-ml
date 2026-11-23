"""Freeze a separate matched training config after inspecting training previews.

Does not train or submit anything. The --preview manifest must be the visually
reviewed export; retain that export and document observations in the results note.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def freeze(preview_path, runtime_path, destination):
    preview_path, runtime_path, destination = map(Path, (preview_path, runtime_path, destination))
    preview = json.loads(preview_path.read_text())
    cases = preview['cases']
    if (preview.get('test_used') is not False or len(cases) != 8 or
            len({c['clip_id'] for c in cases}) != 8 or {c['label'] for c in cases} != {0, 1} or
            any(c['partition'] != 'train' or not c['input_unchanged'] or not c['deterministic_replay'] for c in cases)):
        raise ValueError('Require checked, training-only eight-case previews covering both labels')
    canonical = json.dumps(preview['policy'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != preview['policy_digest']:
        raise ValueError('Preview policy digest differs')
    config = deepcopy(json.loads((ROOT / 'configs/unetpp_rd_002_training.json').read_text()))
    reference = json.loads((ROOT / 'configs/unetpp_rd_002_evaluation.json').read_text())
    for name, digest in reference['training_implementation_sha256'].items():
        if hashlib.sha256((ROOT / f'src/rd_ultrasound_ml/{name}.py').read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen baseline implementation differs')
    if preview['seed'] != config['seed']:
        raise ValueError('Preview seed differs from the baseline')
    config.update(experiment_id='unetpp_rd_aug_001_amp', status='frozen_training_protocol_not_a_result',
        scope='one matched U-Net++ training-augmentation ablation; not an architecture comparison',
        comparison_reference='unetpp_rd_002_amp', protocol='docs/AUGMENTATION_EXPLAINABILITY_PLAN.md',
        training_augmentation=preview['policy'], reference_runtime=json.loads(runtime_path.read_text()),
        reference_implementation_sha256=reference['training_implementation_sha256'],
        cache={'preparation_experiment_id': 'unetpp_rd_002_amp',
            'preparation_config_digest': reference['training_config_digest'],
            'expected_cache_fingerprint': reference['training_data_fingerprint']},
        preview={'clip_ids': [c['clip_id'] for c in cases], 'policy_digest': preview['policy_digest'],
            'manifest_sha256': hashlib.sha256(preview_path.read_bytes()).hexdigest(),
            'review': 'sampled and combined-boundary training panels inspected; tighter bounds reduce border displacement; not clinical validation'})
    config['evaluation']['test_policy'] = ('no test access in training; choices frozen using validation; '
        'any later official-test analysis must disclose the previously exposed benchmark')
    config['disclosures'] += [
        'Training starts from scratch with seed 42; no fine-tuning of the selected baseline checkpoint.',
        'Cached preprocessing remains augmentation=none. Online training augmentation is recorded separately.',
        'Initial proposal +/-7 degrees, +/-3% translation, scale 0.95-1.05 was reduced after training previews to +/-3 degrees, +/-1%, scale 0.98-1.02 to reduce border displacement.',
        'Same spatial and gain parameters across all 96 positions; no temporal transforms; identical policy for both labels.',
        'Views use a local CPU RNG keyed by seed/epoch/clip ID. Epoch comes from the training checkpoint; source arrays remain unchanged.',
        'Mild transforms are engineering hypotheses, not established ultrasound-specific invariances; residual interpolation and border effects remain.',
        'Single-seed comparison cannot establish a reliable augmentation benefit. No test-driven tuning.']
    if destination.exists() and json.loads(destination.read_text()) != config:
        raise ValueError('Existing frozen config differs; preserve it and choose a new experiment')
    destination.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--reference-runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/unetpp_rd_aug_001_training.json')
    args = parser.parse_args()
    result = freeze(args.preview, args.reference_runtime, args.output)
    print(args.output, result['preview']['policy_digest'])
