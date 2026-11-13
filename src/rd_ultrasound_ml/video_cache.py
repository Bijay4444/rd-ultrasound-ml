"""Reuse the verified U-Net++ preparation cache under a new experiment identity.

``load_prepared_shards`` compares each preparation manifest's ``config_digest``
with the digest of the config it receives. That digest covers the original
preparation config (including its model/code identity), so passing a new
model's training config would reject a valid cache. This adapter does not
rewrite manifests or weaken those checks. It reads the preparation config that
each CPU job saved beside its manifest, verifies it against that manifest and
against the digest pinned in the new training config, checks that the two
share preprocessing/dataset identity, then calls the unchanged
``load_prepared_shards`` with the original preparation config. All tensor/source
hashes, label/partition/order and shape checks still run. The returned run
fingerprint binds the preparation identity, the cache fingerprint and the new
training config digest, so checkpoints cannot resume across any of them.
"""
import json
from pathlib import Path

from rd_ultrasound_ml.unetpp_kaggle import load_prepared_shards
from rd_ultrasound_ml.unetpp_training import json_digest

SHARED_FIELDS = ('dataset_revision', 'task', 'label_mapping', 'preprocessing')


def read_preparation_config(input_root):
    """Return the one preparation config shared by both attached shards and its digest."""
    manifests = sorted(Path(input_root).rglob('prepared_manifest.json'))
    if len(manifests) != 2:
        raise ValueError(f'Attach exactly two completed CPU preparation outputs; found {len(manifests)} manifests')
    configs = []
    for path in manifests:
        config_path = path.parent / 'config.json'
        if not config_path.is_file():
            raise ValueError(f'Preparation output lacks its saved config.json: {path.parent}')
        config = json.loads(config_path.read_text())
        if json_digest(config) != json.loads(path.read_text()).get('config_digest'):
            raise ValueError('Saved preparation config does not match its manifest digest')
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError('Attached preparation shards were produced by different configs')
    return configs[0], json_digest(configs[0])


def load_cache_for_run(input_root, training_config, report=print):
    """Validate the attached cache for a new training identity; never modifies inputs."""
    cache = training_config.get('cache')
    if not isinstance(cache, dict) or len(str(cache.get('preparation_config_digest', ''))) != 64:
        raise ValueError('Training config must pin cache.preparation_config_digest')
    preparation, preparation_digest = read_preparation_config(input_root)
    if preparation_digest != cache['preparation_config_digest']:
        raise ValueError('Attached preparation config differs from the pinned preparation identity')
    if preparation.get('experiment_id') != cache.get('preparation_experiment_id'):
        raise ValueError('Attached preparation experiment differs from the pinned one')
    for field in SHARED_FIELDS:
        if preparation.get(field) != training_config.get(field):
            raise ValueError(f'Training config {field} differs from the cached preparation')
    train, validation, paths, cache_fingerprint, identity = load_prepared_shards(
        input_root, preparation, report=report)
    expected = cache.get('expected_cache_fingerprint')
    if expected is not None and cache_fingerprint != expected:
        raise ValueError('Verified cache fingerprint differs from the pinned reference fingerprint')
    record = {'preparation_experiment_id': preparation['experiment_id'],
        'preparation_config_digest': preparation_digest, 'cache_fingerprint': cache_fingerprint,
        'preprocessing_identity': identity, 'training_config_digest': json_digest(training_config)}
    run_fingerprint = json_digest(record)
    return train, validation, paths, run_fingerprint, {**record, 'run_fingerprint': run_fingerprint}
