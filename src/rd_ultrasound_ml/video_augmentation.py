"""Training-only spatial/intensity augmentation of grayscale video tensors.

Own implementation of the proposal in docs/AUGMENTATION_EXPLAINABILITY_PLAN.md.
Parameters are keyed by seed, epoch and clip ID using a local CPU RNG. Thus
preview/probe calls do not consume training RNG, and resumed epochs replay the
same views without changing the frozen training engine or cached preprocessing.
"""
from copy import deepcopy
import hashlib
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from rd_ultrasound_ml.unetpp_training import CheckpointStore, run_training


CANDIDATE_POLICY = {
    'version': 1,
    'affine_probability': 0.5,
    'rotation_degrees': 3.0,
    'translation_fraction': 0.01,
    'scale_range': [0.98, 1.02],
    'gain_probability': 0.5,
    'gain_range': [0.9, 1.1],
    'order': ['affine', 'gain'],
    'interpolation': 'bilinear',
    'padding_mode': 'zeros',
    'align_corners': False,
    'clamp_range': [0.0, 1.0],
    'temporal_policy': 'one parameter set for all temporal positions; preserve order',
    'rng': 'sha256(video_augmentation_v1:seed:epoch:clip_id), first 8 bytes, 63-bit CPU torch.Generator',
}


def validate_policy(policy):
    if set(policy) != set(CANDIDATE_POLICY):
        raise ValueError('Augmentation policy has missing or unexpected fields')
    for key in ('version', 'order', 'interpolation', 'padding_mode', 'align_corners',
                'clamp_range', 'temporal_policy', 'rng'):
        if policy[key] != CANDIDATE_POLICY[key]:
            raise ValueError(f'Unsupported augmentation setting: {key}')
    for key, upper in [('affine_probability', 1), ('gain_probability', 1),
                       ('rotation_degrees', 180), ('translation_fraction', 1)]:
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f'Invalid augmentation setting: {key}')
    for key in ('scale_range', 'gain_range'):
        values = policy[key]
        if (not isinstance(values, list) or len(values) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values) or
                not 0 < values[0] <= values[1]):
            raise ValueError(f'Invalid augmentation range: {key}')


def sample_parameters(policy, seed, epoch, clip_id):
    validate_policy(policy)
    if (type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 1 or
            not isinstance(clip_id, str) or not clip_id):
        raise ValueError('Expected nonnegative integer seed, positive epoch and nonempty clip ID')
    key = f'video_augmentation_v1:{seed}:{epoch}:{clip_id}'.encode()
    rng_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') & ((1 << 63) - 1)
    generator = torch.Generator(device='cpu').manual_seed(rng_seed)
    draws = torch.rand(7, generator=generator, dtype=torch.float64).tolist()
    affine, gain = draws[0] < policy['affine_probability'], draws[1] < policy['gain_probability']
    lo_scale, hi_scale = policy['scale_range']
    lo_gain, hi_gain = policy['gain_range']
    return {'rng_seed': rng_seed, 'affine_applied': affine, 'gain_applied': gain,
        'rotation_degrees': (2 * draws[2] - 1) * policy['rotation_degrees'] if affine else 0.0,
        'translation_x_fraction': (2 * draws[3] - 1) * policy['translation_fraction'] if affine else 0.0,
        'translation_y_fraction': (2 * draws[4] - 1) * policy['translation_fraction'] if affine else 0.0,
        'scale': lo_scale + draws[5] * (hi_scale - lo_scale) if affine else 1.0,
        'gain': lo_gain + draws[6] * (hi_gain - lo_gain) if gain else 1.0}


def apply_parameters(inputs, parameters):
    """Apply a forward pixel-space affine about the image center, then gain.

    Input/output: CPU float32 [1,D,H,W]. Treat D as channels for this *2D*
    resampling operation: one grid transforms every temporal position identically,
    with no interpolation or rearrangement along time. The inverse grid accounts
    for H/W, so rectangular inputs also rotate in pixel rather than normalized space.
    Never modifies the input tensor or its underlying array.
    """
    if (not isinstance(inputs, torch.Tensor) or inputs.device.type != 'cpu' or
            inputs.dtype != torch.float32 or inputs.ndim != 4 or inputs.shape[0] != 1 or
            min(inputs.shape) < 1):
        raise ValueError('Expected a CPU float32 [1,D,H,W] grayscale clip')
    if not torch.isfinite(inputs).all() or inputs.min() < 0 or inputs.max() > 1:
        raise ValueError('Augmentation input must be finite in [0,1]')
    angle, tx, ty, scale, gain = [parameters[key] for key in
        ('rotation_degrees', 'translation_x_fraction', 'translation_y_fraction', 'scale', 'gain')]
    if not all(math.isfinite(v) for v in (angle, tx, ty, scale, gain)) or scale <= 0 or gain <= 0:
        raise ValueError('Invalid affine/gain parameters')
    output = inputs.clone()
    if parameters['affine_applied']:
        height, width = inputs.shape[-2:]
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        theta = inputs.new_tensor([[
            [cosine / scale, sine * height / width / scale,
             -2 * (cosine * tx + sine * ty * height / width) / scale],
            [-sine * width / height / scale, cosine / scale,
             -2 * (-sine * tx * width / height + cosine * ty) / scale]]])
        grid = F.affine_grid(theta, output.shape, align_corners=False)
        output = F.grid_sample(output, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    if parameters['gain_applied']:
        output = output * gain
    return output.clamp(0, 1)


def augment_clip(inputs, policy, seed, epoch, clip_id):
    parameters = sample_parameters(policy, seed, epoch, clip_id)
    return apply_parameters(inputs, parameters), parameters


class TrainingAugmentedClips(Dataset):
    """Wrap training data only; epoch is restored from the authoritative checkpoint."""

    def __init__(self, base, clip_ids, policy, seed, partition='train'):
        validate_policy(policy)
        self.clip_ids = list(clip_ids)
        if partition != 'train':
            raise ValueError('Augmentation wrapper accepts only the train partition')
        if (len(base) != len(self.clip_ids) or not self.clip_ids or
                len(set(self.clip_ids)) != len(self.clip_ids) or
                any(not isinstance(c, str) or not c for c in self.clip_ids)):
            raise ValueError('Expected unique clip IDs matching the training dataset')
        sample_parameters(policy, seed, 1, self.clip_ids[0])
        self.base, self.policy, self.seed = base, deepcopy(policy), seed
        self.epoch = 1

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Epoch must be a positive integer')
        self.epoch = epoch

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        inputs, label = self.base[index]
        output, _ = augment_clip(inputs, self.policy, self.seed, self.epoch, self.clip_ids[index])
        return output, label


def run_augmented_training(model_factory, train_dataset, validation_dataset, config,
                           directory, data_fingerprint, runtime, device='cuda', report=print):
    """Run/resume one epoch with unchanged run_training and its complete state checks."""
    if not isinstance(train_dataset, TrainingAugmentedClips) or isinstance(validation_dataset, TrainingAugmentedClips):
        raise ValueError('Expected augmented training and clean validation datasets')
    if train_dataset.policy != config.get('training_augmentation') or train_dataset.seed != config['seed']:
        raise ValueError('Training augmentation differs from the recorded configuration')
    checkpoint = CheckpointStore(directory).load()
    epoch = 1 if checkpoint is None else checkpoint['epoch'] + int(checkpoint['phase'] == 'epoch_complete')
    train_dataset.set_epoch(epoch)
    return run_training(model_factory, train_dataset, validation_dataset, config, directory,
        data_fingerprint, runtime, device=device, max_epochs_this_call=1, report=report)
