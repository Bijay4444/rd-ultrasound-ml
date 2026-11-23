"""Auditable, training-only previews; no model inference or parameter selection."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from rd_ultrasound_ml.unetpp_training import json_digest, write_json_atomic
from rd_ultrasound_ml.video_augmentation import apply_parameters, augment_clip


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_preview(dataset, manifest, clip_ids, policy, seed, output, source_records=None):
    """Render a fixed manifest of training cases with the exact online transform.

    The caller must supply the verified official training manifest, never a
    combined train/validation frame. The exported IDs/labels allow that claim to
    be independently checked. Boundary examples are explicitly separate from the
    probabilistically sampled epoch-1 views.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ids = manifest['clip_id'].tolist()
    if len(dataset) != len(manifest) or len(set(ids)) != len(ids) or not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError('Preview requires unique IDs and matching training dataset/manifest')
    if not set(clip_ids) <= set(ids):
        raise ValueError('Preview clip is outside the supplied training partition')
    records = []
    positions = [0, 47, 95]
    figure, axes = plt.subplots(len(clip_ids), 6, figsize=(15, 2.8 * len(clip_ids)), squeeze=False)
    boundary, boundary_axes = plt.subplots(len(clip_ids), 3, figsize=(8, 2.3 * len(clip_ids)), squeeze=False)
    for row_number, clip_id in enumerate(clip_ids):
        index = ids.index(clip_id)
        inputs, label = dataset[index]
        if inputs.shape != (1, 96, 128, 128) or int(label.item()) != int(manifest.iloc[index]['label']):
            raise ValueError('Preview input shape/label differs from the training manifest')
        original_hash = tensor_digest(inputs)
        augmented, parameters = augment_clip(inputs, policy, seed, 1, clip_id)
        replay, repeated_parameters = augment_clip(inputs, policy, seed, 1, clip_id)
        unchanged = tensor_digest(inputs) == original_hash
        reproducible = parameters == repeated_parameters and torch.equal(augmented, replay)
        if not unchanged or not reproducible:
            raise ValueError('Preview mutated its input or could not reproduce its transform')
        target = 'RD' if int(label.item()) else 'non-RD'
        description = (f"{clip_id} | {target} | angle {parameters['rotation_degrees']:+.1f}°, "
            f"shift ({parameters['translation_x_fraction']:+.3f}, {parameters['translation_y_fraction']:+.3f}), "
            f"scale {parameters['scale']:.3f}, gain {parameters['gain']:.3f}")
        for column in range(6):
            value, position = (inputs if column < 3 else augmented), positions[column % 3]
            axis = axes[row_number, column]
            axis.imshow(value[0, position], cmap='gray', vmin=0, vmax=1)
            axis.set_title(('Original' if column < 3 else 'Augmented') + f' · {position}', fontsize=8)
            axis.axis('off')
        axes[row_number, 0].text(0, 1.28, description, transform=axes[row_number, 0].transAxes, fontsize=8, va='bottom')
        # Paired animation: both halves use the identical resampled temporal index.
        frames = []
        for position in np.linspace(0, 95, 12, dtype=int):
            frame = Image.new('RGB', (512, 300), 'white')
            draw = ImageDraw.Draw(frame)
            draw.text((8, 5), f'{clip_id} | label {target} | resampled index {position}', fill='black')
            draw.text((8, 23), 'Original', fill='black')
            draw.text((264, 23), 'Augmented (epoch 1)', fill='black')
            for left, value in [(0, inputs), (256, augmented)]:
                image = Image.fromarray(np.rint(value[0, position].numpy() * 255).astype(np.uint8)).resize((256, 256))
                frame.paste(image, (left, 44))
            frames.append(frame)
        frames[0].save(output / f'{clip_id}.gif', save_all=True, append_images=frames[1:], duration=180, loop=0)
        bounds = []
        for column, sign in enumerate([-1, 1], 1):
            extreme = {'affine_applied': True, 'gain_applied': True,
                'rotation_degrees': sign * policy['rotation_degrees'],
                'translation_x_fraction': sign * policy['translation_fraction'],
                'translation_y_fraction': -sign * policy['translation_fraction'],
                'scale': policy['scale_range'][0 if sign < 0 else 1],
                'gain': policy['gain_range'][0 if sign < 0 else 1]}
            transformed = apply_parameters(inputs, extreme)
            bounds.append(extreme)
            boundary_axes[row_number, column].imshow(transformed[0, 47], cmap='gray', vmin=0, vmax=1)
            boundary_axes[row_number, column].set_title(f'Boundary {sign:+d}', fontsize=8)
        boundary_axes[row_number, 0].imshow(inputs[0, 47], cmap='gray', vmin=0, vmax=1)
        boundary_axes[row_number, 0].set_title(f'{clip_id} · {target}\nOriginal · index 47', fontsize=8)
        for axis in boundary_axes[row_number]:
            axis.axis('off')
        records.append({'clip_id': clip_id, 'label': int(label.item()), 'partition': 'train',
            'epoch': 1, 'parameters': parameters, 'boundary_parameters': bounds,
            'shape': list(augmented.shape), 'dtype': str(augmented.dtype),
            'min': float(augmented.min()), 'max': float(augmented.max()),
            'original_tensor_sha256': original_hash, 'augmented_tensor_sha256': tensor_digest(augmented),
            'input_unchanged': unchanged, 'deterministic_replay': reproducible,
            'source': (source_records or {}).get(clip_id)})
    figure.suptitle('Training-only augmentation · fixed cases · epoch 1\nSame spatial/intensity parameters across all 96 positions; dataset labels retained', fontsize=12)
    figure.subplots_adjust(left=.03, right=.995, top=.90, bottom=.02, wspace=.08, hspace=.7)
    figure.savefig(output / 'original_vs_augmented.png', dpi=150)
    plt.close(figure)
    boundary.suptitle('Training-only boundary stress preview\nCombined range endpoints; these are not sampled epoch-1 views', fontsize=11)
    boundary.tight_layout(rect=(0, 0, 1, .96), h_pad=1)
    boundary.savefig(output / 'boundary_stress.png', dpi=150)
    plt.close(boundary)
    result = {'schema_version': 1, 'status': 'preview_generated_not_clinically_validated',
        'policy': policy, 'policy_digest': json_digest(policy), 'seed': seed,
        'selection_rule': 'reuse the eight fixed training-inspection IDs selected before augmentation work',
        'positions': positions, 'cases': records, 'test_used': False,
        'clinical_label_preservation': 'not established by automated checks or visual plausibility alone',
        'animation_timing': '12 resampled positions at 180 ms; not acquisition timing'}
    write_json_atomic(output / 'preview_manifest.json', result)
    return result
