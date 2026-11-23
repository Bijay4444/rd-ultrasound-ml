"""Volumetric Grad-CAM and sensitivity controls for frozen binary clip models.

Grad-CAM adaptation: Selvaraju et al., https://arxiv.org/abs/1610.02391.
Randomization motivation: Adebayo et al., https://arxiv.org/abs/1810.03292.
Maps attribute a score; they are not segmentation or clinical validation.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    factory: object
    layer: str
    head: str
    freeze: dict


def outcome(label, probability):
    return ('TP' if probability >= .5 else 'FN') if label else ('FP' if probability >= .5 else 'TN')


def select_cases(frame, probabilities):
    """Two per baseline outcome, sorted by string clip ID, before inspecting maps."""
    probabilities = np.asarray(probabilities)
    if (len(frame) != len(probabilities) or not frame.clip_id.is_unique
            or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Invalid validation identities or probabilities')
    selected = []
    for category in ('TP', 'TN', 'FP', 'FN'):
        candidates = [i for i, (_, row) in enumerate(frame.iterrows())
                      if outcome(int(row.label), probabilities[i]) == category]
        if len(candidates) < 2:
            raise ValueError('Need two cases in every baseline outcome category')
        for i in sorted(candidates, key=lambda j: str(frame.iloc[j].clip_id))[:2]:
            selected.append({'validation_index': i, 'clip_id': str(frame.iloc[i].clip_id),
                             'label': int(frame.iloc[i].label), 'baseline_outcome': category})
    return selected


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def assert_state(model, before):
    if any(not torch.equal(v.detach().cpu(), before[k]) for k, v in model.state_dict().items()):
        raise ValueError('Explanation changed model state')


def gradcam(model, inputs, layer, sign=1):
    """Batch-one FP32 CAM; no parameter gradients or optimizer updates are used."""
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('All modules must be in evaluation mode')
    if inputs.ndim != 5 or inputs.shape[0] != 1 or inputs.dtype != torch.float32 or sign not in (-1, 1):
        raise ValueError('Require one FP32 B,C,D,H,W clip and sign +/-1')
    if not torch.isfinite(inputs).all():
        raise ValueError('Nonfinite input')
    captured = []
    hook = model.get_submodule(layer).register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        with torch.enable_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
            x = inputs.detach().clone().requires_grad_(True)
            logits = model(x).reshape(-1)
            if logits.numel() != 1 or len(captured) != 1 or captured[0].ndim != 5:
                raise ValueError('Require one logit and one volumetric activation')
            activation = captured[0]
            gradient, = torch.autograd.grad(sign * logits[0], activation)
            if not torch.isfinite(logits).all() or not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite logit or gradient')
            coarse = (gradient.mean((2, 3, 4), keepdim=True) * activation).sum(1, keepdim=True).relu()
            raw = F.interpolate(coarse, size=inputs.shape[2:], mode='trilinear', align_corners=False)[0, 0]
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite map')
            span = raw.max() - raw.min()
            constant = bool(span <= 1e-12)
            normalized = torch.zeros_like(raw) if constant else (raw - raw.min()) / span
            return {'logit': float(logits[0].detach()), 'activation_shape': list(activation.shape),
                    'target_sign': sign, 'constant_map': constant,
                    'zero_map': bool(raw.max() <= 1e-12),
                    'raw': raw.detach().cpu().numpy(), 'normalized': normalized.detach().cpu().numpy()}
    finally:
        hook.remove()
        captured.clear()


def similarity(first, second):
    a, b = np.asarray(first).ravel(), np.asarray(second).ravel()
    correlation = None if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(np.corrcoef(a, b)[0, 1])
    return {'pearson': correlation, 'mean_absolute_difference': float(np.abs(a-b).mean())}


def occlusion(model, inputs, attribution, cuboid=(12, 16, 16), count=8, seed=42):
    """One zero-filled cuboid per intervention; higher drop means stronger sensitivity."""
    shape = tuple(inputs.shape[2:])
    if model.training or tuple(attribution.shape) != shape or any(n % c for n, c in zip(shape, cuboid)):
        raise ValueError('Require eval model, matching map, and evenly tiled volume')
    boxes = [tuple(slice(i, i+c) for i, c in zip((d, h, w), cuboid))
             for d in range(0, shape[0], cuboid[0]) for h in range(0, shape[1], cuboid[1])
             for w in range(0, shape[2], cuboid[2])]
    if not 0 < count <= len(boxes):
        raise ValueError('Invalid intervention count')
    ranking = sorted(range(len(boxes)), key=lambda i: (float(attribution[boxes[i]].mean()), i))
    groups = {'high': ranking[-count:], 'low': ranking[:count],
              'random': np.random.default_rng(seed).choice(len(boxes), count, replace=False).tolist()}
    result = {'cuboid': list(cuboid), 'count': count, 'seed': seed, 'fill': 0,
              'constant_map': bool(np.ptp(attribution) <= 1e-12), 'groups': {}}
    with torch.no_grad():
        baseline = float(model(inputs).item())
        result['baseline_logit'] = baseline
        for name, indices in groups.items():
            rows = []
            for index in indices:
                masked = inputs.clone()
                masked[(slice(None), slice(None), *boxes[index])] = 0
                logit = float(model(masked).item())
                if not np.isfinite(logit):
                    raise ValueError('Nonfinite occlusion output')
                rows.append({'index': int(index), 'start': [s.start for s in boxes[index]],
                             'logit': logit, 'drop': baseline-logit})
            result['groups'][name] = {'interventions': rows, 'mean_drop': float(np.mean([r['drop'] for r in rows]))}
    return result


def randomized_maps(model, inputs, adapter, reference, seed=42):
    """Disposable cumulative randomization, leaving original state and RNG untouched."""
    device = inputs.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    for stage in ('head', 'head_and_block', 'all'):
        copy = deepcopy(model)
        roots = ([copy.get_submodule(adapter.head)] if stage == 'head' else
                 [copy.get_submodule(adapter.head), copy.get_submodule(adapter.layer)] if stage == 'head_and_block'
                 else [copy])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            seen = set()
            for root in roots:
                for module in root.modules():
                    if id(module) not in seen and hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
                        seen.add(id(module))
        copy.eval()
        result = gradcam(copy, inputs, adapter.layer)
        result['similarity'] = similarity(reference, result['normalized'])
        result['stage'] = stage
        del copy
        yield result
