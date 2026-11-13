"""Model-agnostic probe/export helpers for the final-milestone comparison runs.

Validation-only: nothing here opens test data. Reuses the U-Net++ training
engine's metric, validation and precision-probe functions unchanged.
"""
import json
from pathlib import Path

import numpy as np
import torch

from rd_ultrasound_ml.unetpp_training import (binary_metrics, file_digest, probe_training_precision,
    validate_model)


def probe_physical_batches(model_factory, inputs, labels, batches=(4, 2, 1), precision='16-mixed'):
    """Disposable probes in order; stops after the first batch that fits. Weights are discarded."""
    results = []
    for batch in batches:
        try:
            result = probe_training_precision(model_factory, inputs, labels, batch, precision)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = {'status': 'out_of_memory', 'precision': precision, 'physical_batch': batch,
                'weights_discarded': True, 'test_used': False}
        results.append(result)
        if result['status'] == 'passed':
            break
    return results


def prediction_rows(manifest, labels, probabilities, threshold, partition):
    """One recomputable row per clip, in manifest order; rejects any label/order disagreement."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if len(manifest) != len(labels) or len(labels) != len(probabilities):
        raise ValueError('Prediction count differs from the manifest')
    if not np.array_equal(manifest['label'].to_numpy().astype(int), labels.astype(int)):
        raise ValueError('Prediction labels/order differ from the manifest')
    return [{'clip_id': clip_id, 'dataset_label': int(label), 'rd_probability': float(probability),
             'predicted_class': int(probability >= threshold), 'threshold': threshold, 'partition': partition}
            for clip_id, label, probability in zip(manifest['clip_id'], labels, probabilities)]


def verify_best_checkpoint(model_factory, directory, validation_dataset, device, physical_batch,
                           precision, threshold=.5, tolerance=1e-4):
    """Reload best.pt into a fresh model and reproduce the saved selected-epoch validation probabilities."""
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    saved = json.loads((directory / 'best_validation_predictions.json').read_text())
    payload = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    if payload['epoch'] != summary['best_epoch']:
        raise ValueError('best.pt epoch differs from the summary-selected epoch')
    model = model_factory()
    model.load_state_dict(payload['model_state'])
    model.to(device)
    _, labels, probabilities = validate_model(model, validation_dataset, device, physical_batch, precision)
    if labels.tolist() != saved['dataset_labels']:
        raise ValueError('Reloaded validation labels/order differ from saved predictions')
    difference = float(np.max(np.abs(probabilities - np.asarray(saved['rd_probabilities']))))
    saved_metrics = binary_metrics(saved['dataset_labels'], saved['rd_probabilities'], threshold)
    reloaded_metrics = binary_metrics(labels, probabilities, threshold)
    same_counts = all(saved_metrics[key] == reloaded_metrics[key] for key in ['TP', 'TN', 'FP', 'FN'])
    result = {'checkpoint': 'best.pt', 'checkpoint_sha256': file_digest(directory / 'best.pt'),
        'selected_epoch': payload['epoch'], 'max_abs_probability_difference': difference,
        'tolerance': tolerance, 'confusion_counts_reproduced': same_counts,
        'passed': difference <= tolerance and same_counts, 'precision': precision,
        'physical_batch': physical_batch, 'partition': 'validation', 'test_used': False}
    if not result['passed']:
        raise ValueError(f'Selected checkpoint reload did not reproduce validation predictions: {result}')
    return result
