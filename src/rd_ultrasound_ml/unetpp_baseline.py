"""Independent implementation of the released ERDES UNet++ classifier protocol.

Architecture/protocol reference (read-only; no upstream source imported):
https://github.com/OSUPCVLab/ERDES/tree/4af67e0b9f4667edbae75a64a819884ee14d7a3b
The authors' encoder-only classifier uses MONAI TwoConv/Down blocks. The
equivalent operations below use PyTorch directly; this is not a full nested
segmentation UNet++. See docs/research/erdes-unetpp-protocol.md for
attribution, numerical parity checks, and paper/code discrepancies.
"""
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class UNetPPRDClassifier(nn.Module):
    """Five double-convolution blocks, four pools, global pool and 256→128→1."""

    def __init__(self):
        super().__init__()
        widths = (32, 32, 64, 128, 256)
        blocks = []
        incoming = 1
        for index, outgoing in enumerate(widths):
            layers = [] if index == 0 else [nn.MaxPool3d(2, stride=2)]
            for channels_in in (incoming, outgoing):
                layers.extend([
                    nn.Conv3d(channels_in, outgoing, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.InstanceNorm3d(outgoing, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ])
            blocks.append(nn.Sequential(*layers))
            incoming = outgoing
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, inputs):
        features = self.encoder(inputs)
        return self.head(self.pool(features).flatten(1)).squeeze(1)


def resize_grayscale_volume(gray_frames, target=(96, 128, 128)):
    """Full-time-axis trilinear interpolation, not uniform frame subsampling."""
    if not isinstance(gray_frames, np.ndarray) or gray_frames.ndim != 3:
        raise ValueError('Expected a D,H,W numpy array of decoded grayscale values')
    if min(gray_frames.shape) <= 0 or len(target) != 3 or min(target) <= 0:
        raise ValueError('Source and target dimensions must be positive')
    if not np.isfinite(gray_frames).all() or gray_frames.min() < 0 or gray_frames.max() > 255:
        raise ValueError('Decoded grayscale values must be finite and in [0,255]')
    tensor = torch.from_numpy(np.ascontiguousarray(gray_frames, dtype=np.float32))[None]
    height, width = tensor.shape[-2:]
    difference = abs(height - width)
    if difference:
        before, after = difference // 2, difference - difference // 2
        padding = (0, 0, before, after) if height < width else (before, after, 0, 0)
        tensor = F.pad(tensor, padding, mode='constant', value=0)
    tensor = F.interpolate(tensor[None], size=tuple(target), mode='trilinear', align_corners=False)[0]
    tensor.div_(255.)
    minimum, maximum = tensor.min().item(), tensor.max().item()
    if not torch.isfinite(tensor).all() or minimum < -1e-6 or maximum > 1 + 1e-6:
        raise ValueError(f'Unexpected preprocessed range: {minimum}, {maximum}')
    return tensor.clamp_(0., 1.)


def prepare_unetpp_clip(path, expected_frames, expected_height=None, expected_width=None, target=(96, 128, 128)):
    """Decode all frames into one preallocated grayscale buffer, with metadata checks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_frames < 1:
        raise ValueError('Expected frame count must be positive')
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    gray = None
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot decode {path.name}')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f'{path.name}: expected uint8 three-channel decoded frame')
            height, width = frame.shape[:2]
            if ((expected_height is not None and height != expected_height)
                    or (expected_width is not None and width != expected_width)):
                raise ValueError(f'{path.name}: decoded dimensions differ from metadata')
            if decoded >= expected_frames:
                raise ValueError(f'{path.name}: decoded more frames than metadata expects')
            if gray is None:
                gray = np.empty((expected_frames, height, width), dtype=np.float32)
            if frame.shape[:2] != gray.shape[1:]:
                raise ValueError(f'{path.name}: changing frame dimensions')
            # Arithmetic channel mean is invariant to OpenCV's BGR channel order.
            gray[decoded] = frame.mean(axis=2, dtype=np.float32)
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_frames:
        raise ValueError(f'{path.name}: decoded {decoded} frames; metadata expects {expected_frames}')
    return resize_grayscale_volume(gray, target)


def benchmark_cuda_batch(model_factory, inputs, labels, physical_batch, warmup=1, steps=3):
    """Disposable FP32 training/inference probe; never produces a research checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA runtime is required for the hardware probe')
    if physical_batch < 1 or steps < 1 or warmup < 1:
        raise ValueError('Positive batch size, warmup and measurement steps are required')
    device = torch.device('cuda')
    model = optimizer = x = y = logits = loss = None
    initial_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5, weight_decay=0.)
        indices = torch.arange(physical_batch) % len(inputs)
        x, y = inputs[indices].to(device), labels[indices].to(device)
        loss_function = nn.BCEWithLogitsLoss()
        train_seconds = []
        for iteration in range(warmup + steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(x)
            loss = loss_function(logits, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite hardware-probe loss')
            loss.backward()
            if iteration == 0:
                gradients = [parameter.grad for parameter in model.parameters()]
                if not all(g is not None and torch.isfinite(g).all() for g in gradients):
                    raise ValueError('Nonfinite or missing hardware-probe gradients')
                if not any(torch.count_nonzero(g).item() for g in gradients):
                    raise ValueError('All hardware-probe gradients are zero')
                del gradients
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= warmup:
                train_seconds.append(elapsed)
        model.eval()
        inference_seconds = []
        with torch.no_grad():
            for _ in range(steps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                inference_seconds.append(time.perf_counter() - start)
        return {'physical_batch': physical_batch, 'status': 'passed', 'precision': 'float32',
            'training_step_seconds_median': float(np.median(train_seconds)),
            'validation_step_seconds_median': float(np.median(inference_seconds)),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'initial_allocated_bytes': initial_allocated,
            'finite_nonzero_gradients': True, 'loss_last': loss.item(),
            'weights_discarded': True}
    except torch.cuda.OutOfMemoryError:
        return {'physical_batch': physical_batch, 'status': 'out_of_memory', 'precision': 'float32',
            'weights_discarded': True}
    finally:
        del model, optimizer, x, y, logits, loss
        torch.cuda.empty_cache()
