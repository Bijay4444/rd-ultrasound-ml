"""One-channel torchvision R(2+1)D-18 RD clip classifier (comparison model).

Architecture source: torchvision.models.video.r2plus1d_18, constructed with
weights=None and num_classes=1 (Tran et al., CVPR 2018, arXiv:1711.11248).
Only the first stem convolution changes, from three input channels to one, so
the model accepts the existing grayscale cache. The stem's later temporal
convolution, all BatchNorm/ReLU layers, factorized residual blocks, pooling and
Linear(512, 1) head are the builder's defaults. No pretrained weights are used.
"""
import torch
from torch import nn
import torchvision
from torchvision.models.video import r2plus1d_18

GRADCAM_LAYER = 'backbone.layer4'
STEM_REPLACEMENT = {'in_channels': 1, 'out_channels': 45, 'kernel_size': (1, 7, 7),
    'stride': (1, 2, 2), 'padding': (0, 3, 3), 'bias': False}
INITIALIZATION = ("builder defaults for all modules; replacement stem[0] uses "
    "kaiming_normal_(mode='fan_out', nonlinearity='relu'), the builder's Conv3d rule. "
    "fan_out (45*1*7*7) does not depend on input channels, so the replacement "
    "uses the same initialization distribution as the stock first convolution")


class R2Plus1D18RDClassifier(nn.Module):
    """Wraps torchvision VideoResNet as ``backbone``; forward returns raw logits of shape [B]."""

    def __init__(self):
        super().__init__()
        backbone = r2plus1d_18(weights=None, num_classes=1)
        original = backbone.stem[0]
        expected = {'in_channels': 3, 'out_channels': 45, 'kernel_size': (1, 7, 7),
            'stride': (1, 2, 2), 'padding': (0, 3, 3), 'bias': False}
        found = {'in_channels': original.in_channels, 'out_channels': original.out_channels,
            'kernel_size': original.kernel_size, 'stride': original.stride,
            'padding': original.padding, 'bias': original.bias is not None}
        if not isinstance(original, nn.Conv3d) or found != expected:
            raise RuntimeError(f'Installed torchvision {torchvision.__version__} stem differs: {found}')
        replacement = nn.Conv3d(1, original.out_channels, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding, dilation=original.dilation,
            groups=original.groups, bias=False, padding_mode=original.padding_mode)
        nn.init.kaiming_normal_(replacement.weight, mode='fan_out', nonlinearity='relu')
        backbone.stem[0] = replacement
        self.backbone = backbone

    def forward(self, inputs):
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError(f'Expected B,1,D,H,W grayscale clips; received {tuple(inputs.shape)}')
        return self.backbone(inputs).squeeze(1)


def gradcam_layer(model):
    """Return the final pre-pooling feature module (``backbone.layer4``)."""
    return model.get_submodule(GRADCAM_LAYER)


def describe_model(model=None):
    """Plain-data architecture record for configs/reports; builds a model when omitted."""
    model = R2Plus1D18RDClassifier() if model is None else model
    stem = model.backbone.stem
    return {'name': 'torchvision_r2plus1d_18_grayscale_binary',
        'builder': 'torchvision.models.video.r2plus1d_18(weights=None, num_classes=1)',
        'torchvision': torchvision.__version__, 'torch': str(torch.__version__),
        'pretrained': False, 'stem_replacement': {**STEM_REPLACEMENT,
            'kernel_size': list(STEM_REPLACEMENT['kernel_size']), 'stride': list(STEM_REPLACEMENT['stride']),
            'padding': list(STEM_REPLACEMENT['padding'])},
        'stem': [repr(module) for module in stem],
        'initialization': INITIALIZATION,
        'normalization': 'BatchNorm3d builder defaults (eps=1e-5, momentum=0.1, track_running_stats=True)',
        'head': 'AdaptiveAvgPool3d(1), flatten, Linear(512,1); squeeze to [B] raw logits',
        'gradcam_layer': GRADCAM_LAYER,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters())}
