"""Two-level UPerHead-style fusion of search Stage 3 and Stage 4 features.

Preserves PPM, top-down addition, final concatenation and the official SyncBN.
Single-GPU training requires batch >= 2 because PPM includes a 1x1 pooled map.
There is no semantic classifier or tracking prediction head.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TwoStageUPerFusion(nn.Module):
    def __init__(self, in_channels=(384, 768), channels=384,
                 pool_scales=(1, 2, 3, 6)):
        super().__init__()
        if len(in_channels) != 2 or any(c <= 0 for c in in_channels):
            raise ValueError('in_channels must contain Stage 3 and Stage 4 widths')
        if channels <= 0:
            raise ValueError('channels must be positive')
        if not pool_scales or any(not isinstance(s, int) or s <= 0 for s in pool_scales):
            raise ValueError('pool_scales must contain positive integer output sizes')
        self.in_channels = tuple(in_channels)
        self.channels = channels
        self.pool_scales = tuple(pool_scales)

        def conv_norm_act(cin, cout, kernel):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                nn.SyncBatchNorm(cout),
                nn.ReLU(inplace=True),
            )

        c3, c4 = self.in_channels
        self.psp_modules = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(scale), conv_norm_act(c4, channels, 1))
            for scale in self.pool_scales
        ])
        self.psp_bottleneck = conv_norm_act(c4 + len(self.pool_scales) * channels, channels, 3)
        self.lateral_conv = conv_norm_act(c3, channels, 1)
        self.fpn_conv = conv_norm_act(channels, channels, 3)
        self.fpn_bottleneck = conv_norm_act(2 * channels, channels, 3)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, stage3, stage4):
        if stage3.ndim != 4 or stage4.ndim != 4:
            raise ValueError('Expected two B,C,H,W feature tensors')
        if (stage3.shape[1], stage4.shape[1]) != self.in_channels:
            raise ValueError('Feature channels do not match configured Stage 3/4 widths')
        if stage3.shape[0] != stage4.shape[0]:
            raise ValueError('The two stages must have the same batch size')
        context = [stage4]
        context.extend(self._resize(pool(stage4), stage4.shape[-2:]) for pool in self.psp_modules)
        p4 = self.psp_bottleneck(torch.cat(context, dim=1))
        p4_up = self._resize(p4, stage3.shape[-2:])
        # Reuse the same upsampled tensor in both fusion paths.
        p3 = self.fpn_conv(self.lateral_conv(stage3) + p4_up)
        return self.fpn_bottleneck(torch.cat((p3, p4_up), dim=1))
