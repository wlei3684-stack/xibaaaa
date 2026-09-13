"""VMamba-T s1l8 tracking backbone, fusion and optional OSTrack center head."""
import torch
from .vmamba import VSSM


class VMambaTrack(VSSM):
    def __init__(self, pretrained=None, **kwargs):
        config = dict(
            depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
            patch_size=4, in_chans=3, ssm_d_state=1, ssm_ratio=1.0,
            ssm_dt_rank='auto', ssm_act_layer='silu', ssm_conv=3,
            ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
            forward_type='v05_noz', mlp_ratio=4.0, mlp_act_layer='gelu',
            mlp_drop_rate=0.0, patch_norm=True, norm_layer='ln2d',
            downsample_version='v3', patchembed_version='v2', posembed=False,
        )
        config.update(kwargs)
        if config['norm_layer'] != 'ln2d' or config['forward_type'] != 'v05_noz' or config['posembed']:
            raise ValueError('Paired backbone supports the official s1l8 ln2d/v05_noz configuration')
        super().__init__(**config)
        del self.classifier
        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state = {k.removeprefix('module.'): value for k, value in state.items()}
        state = {k: value for k, value in state.items() if not k.startswith('classifier.')}
        # Fail loudly on any missing/mismatched backbone weights.
        return self.load_state_dict(state, strict=True)

    def forward(self, template, search=None):
        """Single image -> four maps; pair -> (four template maps, four search maps).

        Maps are raw stage outputs before downsampling, in B,C,H,W format.
        Typical image inputs: template B,3,128,128; search B,3,256,256.
        """
        if search is None:
            x = self.patch_embed(template)
            outputs = []
            for layer in self.layers:
                x = layer.blocks(x)
                outputs.append(x)
                x = layer.downsample(x)
            return tuple(outputs)
        if template.ndim != 4 or search.ndim != 4:
            raise ValueError('Both inputs must be B,3,H,W images')
        if template.shape[:2] != search.shape[:2] or template.shape[1] != 3:
            raise ValueError('Inputs must have matching batch size and three channels')
        t, s = self.patch_embed(template), self.patch_embed(search)
        template_outputs, search_outputs = [], []
        for layer in self.layers:
            t, s = layer.blocks((t, s))
            template_outputs.append(t)
            search_outputs.append(s)
            t, s = layer.downsample(t), layer.downsample(s)
        return tuple(template_outputs), tuple(search_outputs)


def vmamba_tiny_track(pretrained=None, use_checkpoint=False):
    return VMambaTrack(pretrained=pretrained, use_checkpoint=use_checkpoint)


class VMambaTrackWithFusion(torch.nn.Module):
    """Paired VMamba backbone followed by a two-stage search-only fusion neck."""
    def __init__(self, backbone, channels=384):
        super().__init__()
        from .tracking_fusion import TwoStageUPerFusion
        self.backbone = backbone
        self.fusion = TwoStageUPerFusion(
            in_channels=tuple(backbone.dims[-2:]), channels=channels,
        )

    def forward(self, template, search, return_features=False):
        template_features, search_features = self.backbone(template, search)
        fused_search = self.fusion(search_features[-2], search_features[-1])
        if return_features:
            return dict(template_features=template_features,
                        search_features=search_features, fused_search=fused_search)
        return fused_search


def vmamba_tiny_track_with_fusion(pretrained=None, use_checkpoint=False):
    """Load official backbone weights; initialize the 384-channel fusion neck anew."""
    return VMambaTrackWithFusion(vmamba_tiny_track(pretrained, use_checkpoint))


class VMambaTracker(VMambaTrackWithFusion):
    """Two-stage fusion -> learned 1x1 projection -> official OSTrack center head."""
    def __init__(self, backbone, head_pretrained=None):
        super().__init__(backbone, channels=384)
        from .ostrack_head import CenterPredictor
        self.projection = torch.nn.Conv2d(384, 768, kernel_size=1, bias=True)
        self.box_head = CenterPredictor(inplanes=768, channel=256, feat_sz=16, stride=16)
        if head_pretrained is not None:
            self.load_ostrack_head(head_pretrained)

    def load_ostrack_head(self, path):
        """Load all head parameters/buffers from an OSTrack or head-only state dict.

        Use the OSTrack-256 ViT-B CENTER checkpoint. Backbone weights are ignored.
        Fusion and projection are new and must be trained.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('net', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        if any(k.startswith('box_head.') for k in state):
            state = {k.removeprefix('box_head.'): v for k, v in state.items()
                     if k.startswith('box_head.')}
        expected = self.box_head.state_dict()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError('Checkpoint does not contain a complete compatible OSTrack CENTER head')
        return self.box_head.load_state_dict(state, strict=True)

    def forward_head(self, fused_search, gt_score_map=None):
        if fused_search.ndim != 4 or tuple(fused_search.shape[1:]) != (384, 16, 16):
            raise ValueError('Expected fused search features B,384,16,16 (search crop 256x256)')
        projected = self.projection(fused_search)
        score, boxes, size, offset = self.box_head(projected, gt_score_map)
        return dict(pred_boxes=boxes.unsqueeze(1), score_map=score,
                    size_map=size, offset_map=offset)

    def forward(self, template, search, gt_score_map=None, return_features=False):
        features = super().forward(template, search, return_features=return_features)
        fused = features['fused_search'] if return_features else features
        result = self.forward_head(fused, gt_score_map)
        if return_features:
            result.update(features)
        return result


def vmamba_tiny_tracker(pretrained=None, head_pretrained=None, use_checkpoint=False):
    return VMambaTracker(vmamba_tiny_track(pretrained, use_checkpoint), head_pretrained)
