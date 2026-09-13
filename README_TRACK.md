# VMamba-T tracking backbone

Based on official VMamba commit `2ed52ead062a51a64521ed3871d52914bf532876`.
Variant: `classification/configs/vssm/vmambav2v_tiny_224.yaml` (VMamba-T s1l8).
Depths `[2,2,8,2]`; dimensions `[96,192,384,768]`; SSM ratio 1; state size 1.

## Usage

Run from `E:\vmanba_track` using `.venv\Scripts\python.exe`:

```python
import torch
from classification.models.tracking import vmamba_tiny_track

model = vmamba_tiny_track(
    pretrained='pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth'
).cuda()
template = torch.randn(1, 3, 128, 128, device='cuda')
search = torch.randn(1, 3, 256, 256, device='cuda')
template_features, search_features = model(template, search)
# template: (B,96,32,32), (B,192,16,16), (B,384,8,8), (B,768,4,4)
# search:   (B,96,64,64), (B,192,32,32), (B,384,16,16), (B,768,8,8)
```

Inputs are normalized RGB tensors (ImageNet mean/std, as in OSTrack), after target-centered crop/resize. Data loading is not implemented here. Outputs are raw pre-downsample stage features, not normalized by an added detection neck. The original entry point remains backbone-only. A two-stage fusion entry point is documented below. The optional tracker entry point adds the official OSTrack CENTER head. No loss, optimizer or dataset training is included.

## Interaction

Each VSS Block independently applies local operations to the two spatial grids, then uses one grouped selective-scan call. Official branch order:

1. Row forward: template then search.
2. Column forward: template then search.
3. Row reverse: search then template.
4. Column reverse: search then template.

Direction-specific splitting precedes each image's inverse cross merge. Stages downsample the two images independently. All backbone weights are shared and preserve official checkpoint keys; only classification-head parameters are excluded. Checkpoint loading is strict for backbone keys.

## Backend and environment

`.venv` is a local venv created with `--system-site-packages` from `D:\software\minanaconda\envs\learn`; it reuses existing dependencies read-only. No packages in the source environment were changed. This venv depends on that Python installation. `requirements-track.txt` lists runtime dependencies for a separate installation.

The code retains upstream CUDA selective-scan and Triton cross-scan/merge dispatch. It adds no per-token Python loop and does not force the slow backend. However this Windows/GTX 1650 machine has neither extension, so upstream's PyTorch fallback is used and emits warnings. Production-speed claims require measuring with the official compiled operators on a supported environment. Pair concatenation/merge adds memory copies and processing template costs extra work; zero overhead is not guaranteed.

`use_checkpoint=True` enables non-reentrant block checkpointing for paired training to reduce memory, at the cost of recomputation. It is off by default. A single-image call `model(search)` retains the original single-image block path and returns four maps.

Modified upstream files: `classification/models/vmamba.py` (paired branches); `classification/models/csm_triton.py` (guard missing optional Triton at import).
Added files: `classification/models/pair_scan.py`, `classification/models/tracking.py`.

Temporary validation scripts, benchmark outputs and Python bytecode produced by testing are removed after validation. This project and `.venv` are the retained deliverables.


## Stage 3 + Stage 4 search fusion

```python
from classification.models.tracking import vmamba_tiny_track_with_fusion

model = vmamba_tiny_track_with_fusion(
    pretrained='pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth'
).cuda()
fused_search = model(template, search)  # B,384,16,16 for a 256x256 search crop
features = model(template, search, return_features=True)
# keys: template_features, search_features, fused_search
```

`classification/models/tracking_fusion.py` implements two-level UPerHead-style
fusion: Stage 4 PPM with output grids (1,2,3,6), concatenation with original Stage 4,
3x3 projection to 384 channels, top-down addition into the Stage 3 lateral,
3x3 FPN convolution, concatenation of both levels at Stage 3 resolution, and a
final 3x3 projection to 384 channels. The upsampled Stage 4 tensor is reused.

The neck uses SyncBatchNorm and ReLU, matching the official segmentation
configuration. Single-GPU training requires batch >= 2 due to the 1x1 PPM branch;
batch-one inference uses model.eval(). Distributed SyncBN requires the normal
PyTorch DDP setup; multi-GPU synchronization was not tested on this machine. It depends only on PyTorch; no mmcv/mmseg installation is required.
Fusion weights are randomly initialized. Only backbone weights are loaded from
the existing classification checkpoint; no segmentation-head weights are claimed.

The existing `vmamba_tiny_track` and its four-stage output interface are unchanged.
The new wrapper exposes `model.backbone` and `model.fusion` for separate optimizer
learning rates. The fused output has 384 channels.

## OSTrack prediction head

```python
from classification.models.tracking import vmamba_tiny_tracker
model = vmamba_tiny_tracker(
    pretrained='pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth',
    # head_pretrained='path/to/OSTrack-256-ViT-B-checkpoint.pth.tar',
).cuda()
model.eval()
with torch.no_grad():
    predictions = model(template, search)
```

Fusion B,384,16,16 -> Conv2d(384,768,1,bias=True) -> official CenterPredictor
(inplanes=768, channel=256, feat_sz=16, stride=16).
Projection adds 295,680 trainable parameters. It has no extra normalization or activation.
Outputs: pred_boxes B,1,4 (normalized cx,cy,w,h relative to the search crop),
score_map B,1,16,16, size_map and offset_map B,2,16,16.
Optional gt_score_map is B,16,16, as in upstream. return_features=True also returns
the original template/search stage maps and fused_search.

Official source, pinned commit and license are retained in classification/models/ostrack_head.
Only its frozen_bn import path is adapted. CENTER forward and checkpoint keys are unchanged.
head_pretrained strictly loads complete compatible box_head.* weights, accepting net/model/state_dict
wrappers, module. prefixes or a plain head-only state dict. Incompatible/incomplete heads raise an error.
No OSTrack checkpoint has been downloaded or loaded by default; head, fusion and projection are
randomly initialized unless their weights are explicitly supplied. Synthetic compatibility tests
are not a test of a trained OSTrack checkpoint. New fusion/projection require tracking training.
The prior backbone-only and fusion-only entry points remain available.
