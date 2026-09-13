"""OSTrack CENTER objective, with explicit failure on invalid/nonfinite outputs."""
import torch
from torch import nn
from torch.nn.functional import l1_loss
from .upstream.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from .upstream.focal_loss import FocalLoss
from .upstream.heapmap_utils import generate_heatmap

class TrackerLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.focal=FocalLoss()

    def forward(self,prediction,annotations):
        if annotations.ndim!=3 or annotations.shape[0]!=1 or annotations.shape[-1]!=4:
            raise ValueError('Expected search annotations [1,B,4] normalized xywh')
        boxes=prediction['pred_boxes'].float()
        scores=prediction['score_map'].float()
        gt=annotations.float()
        if not torch.isfinite(gt).all() or not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise FloatingPointError('Nonfinite predictions or annotations')
        if (gt[...,2:]<=0).any() or (boxes[...,2:]<=0).any():
            raise ValueError('Bounding box widths and heights must be positive')
        if (scores<=0).any() or (scores>=1).any():
            raise ValueError('CENTER scores must be probabilities strictly between zero and one')
        # The official heatmap generator draws on CPU; avoid per-box GPU synchronization.
        heatmap=generate_heatmap(gt.detach().cpu(),self.config['search_size'],self.config['stride'])[-1].unsqueeze(1)
        heatmap=heatmap.to(scores.device,non_blocking=True)
        if scores.shape!=heatmap.shape: raise ValueError('Score/target heatmap shapes differ')
        queries=boxes.shape[1]
        predicted=box_cxcywh_to_xyxy(boxes).reshape(-1,4)
        target=box_xywh_to_xyxy(gt[-1])[:,None].repeat(1,queries,1).reshape(-1,4).clamp(0,1)
        giou,iou=giou_loss(predicted,target)
        l1=l1_loss(predicted,target)
        focal=self.focal(scores,heatmap)
        total=self.config['giou_weight']*giou+self.config['l1_weight']*l1+self.config['focal_weight']*focal
        if not torch.isfinite(total): raise FloatingPointError('Nonfinite tracking loss')
        return total,{'loss_total':total.detach(),'loss_giou':giou.detach(),
                      'loss_l1':l1.detach(),'loss_focal':focal.detach(),'mean_box_iou':iou.detach().mean()}
