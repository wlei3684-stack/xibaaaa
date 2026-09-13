"""OSTrack data flow with explicit paths and targeted compatibility/error fixes."""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from .upstream.data.sampler import TrackingSampler
from .upstream.data.processing import STARKProcessing
from .upstream.data.loader import LTRLoader
from .upstream.data.image_loader import opencv_loader
from .upstream.data import transforms as tfm

def make_source(spec):
    root=str(Path(spec['root']).expanduser())
    if not Path(root).is_dir(): raise ValueError(f'Dataset root does not exist: {root}')
    kind=spec['type'].lower()
    if 'split_file' in spec:
        raise ValueError('Use official split names; custom splits need a separately reviewed dataset adapter')
    if kind=='got10k':
        from .upstream.dataset.got10k import Got10k
        split=spec.get('split','train_full')
        ds=Got10k(root,image_loader=opencv_loader,split=None if split=='official_val' else split)
    elif kind=='lasot':
        from .upstream.dataset.lasot import Lasot
        ds=Lasot(root,image_loader=opencv_loader,split=spec.get('split','train'))
    elif kind=='trackingnet':
        from .upstream.dataset.tracking_net import TrackingNet
        ds=TrackingNet(root,image_loader=opencv_loader,set_ids=spec.get('sets'))
    elif kind=='coco':
        from .upstream.dataset.coco_seq import MSCOCOSeq
        ds=MSCOCOSeq(root,image_loader=opencv_loader,split='train',version='2017',
                    image_dir=root if 'annotations' in spec else None,annotation_file=spec.get('annotations'))
    else: raise ValueError(f'Unsupported dataset: {kind}')
    if ds.get_num_sequences()==0: raise ValueError(f'Empty dataset: {root}')
    return ds

def make_processing(config):
    training=config.get('training',True)
    joint=tfm.Transform(tfm.ToGrayscale(probability=.05),tfm.RandomHorizontalFlip(probability=.5))
    normalize=tfm.Normalize(mean=config.get('mean',[.485,.456,.406]),std=config.get('std',[.229,.224,.225]))
    transform=(tfm.Transform(tfm.ToTensorAndJitter(.2),tfm.RandomHorizontalFlip_Norm(.5),normalize)
               if training else tfm.Transform(tfm.ToTensor(),normalize))
    # Official validation deliberately shares joint augmentation and crop jitter.
    return STARKProcessing(
        search_area_factor={'template':config.get('template_factor',2.),'search':config.get('search_factor',4.)},
        output_sz={'template':config.get('template_size',128),'search':config.get('search_size',256)},
        center_jitter_factor={'template':config.get('template_center_jitter',0),'search':config.get('search_center_jitter',3)},
        scale_jitter_factor={'template':config.get('template_scale_jitter',0),'search':config.get('search_scale_jitter',.25)},
        mode='sequence',transform=transform,joint_transform=joint)

def build_loader(config):
    specs=config.get('datasets',[]); weights=config.get('weights',[1]*len(specs))
    if not specs or len(weights)!=len(specs) or not np.isfinite(weights).all() or any(w<0 for w in weights) or sum(weights)<=0:
        raise ValueError('datasets and weights must match; weights must be finite/nonnegative with positive sum')
    batch=config.get('batch_size',32); workers=config.get('num_workers',10)
    samples=config.get('samples_per_epoch',60000); attempts=config.get('max_attempts',20)
    for name,n in [('batch_size',batch),('samples_per_epoch',samples),('max_attempts',attempts),('max_gap',config.get('max_gap',200))]:
        if not isinstance(n,int) or n<=0: raise ValueError(f'{name} must be positive integer')
    if not isinstance(workers,int) or workers<0: raise ValueError('num_workers must be nonnegative integer')
    training=config.get('training',True)
    data=TrackingSampler([make_source(s) for s in specs],p_datasets=weights,samples_per_epoch=samples,
                         max_gap=config.get('max_gap',200),num_search_frames=1,num_template_frames=1,
                         processing=make_processing(config),frame_sample_mode='causal',train_cls=False)
    data.max_attempts=attempts
    distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler=DistributedSampler(data) if distributed else None
    loader=LTRLoader('train' if training else 'val',data,training=training,batch_size=batch,
                     shuffle=(training and not distributed),num_workers=workers,drop_last=True,stack_dim=1,
                     sampler=sampler,epoch_interval=1 if training else config.get('val_epoch_interval',20))
    return loader

def seed_data(seed):
    """Optional run-level seed; sampling uses the official global RNGs."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def set_loader_epoch(loader,epoch):
    if isinstance(loader.sampler,DistributedSampler): loader.sampler.set_epoch(epoch)

def model_inputs(batch):
    """Adapt official [frames,B,C,H,W] to our single-template/search model."""
    t,s=batch['template_images'],batch['search_images']
    if t.ndim!=5 or s.ndim!=5 or t.shape[0]!=1 or s.shape[0]!=1:
        raise ValueError('Expected one template/search in official frame-first format')
    return t[0],s[0]
