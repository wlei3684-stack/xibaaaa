"""Bounded causal pair sampling; reproducible across worker counts and epochs."""
import warnings
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from .datasets import TrackingSource
from .processing import PairProcessing

class PairDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        specs = config.get('datasets', [])
        if not specs:
            raise ValueError('datasets must not be empty')
        weights = np.asarray(config.get('weights',[1]*len(specs)),dtype=np.float64)
        if weights.shape != (len(specs),) or not np.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('weights must contain one finite positive value per dataset')
        self.weights = weights/weights.sum()
        self.samples = config.get('samples_per_epoch',60000)
        self.attempts = config.get('max_attempts',20)
        self.gap = config.get('max_gap',200)
        self.seed = config.get('seed',0)
        for key,value in [('samples_per_epoch',self.samples),('max_attempts',self.attempts),('max_gap',self.gap)]:
            if not isinstance(value,int) or value<=0: raise ValueError(f'{key} must be a positive integer')
        if not isinstance(self.seed,int) or self.seed<0: raise ValueError('seed must be a nonnegative integer')
        self.training = config.get('training',True)
        self.sources = [TrackingSource(s,self.training) for s in specs]
        self.processing = PairProcessing(self.training)
        self.reported_failure = False

    def __len__(self): return self.samples

    def __getitem__(self,index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch if self.training else 0,index]))
        last_error = None
        for attempt in range(self.attempts):
            source = self.sources[int(rng.choice(len(self.sources),p=self.weights))]
            seq = int(rng.integers(len(source)))
            name = source.records[seq][0]
            try:
                boxes, visible = source.annotation(seq)
                if source.is_image:
                    t = s = 0
                else:
                    # Same causal constraint as OSTrack. Widen gap in increments of five
                    # when needed, computed directly instead of an unbounded retry loop.
                    pos = int(rng.integers(len(visible)-1))
                    t = int(visible[pos])
                    needed = int(visible[pos+1])-t+1
                    gap = self.gap + max(0,(needed-self.gap+4)//5)*5
                    end = int(np.searchsorted(visible,t+gap,side='left'))
                    s = int(visible[int(rng.integers(pos+1,end))])
                result = self.processing([source.frame(seq,t),source.frame(seq,s)],[boxes[t],boxes[s]],rng)
                result.update(dataset=source.name, sequence=name, frame_ids=torch.tensor([t,s]))
                return result
            except (OSError,ValueError,cv2.error) as error:
                last_error = f'{source.name}/{name}: {error}'
                if not self.reported_failure:
                    warnings.warn(f'Sample failed; retrying (limit {self.attempts}): {last_error}',stacklevel=2)
                    self.reported_failure = True
        raise RuntimeError(f'Failed to sample after {self.attempts} attempts. Last failure: {last_error}')

def worker_init(worker_id):
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

def build_loader(config):
    """Build one train or validation loader. Paths are resolved from process cwd."""
    batch = config.get('batch_size',2)
    workers = config.get('num_workers',4)
    if not isinstance(batch,int) or batch<=0: raise ValueError('batch_size must be positive')
    if not isinstance(workers,int) or workers<0: raise ValueError('num_workers must be nonnegative')
    dataset = PairDataset(config)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler = DistributedSampler(dataset,shuffle=dataset.training,seed=dataset.seed,drop_last=dataset.training) if distributed else None
    if dataset.training and len(dataset)//(sampler.num_replicas if sampler else 1)<batch:
        raise ValueError('samples_per_epoch is too small for one full batch per rank')
    generator = torch.Generator().manual_seed(dataset.seed)
    options = dict(batch_size=batch,num_workers=workers,sampler=sampler,shuffle=False,
                   drop_last=dataset.training,pin_memory=config.get('pin_memory',True),
                   worker_init_fn=worker_init,generator=generator)
    if workers:
        options.update(prefetch_factor=2,persistent_workers=False)
    return DataLoader(dataset,**options)

def set_loader_epoch(loader,epoch):
    """Call before each epoch, before creating the loader iterator (including DDP)."""
    if not isinstance(epoch,int) or epoch<0: raise ValueError('epoch must be nonnegative')
    loader.dataset.epoch = epoch
    if isinstance(loader.sampler,DistributedSampler): loader.sampler.set_epoch(epoch)
