"""Epoch execution, explicit parameter groups, and resumable checkpoints."""
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from tracking_data import model_inputs

def unwrap(model):
    return model.module if hasattr(model,'module') else model

def build_optimizer(model,config):
    if config['type']!='AdamW': raise ValueError('Only configured AdamW is supported')
    groups=[dict(name=g['name'],lr=g['lr'],params=[]) for g in config['parameter_groups']]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        matches=[i for i,g in enumerate(config['parameter_groups']) if name.startswith(g['parameter_prefix'])]
        if len(matches)!=1: raise ValueError(f'Parameter must match exactly one optimizer group: {name}')
        groups[matches[0]]['params'].append(p)
    if any(not g['params'] for g in groups): raise ValueError('Empty optimizer parameter group')
    return torch.optim.AdamW(groups,weight_decay=config['weight_decay'],betas=tuple(config['betas']),eps=config['eps'])

def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None,clip=0.1,amp=False,log_interval=50):
    training=optimizer is not None
    model.train(training)
    keys=['loss_total','loss_giou','loss_l1','loss_focal','mean_box_iou']
    totals=torch.zeros(len(keys)+1,dtype=torch.float64,device=device)
    if len(loader)==0: raise ValueError('DataLoader has no full batches; check samples/batch/world size')
    with torch.set_grad_enabled(training):
        for step,batch in enumerate(loader,1):
            template,search=model_inputs(batch)
            template=template.to(device,non_blocking=True); search=search.to(device,non_blocking=True)
            annotations=batch['search_anno'].to(device,non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16):
                prediction=model(template,search)
            # Evaluate focal/box losses in float32 even if backbone AMP is enabled.
            loss,stats=criterion(prediction,annotations)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
            count=search.shape[0]
            totals[:-1]+=torch.stack([stats[k] for k in keys]).double()*count
            totals[-1]+=count
            if log_interval and step%log_interval==0 and (not dist.is_initialized() or dist.get_rank()==0):
                print(f'{"train" if training else "val"} step {step}/{len(loader)} loss={loss.item():.6f}',flush=True)
    if dist.is_initialized(): dist.all_reduce(totals)
    return {k:float(v) for k,v in zip(keys,(totals[:-1]/totals[-1]).cpu())}

def rng_state():
    np_state=np.random.get_state()
    return {'python':random.getstate(),'numpy':(np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state['cuda'])

def save_checkpoint(path,model,optimizer,scheduler,scaler,epoch,best,config,rng_states):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model':unwrap(model).state_dict(),'optimizer':optimizer.state_dict(),
             'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
             'best':best,'config':config,'rng_states':rng_states,'world_size':len(rng_states)}
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary); os.replace(temporary,path)

def load_checkpoint(path,model,optimizer,scheduler,scaler,config,rank=0,world_size=1):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['world_size']!=world_size: raise ValueError('Resume requires unchanged world size for RNG/state consistency')
    for key in ['optimizer','scheduler','loss','data','resolved_data']:
        if state['config'].get(key)!=config.get(key): raise ValueError(f'Resume configuration differs: {key}')
    for key in ['amp','gradient_accumulation_steps','gradient_clip_max_norm','seed']:
        if state['config']['training'][key]!=config['training'][key]: raise ValueError(f'Resume training setting differs: {key}')
    unwrap(model).load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng_states'][rank])
    return state['epoch'],state['best']
