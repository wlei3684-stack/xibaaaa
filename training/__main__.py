"""python -m training --config configs/train_got10k_60ep.json [--resume PATH]."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tracking_data import build_loader,seed_data,set_loader_epoch
from .losses import TrackerLoss
from .engine import build_optimizer,run_epoch,rng_state,save_checkpoint,load_checkpoint

ROOT=Path(__file__).resolve().parents[1]

def resolve(path):
    p=Path(path).expanduser(); return p if p.is_absolute() else ROOT/p

def read_json(path):
    with resolve(path).open(encoding='utf-8-sig') as f: return json.load(f)

def validate(config):
    t=config['training']; l=config['loss']; s=config['scheduler']
    if t['gradient_accumulation_steps']!=1: raise ValueError('This official-style loop supports accumulation_steps=1 only')
    if t['automatic_lr_scaling']: raise ValueError('Automatic LR scaling is not enabled for this recipe')
    if config['model']['freeze_modules']: raise ValueError('This recipe jointly trains all four modules')
    if config['model']['factory']!='classification.models.tracking.vmamba_tiny_tracker': raise ValueError('Unsupported model factory')
    if (l['search_size'],l['stride'],l['heatmap_size'])!=(256,16,16): raise ValueError('Current tracker requires search256/stride16')
    if l['pass_gt_heatmap_to_head'] or l['independent_size_offset_losses']: raise ValueError('Use official CENTER loss behavior')
    if l['clamp_target_xyxy']!=[0.,1.]: raise ValueError('Official target clamp is [0,1]')
    if s['type']!='StepLR' or s['step_when']!='after_training_epoch': raise ValueError('Unsupported scheduler')
    if config['validation']['best_checkpoint_metric']!='loss_total' or config['validation']['best_checkpoint_mode']!='min':
        raise ValueError('Best checkpoint uses minimum validation loss_total')
    for value in [t['epochs'],s['step_size'],config['validation']['interval_epochs'],config['checkpoint']['save_every_epochs']]:
        if not isinstance(value,int) or value<=0: raise ValueError('Epoch intervals must be positive integers')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume')
    args=parser.parse_args()
    config=read_json(args.config); validate(config)
    world=int(os.environ.get('WORLD_SIZE','1')); local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if not torch.cuda.is_available(): raise RuntimeError('GPU training requires CUDA and the VMamba scan dependencies')
    torch.cuda.set_device(local_rank); device=torch.device('cuda',local_rank)
    if world>1: dist.init_process_group(backend='nccl')
    rank=dist.get_rank() if dist.is_initialized() else 0
    try:
        seed_data(config['training']['seed']+rank)
        resume=args.resume or config['checkpoint']['resume']
        model_cfg=config['model']
        pretrained={}
        for name,flag in [('pretrained','require_pretrained_backbone'),('head_pretrained','require_pretrained_head')]:
            path=model_cfg.get(name)
            if not resume and model_cfg[flag] and not path: raise ValueError(f'Required pretrained path: {name}')
            if not resume and path and not resolve(path).is_file(): raise FileNotFoundError(f'Missing {name}: {resolve(path)}')
            pretrained[name]=str(resolve(path)) if path and not resume else None
        from classification.models.tracking import vmamba_tiny_tracker
        model=vmamba_tiny_tracker(**pretrained,use_checkpoint=model_cfg['use_checkpoint']).to(device)
        optimizer=build_optimizer(model,config['optimizer'])
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=config['scheduler']['step_size'],gamma=config['scheduler']['gamma'])
        scaler=(torch.amp.GradScaler('cuda',enabled=config['training']['amp'])
                if hasattr(torch.amp,'GradScaler') else torch.cuda.amp.GradScaler(enabled=config['training']['amp']))
        if world>1: model=DistributedDataParallel(model,device_ids=[local_rank],find_unused_parameters=False)
        loaders={}; resolved={}
        for mode,key in [('train','train_config'),('val','val_config')]:
            dc=read_json(config['data'][key])
            if bool(dc['training'])!=(mode=='train'): raise ValueError(f'Wrong training flag in {key}')
            if mode=='train' and dc['batch_size']<max(2,config['training']['minimum_batch_size_per_rank']):
                raise ValueError('SyncBN fusion training requires per-rank batch>=2 in this recipe')
            if (dc['template_size'],dc['search_size'])!=(128,256): raise ValueError('Tracker requires template128/search256')
            for spec in dc['datasets']:
                for field in ['root','annotations']:
                    if field in spec: spec[field]=str(resolve(spec[field]))
            resolved[mode]=dc; loaders[mode]=build_loader(dc)
            if len(loaders[mode])==0: raise ValueError(f'{mode} loader has no complete batch')
        config['resolved_data']=resolved
        epoch0=0; best=float('inf')
        if resume: epoch0,best=load_checkpoint(resolve(resume),model,optimizer,scheduler,scaler,config,rank,world)
        if epoch0>=config['training']['epochs']: raise ValueError('Resume checkpoint already reached configured final epoch')
        criterion=TrackerLoss(config['loss']).to(device)
        out=resolve(config['checkpoint']['output_dir'])
        if not resume and (out/'last.pth').exists():
            raise FileExistsError('Output already has last.pth; use --resume or a new output_dir')
        if rank==0:
            out.mkdir(parents=True,exist_ok=True)
            (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        for epoch in range(epoch0+1,config['training']['epochs']+1):
            set_loader_epoch(loaders['train'],epoch)
            rates={g['name']:g['lr'] for g in optimizer.param_groups}
            train=run_epoch(model,loaders['train'],criterion,device,optimizer,scaler,
                            clip=config['training']['gradient_clip_max_norm'],amp=config['training']['amp'])
            val=None; improved=False
            if epoch%config['validation']['interval_epochs']==0 or (epoch==config['training']['epochs'] and config['validation']['run_on_final_epoch']):
                set_loader_epoch(loaders['val'],epoch)
                val=run_epoch(model,loaders['val'],criterion,device,amp=config['training']['amp'])
                improved=val['loss_total']<best
                if improved: best=val['loss_total']
            scheduler.step()
            local_rng=rng_state(); states=[None]*world
            if world>1: dist.all_gather_object(states,local_rng)
            else: states=[local_rng]
            if rank==0:
                record={'epoch':epoch,'learning_rates':rates,'train':train,'val':val}
                with (out/'metrics.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                names=[]
                if config['checkpoint']['save_last_every_epoch']: names.append('last.pth')
                if epoch%config['checkpoint']['save_every_epochs']==0 or epoch==config['training']['epochs']: names.append(f'epoch_{epoch:04d}.pth')
                if improved and config['checkpoint']['save_best']: names.append('best.pth')
                for name in names: save_checkpoint(out/name,model,optimizer,scheduler,scaler,epoch,best,config,states)
            if world>1: dist.barrier()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

if __name__=='__main__': main()
