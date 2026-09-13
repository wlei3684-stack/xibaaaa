OSTrack loss utilities copied from commit 33b5e12586216b7fd0e95d255bd01ba44cbec759 (MIT).
The CENTER objective follows lib/train/actors/ostrack.py: 2 GIoU + 5 L1 + Focal,
normalized xywh annotation converted/clamped to xyxy, normalized cxcywh prediction,
official generate_heatmap with search256/stride16. No extra size/offset loss or
ground-truth heatmap passed into the head. Invalid/nonfinite losses raise explicitly
instead of the upstream actor's broad exception fallback to zero GIoU.
Heatmap generation runs on detached CPU boxes before moving the map to the score
device; loss math runs in FP32, including when optional backbone AMP is enabled.
CLI: python -m training --config configs/train_got10k_60ep.json
Mixed datasets: configs/train_mixed_60ep.json
DDP: torchrun --standalone --nproc_per_node=4 -m training --config configs/train_mixed_60ep.json
Resume: append --resume outputs/got10k_60ep/last.pth
Relative paths resolve from repository root. Configure dataset and checkpoint paths
before running. Required pretrained head must be OSTrack-256 ViT-B CENTER.
No OSTrack head checkpoint is bundled. A missing/incompatible checkpoint aborts.
Adjust per-rank batch/workers in referenced data JSONs for server resources.
Single-GPU SyncBN training requires batch>=2. Compiled scan operators determine speed.
Full train/validation metrics are sample-weighted, reduced across DDP ranks.
Validation retains official stochastic preprocessing and sampler behavior. Loss/IoU
validation is not a sequential tracking benchmark.
Scheduler steps after every completed epoch: epochs1-48 initial LR,49-60 x0.1.
Checkpoints save model,optimizer,scheduler,scaler,best metric,epoch,config and per-rank
Python/NumPy/Torch/CUDA RNG. Resume requires same world size, optimizer/loss/scheduler
and resolved data configuration. Exact reproducibility also depends on CUDA kernels.
Runs save last every epoch, numbered checkpoints every5 epochs and final epoch,
best at improving validation epochs, and metrics.jsonl. Existing runs require resume
or another output directory. No automatic fallback, hidden restart or LR scaling.

CPU loss/resume tests and one full GPU training/validation step passed on synthetic data.
Real dataset convergence, AMP training and multi-GPU execution are not yet validated.
