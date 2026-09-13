OSTrack source: https://github.com/botaoye/OSTrack/tree/33b5e12586216b7fd0e95d255bd01ba44cbec759
Official data components are under upstream/; MIT license is retained here.
The prior handwritten dataset adapters/processing/sampler have been removed.
Preserved: official dataset order/readers/COCO instance masks, split files, causal
sampling, >=20 frames and >4 visible frames for a one-template/search pair,
training augmentation, validation joint grayscale/flip AND crop jitter, normalized
xywh targets, TensorDict fields, frame-first collation, drop_last=True for both
loaders, training shuffle and official DistributedSampler behavior.
Fixes: relocated imports and explicit root paths; optional jpeg4py since the official
builder uses OpenCV; torch._six/collections compatibility; modern shared output
allocation preserving stack dimensions; configuration weight length checks (#114);
GOT split index validation (#96); finite retries and exception warnings for apparent
hangs (#137; this is mitigation, not a confirmed diagnosis of that issue).
Every sampler while-loop has a configurable limit, default max_attempts=20.
Successful sampling before that limit keeps official RNG order and behavior.
Exhaustion raises rather than silently changing sampling rules or hiding failures.
COCO permits explicit image_dir/annotation_file paths without changing annotation logic.
Missing dataset-specific labels remain errors; generic custom data is not silently
treated as GOT10k (#66). Custom datasets need a dedicated adapter.
Default JSON batch32/workers10 and mixed dataset order/ratios match the referenced
OSTrack-256 config. Adjust runtime batch/workers for server memory.
Single-GPU training with our SyncBN fusion requires batch>=2.
Run-level reproducibility: call seed_data(seed) once before constructing/iterating
loaders. set_loader_epoch(loader,epoch) forwards to DistributedSampler only.
Official validation sampling remains stochastic. Multi-GPU execution has not been tested.
Usage: loader=build_loader(config); batch=next(iter(loader)); t,s=model_inputs(batch).
The loader returns template_images[1,B,3,128,128], search_images[1,B,3,256,256].
model_inputs only removes the leading singleton frame dimension for our backbone.
Boxes retain [1,B,4] normalized xywh; no loss conversion or training loop is included.
Windows multiprocessing callers must use an if __name__ == '__main__' guard.
LaSOT class-name path parsing uses os.path for Windows/Linux equivalence.
