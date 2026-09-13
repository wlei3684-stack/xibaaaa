Based on OSTrack 33b5e12586216b7fd0e95d255bd01ba44cbec759 (MIT).
processing_utils.py and data_specs retain upstream logic/data; trailing blank lines are normalized.
Other files are a new implementation of its single-template/search training input flow.
Defaults: template128/factor2/jitter0; search256/factor4/center3/scale0.25;
joint grayscale0.05/flip0.5, independent brightness0.2/flip0.5, ImageNet normalization.
Deliberate differences: deterministic per-index RNG; validation has no jitter/augmentation;
visible causal sampling permits short videos with >=2 valid frames; bounded retries;
crop allocation limited to 8 times the larger image dimension; modern default collate.
This is not a claim of exact stochastic replication of the paper training recipe.
Boxes are normalized xywh (not cxcywh); they can extend outside [0,1] after crop.
template_images/search_images are B,3,128,128 / B,3,256,256.
template_anno/search_anno are B,4. *_att is a padding mask (True = padding).
The current model does not take these masks; they are returned for inspection.
Set training=false for deterministic validation pairs, separate from video evaluation.
Call set_loader_epoch(loader,epoch) before each epoch; persistent workers are disabled
so workers receive the updated epoch. DDP uses DistributedSampler when initialized.
Windows callers must create/iterate loaders inside an if __name__ == '__main__' guard.
Four dataset adapters support original image folders, not LMDB.
GOT root points to train (official_val root points to val); retain list.txt and label files.
LaSOT root contains category/sequence/img. Default split is official training names.
TrackingNet root contains TRAIN_0..TRAIN_11/frames and anno.
COCO root points to the image directory; annotations points to instances JSON.
Explicit split_file uses GOT integer indices or LaSOT sequence names, never mixed.
Custom split files must be disjoint for train and validation. No automatic split guessing.
Only source paths are checked up front; annotations are cached lazily (128 sequences/worker),
images are decoded on demand. No full-dataset integrity or accuracy validation is claimed.
