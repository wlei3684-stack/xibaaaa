"""OSTrack-style augmentation with upstream crop and box-coordinate helpers."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from .processing_utils import sample_target, transform_image_to_crop

class PairProcessing:
    def __init__(self, training=True):
        self.training = training
        self.mean = torch.tensor([.485,.456,.406])[:,None,None]
        self.std = torch.tensor([.229,.224,.225])[:,None,None]

    def __call__(self, images, boxes, rng):
        gray = self.training and rng.random() < .05
        flip = self.training and rng.random() < .5
        output = {}
        for key, image, rawbox, size, factor, center, scale in zip(
                ('template','search'), images, boxes, (128,256), (2.,4.), (0.,3.), (0.,.25)):
            box = torch.as_tensor(rawbox,dtype=torch.float32).clone()
            if gray:
                image = np.repeat(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)[...,None],3,axis=2)
            if flip:
                image = image[:,::-1].copy()
                box[0] = image.shape[1] - 1 - box[0] - box[2]
            extract = box.clone()
            if self.training:
                wh = box[2:] * torch.from_numpy(np.exp(rng.normal(size=2)*scale).astype(np.float32))
                xy = box[:2] + .5*box[2:] + wh.prod().sqrt()*center*torch.from_numpy((rng.random(2)-.5).astype(np.float32))
                extract = torch.cat((xy-.5*wh,wh))
            # Reject pathological annotations before allocating a huge padded crop.
            crop_side = float(extract[2:].prod().sqrt()*factor)
            if not np.isfinite(crop_side) or crop_side < 1 or crop_side > 8*max(image.shape[:2]):
                raise ValueError(f'Invalid/oversized {key} crop: {crop_side}')
            crop, resize, mask = sample_target(image,extract,factor,size)
            mask = torch.from_numpy(mask)
            if F.interpolate(mask[None,None].float(),size=(size//16,size//16),mode='nearest').bool().all():
                raise ValueError(f'{key} crop contains only padding at head resolution')
            bbox = transform_image_to_crop(box,extract,resize,torch.tensor([size,size]),normalize=True)
            tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2,0,1))).float()
            brightness = rng.uniform(.8,1.2) if self.training else 1.
            tensor = (tensor * (brightness/255.)).clamp_(0,1)
            if self.training and rng.random() < .5:
                tensor = tensor.flip(-1); mask = mask.flip(-1)
                bbox[0] = 1 - bbox[0] - bbox[2]
            output[key+'_images'] = (tensor-self.mean)/self.std
            output[key+'_anno'] = bbox
            output[key+'_att'] = mask
        return output
