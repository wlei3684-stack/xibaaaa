"""Dataset adapters with explicit splits and lazy, bounded annotation caching."""
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import cv2

SPECS = Path(__file__).parent / 'data_specs'

def lines(path):
    return [s.strip() for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def numbers(path, columns=None):
    text = Path(path).read_text(encoding='utf-8-sig').replace(',', ' ')
    values = np.array([float(x) for x in text.split()], dtype=np.float32)
    if columns is not None:
        if values.size % columns:
            raise ValueError(f'{path}: expected {columns} columns per box')
        values = values.reshape(-1, columns)
    return values

class TrackingSource:
    def __init__(self, spec, training=True):
        self.kind = spec['type'].lower()
        self.name = spec.get('name', self.kind)
        self.root = Path(spec['root']).expanduser()
        if not self.root.is_dir():
            raise ValueError(f'{self.name}: dataset root does not exist: {self.root}')
        self.records = []
        self.cache = OrderedDict()
        self.is_image = self.kind == 'coco'
        if self.kind == 'got10k':
            names = lines(self.root / 'list.txt')
            if 'split_file' in spec:
                split = Path(spec['split_file'])
            elif spec.get('split') == 'official_val' and not training:
                split = None
            else:
                name = spec.get('split', 'train_full')
                allowed = {'train','val','train_full','vottrain','votval'}
                if name not in allowed:
                    raise ValueError(f'Unsupported GOT10k split: {name}')
                part = {'vottrain':'vot_train','votval':'vot_val'}.get(name, name)
                split = SPECS / f'got10k_{part}_split.txt'
            ids = [int(x) for x in lines(split)] if split else list(range(len(names)))
            if any(i < 0 or i >= len(names) for i in ids):
                raise ValueError(f'{self.name}: split indices do not match {self.root}/list.txt')
            self.records = [(names[i], self.root / names[i], None) for i in ids]
        elif self.kind == 'lasot':
            if 'split_file' in spec:
                names = lines(spec['split_file'])
            elif spec.get('split', 'train') == 'train':
                names = lines(SPECS / 'lasot_train_split.txt')
            elif spec.get('split') == 'test' and not training:
                raise ValueError('Supply an explicit LaSOT validation split_file; benchmark evaluation is separate')
            else:
                raise ValueError('LaSOT requires train or an explicit split_file')
            self.records = [(n, self.root / n.rsplit('-', 1)[0] / n, None) for n in names]
        elif self.kind == 'trackingnet':
            sets = spec.get('sets', list(range(12)))
            if not sets or any(not isinstance(i, int) or not 0 <= i < 12 for i in sets):
                raise ValueError('TrackingNet sets must be training subset numbers 0..11')
            for i in sets:
                base = self.root / f'TRAIN_{i}'
                annotations = sorted((base / 'anno').glob('*.txt'))
                if not annotations:
                    raise ValueError(f'No TrackingNet annotations: {base}/anno')
                self.records.extend((f'TRAIN_{i}/{p.stem}', base / 'frames' / p.stem, p) for p in annotations)
        elif self.kind == 'coco':
            with Path(spec['annotations']).open(encoding='utf-8') as f:
                coco = json.load(f)
            images = {i['id']:i['file_name'] for i in coco['images']}
            self.records = [(str(a['id']), self.root / images[a['image_id']], np.array(a['bbox'], np.float32))
                            for a in coco['annotations'] if not a.get('iscrowd', 0)
                            and len(a['bbox']) == 4 and np.isfinite(a['bbox']).all()
                            and a['bbox'][2] > 0 and a['bbox'][3] > 0]
        else:
            raise ValueError(f'Unknown dataset type: {self.kind}')
        if not self.records:
            raise ValueError(f'{self.name}: no sequences/objects selected')
        missing = [str(r[1]) for r in self.records if not r[1].exists()]
        if missing:
            raise ValueError(f'{self.name}: {len(missing)} missing paths; first: {missing[0]}')

    def __len__(self):
        return len(self.records)

    def annotation(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        name, path, extra = self.records[index]
        if self.is_image:
            boxes = extra.reshape(1,4)
        else:
            boxes = numbers(extra if self.kind == 'trackingnet' else path/'groundtruth.txt', 4)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:,2:] > 0).all(axis=1)
        for filename, visible_when_zero in ([('absence.label',True),('cover.label',False)] if self.kind == 'got10k'
                                            else [('full_occlusion.txt',True),('out_of_view.txt',True)] if self.kind == 'lasot' else []):
            flag = numbers(path/filename)
            if len(flag) != len(boxes):
                raise ValueError(f'{self.name}/{name}: {filename} length does not match boxes')
            valid &= np.isfinite(flag) & ((flag == 0) if visible_when_zero else (flag > 0))
        ids = np.flatnonzero(valid)
        if len(ids) < (1 if self.is_image else 2):
            raise ValueError(f'{self.name}/{name}: insufficient visible valid frames')
        result = boxes, ids
        self.cache[index] = result
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result

    def frame(self, index, frame_id):
        name, path, _ = self.records[index]
        if self.kind == 'got10k': path = path / f'{frame_id+1:08d}.jpg'
        elif self.kind == 'lasot': path = path / 'img' / f'{frame_id+1:08d}.jpg'
        elif self.kind == 'trackingnet': path = path / f'{frame_id}.jpg'
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'Cannot decode {self.name}/{name}, frame {frame_id}: {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
