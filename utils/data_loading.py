import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        im = Image.open(filename)
        #print(im.size)
        return im #Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class GemsySplitLoader():
    def __init__(self, images_dir, mask_dir):
        self.images_dir = images_dir
        self.mask_dir = mask_dir

        self.model_splits = {}

        self.features = {
            "full": "full.png",
            "sized": "sized.png",
            "opacity": "opacity.png",
            "dbscan": "dbscan.png",
            "gt": "gt.png",
            "gt_mask": "gt_mask.png"
        }

        self.pallete = {
            "dot": [237, 28, 36],
            "scratch": [255, 127, 39],
            "crack": [255, 242, 0],
            "smudge": [63, 72, 204],
            "surfaceImperfection": [185, 122, 87],
            "missingGlass": [136, 0, 21],
            "camera": [181, 230, 29],
            "border": [255, 174, 201],
            "intBorder": [112, 146, 190],
            "button": [34, 177, 76],
            "phoneFeat": [153, 217, 234],
            "speaker": [200, 191, 231]
        }
        self.load_ids()

    def unique_gemsy_mask_values(self):
        return self.pallete
        #return np.array([v for k, v in self.pallete.items()])

    def load_ids(self):
        for fname in listdir(self.mask_dir):
            if fname.startswith('.'):
                continue

            fname = splitext(fname)[0]
            splits = fname.split('_')

            id = splits[0]
            x = splits[1]
            y = splits[2]
            feature = splits[3]

            model_split = self.model_splits.get(id, [])

            split = {
                'id': id,
                'x': x,
                'y': y
            }

            model_split.append(split)
            self.model_splits[id] = model_split

    def get_ids(self):
        ids = []
        for id, splits in self.model_splits.items():
            for split in splits:
                ids.append(f"{id}_{split['x']}_{split['y']}")
        return ids #[:20]
    
    def get_mask_fpath(self, id):
        return f"{id}_{self.features['gt_mask']}"
    
    def get_full_fpath(self, id):
        return f"{id}_{self.features['full']}"
    
    def get_gt_fpath(self, id):
        return f"{id}_{self.features['gt']}"

    def get_mask_fpaths(self):
        mask_fpaths = []
        for id, splits in self.model_splits.items():
            for split in splits:
                mask_fpaths.append(f"{id}_{split['x']}_{split['y']}_{self.features['gt_mask']}")
        return mask_fpaths

    def get_full_fpaths(self):
        full_fpaths = []
        for id, splits in self.model_splits.items():
            for split in splits:
                full_fpaths.append(f"{id}_{split['x']}_{split['y']}_{self.features['full']}")
        return full_fpaths


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

class GemsyDataset():
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.split_loader = GemsySplitLoader(self.images_dir, self.mask_dir)

        self.ids = self.split_loader.get_ids()
        #self.ids = [splitext(file)[0] for file in listdir(mask_dir) if not file.startswith('.')]
        #filenames = [splitext(file)[0] for file in listdir(mask_dir) if not file.startswith('.')]
        #self.ids = list(set([filename.split('_')[0] for filename in filenames]))

        if not self.ids:
            raise RuntimeError(f'No input file found in {mask_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        #with Pool() as p:
        #    unique = list(tqdm(
        #        p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #        total=len(self.ids)
         #   ))

        self.mask_values = self.split_loader.unique_gemsy_mask_values()#list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)



        # self.pallete = {
        #     "dot": [237, 28, 36],
        #     "scratch": [255, 127, 39],
        #     "crack": [255, 242, 0],
        #     "smudge": [63, 72, 204],
        #     "surfaceImperfection": [185, 122, 87],
        #     "missingGlass": [136, 0, 21],
        #     "camera": [181, 230, 29],
        #     "border": [255, 174, 201],
        #     "intBorder": [112, 146, 190],
        #     "button": [34, 177, 76],
        #     "phoneFeat": [153, 217, 234],
        #     "speaker": [200, 191, 231]
        # }

        if is_mask:
            # TODO: TEMP BINARY ENCODING - REMOVE LATER
            mask = np.zeros((newH, newW), dtype=np.int64)
            for k, v in mask_values.items():
                if k in ['dot', 'scratch', 'crack', 'smudge', 'surfaceImperfection', 'missingGlass']:
                    i = 1
                else:
                    i = 0
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        
            # mask = np.zeros((newH, newW), dtype=np.int64)
            # for i, v in enumerate(mask_values):
            #     if img.ndim == 2:
            #         mask[img == v] = i
            #     else:
            #         mask[(img == v).all(-1)] = i

            # return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_fname = self.split_loader.get_mask_fpath(name)
        #full_fname = self.split_loader.get_full_fpath(name)
        gt_fname = self.split_loader.get_gt_fpath(name)

        #mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        #img_file = list(self.images_dir.glob(name + '_full'+ '.*'))

        mask_file = [self.mask_dir / mask_fname] #list(self.mask_dir.glob(mask_fname))
        img_file = [self.images_dir / gt_fname] #list(self.images_dir.glob(full_fname))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }