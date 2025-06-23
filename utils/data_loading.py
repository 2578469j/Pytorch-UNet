import logging
import time
import numpy as np
from patchify import patchify
import torch
from PIL import Image, ImageOps
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        im = Image.open(filename)
        im = ImageOps.exif_transpose(im).convert("RGB")
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
    def __init__(self, images_dir, mask_dir, test_ids=[]): #'3', '6', '9', '12'
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.test_ids = test_ids
        self.model_splits = {}

        self.features = {
            "full": "full.png",
            "sized": "sized.png",
            "opacity": "opacity.png",
            "dbscan": "dbscan.png",
            "gt": "gt.png",
            "gt_mask": "gt_mask.png",
            "dbscantuned": "dbscantuned.png",
            "front": "front.JPG",
            "front_gt": "front_annotated.png"
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
            id = fname

            model_split = self.model_splits.get(id, [])

            split = {
                'id': id,
            }

            model_split.append(split)
            self.model_splits[id] = model_split

    def get_ids_with_feature(self, feature):
        if feature not in self.features.keys():
            return []
        
        ids = []
        for id, _ in self.model_splits.items():
            if id in self.test_ids:
                pass
            try:
                if self.features[feature] in listdir(self.images_dir / id):
                    ids.append(f"{id}")
            except Exception as e:
                pass
        return ids

    def get_ids(self):
        ids = []
        for id, _ in self.model_splits.items():
            if id not in self.test_ids:
                ids.append(f"{id}")
        return ids
    
    def get_feature_fpaths(self, id, features):
        feature_fpaths = {}

        for feature in features:
            feature_fpaths[feature] = f"{id}_{self.features[feature]}"
        return feature_fpaths

    def get_mask_fpath(self, id):
        return f"{id}.png"
    
    # def get_full_fpath(self, id):
    #     return f"{id}_{self.features['full']}"
    
    # def get_gt_fpath(self, id):
    #     return f"{id}_{self.features['gt']}"
    
    # def get_opacity_fpath(self, id):
    #     return f"{id}_{self.features['opacity']}"

    # def get_dbscan_fpath(self, id):
    #     return f"{id}_{self.features['dbscan']}"

    def get_mask_fpaths(self):
        mask_fpaths = []
        for id, splits in self.model_splits.items():
            for split in splits:
                mask_fpaths.append(f"{id}.png")
        return mask_fpaths

    def get_full_fpaths(self):
        full_fpaths = []
        for id, splits in self.model_splits.items():
            for split in splits:
                full_fpaths.append(f"{id}_{split['x']}_{split['y']}_{self.features['full']}")
        return full_fpaths

class GemsyPatchSplitLoader():
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
            "gt_mask": "gt_mask.png",
            "dbscantuned": "dbscantuned.png"
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
        return ids #[:200]
    
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

class GemsyPatchDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', transform=None):
        self.images_dir = Path(images_dir) # Path to already pre-patched images
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.transform = transform

        self.split_loader = GemsyPatchSplitLoader(self.images_dir, self.mask_dir)

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

        if self.transform:
            img = img.transpose(1, 2, 0).astype(np.float32)
            mask = mask.astype(np.uint8)

            augmented = self.transform(image=img, mask=mask)
            img = augmented['image'].float().contiguous()
            mask = augmented['mask'].long().contiguous()
        else:
            img = torch.as_tensor(img.copy()).float().contiguous()
            mask = torch.as_tensor(mask.copy()).long().contiguous()

        return {
            'image': img,
            'mask': mask,
        }
    
class GemsyDataset(Dataset):
    def __init__(self, images_dir:str, mask_dir:str, patch_size=512, patches_per_image=300, scale=1.0, features = ['gt'], defect_focus_rate = 0.7, transform=None, ids=None, validation=False, target_size=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.target_size = target_size
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.features = features
        self.defect_focus_rate = defect_focus_rate
        self.validation = validation
        self.overlap = 0
        
        self.cached_masks = {}
        self.cached_imgs = {}
        self.cached_img_sizes = {}

        logging.info('Scanning mask files to determine unique values')
        self.split_loader = GemsySplitLoader(self.images_dir, self.mask_dir)
        self.ids = ids if ids is not None else self.split_loader.get_ids()
        logging.info(f'Creating dataset with {len(self.ids) * self.patches_per_image} examples')
        self.mask_values = self.split_loader.unique_gemsy_mask_values()
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        if self.validation:
            return len(self.ids)
        else:
            return len(self.ids) * self.patches_per_image

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, target = None):
        w, h = pil_img.size # 3000 4500
        if target:
            if w > h:
                newW, newH = target
            else:
                newH, newW = target
        else:
            newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        if w != newW or h != newH:
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int16)
            for k, v in mask_values.items():
                i = 1 if k in ['dot', 'scratch', 'crack', 'smudge', 'surfaceImperfection', 'missingGlass'] else 0
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
            return img.astype('float16')

    def extract_patch(self, img, mask, defect_focus=True):
        H, W = mask.shape
        ph, pw = self.patch_size, self.patch_size

        if defect_focus and (mask == 1).any():
            # Get defect pixel indices
            yx = np.argwhere(mask == 1)
            y, x = yx[np.random.randint(len(yx))]
            top = max(0, y - ph // 2)
            left = max(0, x - pw // 2)
        else:
            top = np.random.randint(0, H - ph + 1)
            left = np.random.randint(0, W - pw + 1)

        img_patch = img[:, top:top+ph, left:left+pw]
        mask_patch = mask[top:top+ph, left:left+pw]

        return img_patch, mask_patch

    def extract_patches(self, imgs, mask, defect_focus=True):
        H, W = mask.shape
        ph, pw = self.patch_size, self.patch_size

        if defect_focus and (mask == 1).any():
            # Get defect pixel indices
            yx = np.argwhere(mask == 1)
            y, x = yx[np.random.randint(len(yx))]
            top = max(0, y - ph // 2)
            left = max(0, x - pw // 2)
        else:
            top = np.random.randint(0, H - ph + 1)
            left = np.random.randint(0, W - pw + 1)

        img_patches = imgs[:, top:top+ph, left:left+pw]
        mask_patch = mask[top:top+ph, left:left+pw]

        return img_patches, mask_patch

    def extract_patches_fast(self, imgs, mask, defect_focus=True):
        H, W = mask.shape[-2], mask.shape[-1]
        ph, pw = self.patch_size, self.patch_size

        if defect_focus and torch.any(mask > 0):
            # Find defect pixels (nonzero mask values)
            yx = torch.nonzero(mask > 0, as_tuple=False)

            rand_idx = torch.randint(0, yx.shape[0], (1,))

            y, x = yx[rand_idx][0]

            top = max(0, min(H - ph, y - ph // 2))
            left = max(0, min(W - pw, x - pw // 2))
        else:
            top = torch.randint(0, H - ph + 1, (1,)).item()
            left = torch.randint(0, W - pw + 1, (1,)).item()

        img_patches = imgs[:, top:top+ph, left:left+pw]
        mask_patch = mask[top:top+ph, left:left+pw]
        return img_patches, mask_patch

    def get_prediction_data_predict(self, id, overlap=0):
        img_fnames = self.split_loader.get_feature_fpaths(id, self.features)
        mask_fname = self.split_loader.get_mask_fpath(id)
        mask = load_image(self.mask_dir / mask_fname)

        # Feature fetching
        imgs = []
        og_img = None
        size = None
        i = 0
        for feature, img_fname in img_fnames.items():
            # Load image from raw
            img = load_image(self.images_dir / img_fnames[feature])
            if i == 0:
                og_img = img
            img = self.preprocess(None, img, self.scale, is_mask=False, target=self.target_size)
            size = img.shape

            imgs.append(img)
        img_stack_np = np.concatenate(imgs, axis=0)  # [C_total, H, W]

        step = int(self.patch_size * (1-overlap))
        img_stack_swapped = np.moveaxis(img_stack_np, 0, -1)
        patches = patchify(img_stack_swapped, (self.patch_size, self.patch_size, len(self.features)*3), step=step)
        num_y, num_x = patches.shape[:2]
        patches = patches.reshape(-1, self.patch_size, self.patch_size, len(self.features)*3)
        patches = np.moveaxis(patches, -1, 1)

        stacked_img_patches = torch.from_numpy(patches).to(torch.float16)
        
        return stacked_img_patches.contiguous(), size, og_img, mask
    
    def get_prediction_data(self, img_idx):
        name = self.ids[img_idx]

        img_fnames = self.split_loader.get_feature_fpaths(name, self.features)
        mask_fname = self.split_loader.get_mask_fpath(name)

        preprocessed_mask_path = self.mask_dir / ".preprocessed" / f"{mask_fname}.pt"
        preprocessed_img_paths = {feature: self.images_dir / ".preprocessed" / f"{img_fname}.pt" for feature, img_fname in img_fnames.items()}

        imgs = []

        if img_idx in self.cached_masks:
            mask_tensor = self.cached_masks[img_idx]
            img_stack_np = self.cached_imgs[img_idx]
            img_size = self.cached_img_sizes[img_idx]
        else:
            # Mask fetching
            #mask_size = None
            if preprocessed_mask_path.exists():
                mask = torch.load(preprocessed_mask_path, weights_only=False)
            else:
                mask = load_image(self.mask_dir / mask_fname)
                mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, target=self.target_size)
                # Save preprocessed tensors to disk
                preprocessed_mask_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(mask, preprocessed_mask_path)

            # Store in memory cache
            mask_tensor = torch.from_numpy(mask).to(torch.float16).unsqueeze(0)
            self.cached_masks[img_idx] = mask_tensor
            self.cached_img_sizes[img_idx] = mask.shape
            img_size = mask.shape

            # Add the mask as first tensor, for patchifying
            imgs.append(mask_tensor)
            # Feature fetching
            for feature, ppath in preprocessed_img_paths.items():
                # Load from disk if available
                if ppath.exists():
                    img = torch.load(ppath, weights_only=False)
                else:
                    # Load image from raw
                    img = load_image(self.images_dir / img_fnames[feature])
                   # assert img.size == mask_size, \
                   #     f'Image and mask {name} should be the same size, but are {img.size} and {mask_size}'
                    img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, target=self.target_size)
                    # Save preprocessed tensors to disk
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(img, ppath)
    
                imgs.append(img)

            # Store in memory cache
            img_stack_np = np.concatenate(imgs, axis=0)  # [C_total, H, W]
            #stacked_img = torch.from_numpy(img_stack_np).to(torch.float16)
            self.cached_imgs[img_idx] = img_stack_np

        step = int(self.patch_size * (1-self.overlap))
        img_stack_swapped = np.moveaxis(img_stack_np, 0, -1)
        patches = patchify(img_stack_swapped, (self.patch_size, self.patch_size, len(self.features)*3+1), step=step)
        #num_y, num_x = patches.shape[:2]
        patches = patches.reshape(-1, self.patch_size, self.patch_size, len(self.features)*3+1)
        patches = np.moveaxis(patches, -1, 1)

        stacked_img_patches = torch.from_numpy(patches).to(torch.float32)
        
        return stacked_img_patches.contiguous(), img_size
        

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        name = self.ids[img_idx]

        mask_fname = self.split_loader.get_mask_fpath(name)
       # img_fnames = self.split_loader.get_gt_fpath(name, self.features)
        img_fnames = self.split_loader.get_feature_fpaths(name, self.features)
        #img_fname = self.split_loader.get_full_fpath(name)
        #img_fname = self.split_loader.get_opacity_fpath(name)
        #img_fname = self.split_loader.get_dbscan_fpath(name)
        # Build preprocessed cache paths
        preprocessed_mask_path = self.mask_dir / ".preprocessed" / f"{mask_fname}.pt"
        preprocessed_img_paths = {feature: self.images_dir / ".preprocessed" / f"{img_fname}.pt" for feature, img_fname in img_fnames.items()}
        #preprocessed_img_paths['mask'] = mask_fname
        imgs = []
 
        if img_idx in self.cached_masks:
            mask_tensor = self.cached_masks[img_idx]
            stacked_img = self.cached_imgs[img_idx]
        else:
            # Mask fetching
            #mask_size = None
            if preprocessed_mask_path.exists():
                mask = torch.load(preprocessed_mask_path, weights_only=False)
            else:
                mask = load_image(self.mask_dir / mask_fname)
                # 4000 6000
                mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, target=self.target_size)
                # 4000 6000
                # Save preprocessed tensors to disk
                preprocessed_mask_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(mask, preprocessed_mask_path)

            # Store in memory cache
            mask_tensor = torch.from_numpy(mask).to(torch.float16).squeeze(0)
            self.cached_masks[img_idx] = mask_tensor

            # Feature fetching
            for feature, ppath in preprocessed_img_paths.items():
                # Load from disk if available
                if ppath.exists():
                    img = torch.load(ppath, weights_only=False)
                else:
                    # Load image from raw
                    img = load_image(self.images_dir / img_fnames[feature])
                   # 6000 4000
                   # assert img.size == mask_size, \
                   #     f'Image and mask {name} should be the same size, but are {img.size} and {mask_size}'
                    img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, target=self.target_size)
                    # Save preprocessed tensors to disk
                    ppath.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(img, ppath)
    
                imgs.append(img)
            # Store in memory cache
            img_stack_np = np.concatenate(imgs, axis=0)  # [C_total, H, W]
            stacked_img = torch.from_numpy(img_stack_np).to(torch.float16)
            self.cached_imgs[img_idx] = stacked_img

        if self.validation:
            defect_focus = False
        else:
            defect_focus = np.random.rand() < self.defect_focus_rate

        # Check that mask and img is aligned
        assert stacked_img.shape[1:] == mask_tensor.shape

        stacked_img_patches, mask_patch = self.extract_patches_fast(stacked_img, mask_tensor, defect_focus=defect_focus)

        if self.transform:
            # Albumentations expects HWC
            stacked_img_patches = stacked_img_patches.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            mask_patch = mask_patch.cpu().numpy().astype(np.uint8)
            augmented = self.transform(image=stacked_img_patches, mask=mask_patch)
             #
            img_patch = augmented['image'].float().contiguous() #6 512 512
            mask_patch = augmented['mask'].long().contiguous()
        else:
            img_patch = stacked_img_patches.contiguous()
            mask_patch = mask_patch.contiguous()
            #img_patch = torch.tensor(stacked_img_patches.copy()).float().contiguous() # 3 512 512
            #mask_patch = torch.tensor(mask_patch.copy()).long().contiguous()

        assert img_patch.shape[-1] == self.patch_size
        assert mask_patch.shape[-1] == self.patch_size

        return {
            'image': img_patch,
            'mask': mask_patch,
        }