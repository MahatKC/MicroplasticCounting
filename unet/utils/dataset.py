from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageOps
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='', img_transforms=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.img_transforms = img_transforms
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, mode):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        #img_nd = np.array(pil_img)

        #if len(img_nd.shape) == 2:
        #    img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        #img_trans = img_nd.transpose((2, 0, 1))
        #if mode == "img":
        #    if img_trans.max() > 1:
        #        img_trans = img_trans / 255
        #elif mode == "mask":
        #    img_trans = np.where(img_trans > 10, 1, 0)

        if mode == "mask":
            img_nd = np.array(pil_img)

            if len(img_nd.shape) == 2:
                img_nd = np.expand_dims(img_nd, axis=2)

            # HWC to CHW
            img_trans = img_nd.transpose((2, 0, 1))
            img_trans = np.where(img_trans > 10, 1, 0)
            return img_trans

        elif mode == "img":
            return pil_img



        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0]).convert('L')
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        random_flip = np.random.normal()
        random_mirror = np.random.normal()

        if (random_flip > 0):
            mask = ImageOps.flip(mask)
            img = ImageOps.flip(img)
        if (random_mirror > 0):
            mask = ImageOps.mirror(mask)
            img = ImageOps.mirror(img)

        img = self.preprocess(img, self.scale, 'img')
        mask = self.preprocess(mask, self.scale, 'mask')
        #mask = np.where(mask > 0.1, 1, 0)

        return {
            'image': self.img_transforms(img),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
