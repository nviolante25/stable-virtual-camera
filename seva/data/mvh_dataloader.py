import os
import json
import pickle
import glob
from typing import Tuple, Optional, Dict, Union, Callable
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image
import torchvision.transforms.v2 as T
import pytorch_lightning as pl
from seva.data.preprocessing import (
    update_intrinsics,
    get_bbox_center_and_size,
    get_mvhumannet_extrinsics,
    load_json,
    load_pickle,
    update_intrinsics_resize,
    generate_gaussian_mixture_samples,
    generate_gaussian_samples
)

class RandomBBoxCrop(object):
    def __init__(
        self,
        center_sampler: Callable, # simply takes a batchsize B and returns B sampled centers
        length_sampler: Callable, # same as above... returns B sampled lengths
    ):
        """
        Random crop transform centered around a bounding box with Gaussian mixture sampling.
        Sampler functions have baked-in distribution parameters.
        Expected OFFSETS (zero mean) instead of pixel-space means.
        """

        self.center_sampler = center_sampler
        self.length_sampler = length_sampler

    def _get_crop_params(
        self, 
        bbox: torch.Tensor, 
        K: torch.Tensor,
    ) -> Tuple[int, int, int, int, torch.Tensor]:
        """
        Calculate crop parameters based on bbox and intrinsics.
        NOTE: `pre_scale` will affect bbox parameters here.
        
        Args:
            bbox: Tensor of shape (4,) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (3, 3)
            
        Returns:
            (x1, y1, x2, y2): Crop coordinates
            K_new: Updated intrinsics matrix
        """
        # get bbox center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # sample new center and length of crop
        center_sample = self.center_sampler(1)[0]
        size_sample = self.length_sampler(1)[0]
        
        # calculate crop coordinates
        center_x, center_y = map(int, center_sample)
        crop_size = int(size_sample[0])

        # "clamp" center within bbox
        x1 = center_x - (crop_size // 2)
        y1 = center_y - (crop_size // 2)
        x2 = center_x + (crop_size // 2)
        y2 = center_y + (crop_size // 2)
        
        # update intrinsics
        K_new = update_intrinsics(
            torch.as_tensor(K), 
            crop_x=x1, 
            crop_y=y1, 
            scale=1, # for MVHumanNet images (downsampled)
            crop_first=False,
            padding_mode=True
        )

        # crop parameters, updated intrinsics
        return (x1, y1, x2, y2), K_new

    def __call__(
        self, 
        image: torch.Tensor, 
        bbox: torch.Tensor, 
        K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: Tensor of shape (C, H, W)
            bbox: Tensor of shape (4,) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (3, 3)
            
        Returns:
            Cropped image and updated intrinsics matrix
        """
        crop_params, K_new = self._get_crop_params(bbox, K)
        x1, y1, x2, y2 = crop_params
        
        # Handle padding if needed
        H, W = image.shape[1:3]
        pad_left = int(max(0, -x1))
        pad_top = int(max(0, -y1))
        pad_right = int(max(0, x2 - W))
        pad_bottom = int(max(0, y2 - H))
        
        # if the crop parameters extend beyond the image, pad the image
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padding = [pad_left, pad_right, pad_top, pad_bottom]
            image = T.Pad(padding)(image)
            
            # Adjust crop coordinates
            x1, x2 = int(x1 + pad_left), int(x2 + pad_left)
            y1, y2 = int(y1 + pad_top), int(y2 + pad_top)
            
            # and then update the intrinsics from padding (left or top; 
            # (negative because negative cropping is positive padding)
            # if right/bottom, no need to update intrinsics
            K_new = update_intrinsics(
                K_new,
                crop_x=-pad_left,
                crop_y=-pad_top,
                scale=1,
                crop_first=False,
                padding_mode=True
            )
        image_ = torch.as_tensor(image)
        image = image_[:, y1:y2, x1:x2] # actual crop
        return image, K_new


class MVHumanNetDataset(Dataset):
    def __init__(self, root_dir, transforms=None, pre_scale=0.5):
        self.root_dir = root_dir             # directory of all subject directories
        self.image_paths = []                # main image data paths
        self.mask_paths = []                 # (+ masks)
        self.annots_paths = []               # bbox parameters in stored here
        # These are per subject rather than per timestep
        self.subject_projective_params = {}  # Dict[subject: (extrinsics, intrinsics)]
        self.camera_scale = {}               # float, to be multiplied with camera center.
        self.transforms = transforms         # transforms for the random crop
        self.pre_scale = pre_scale           # since MVHumanNet is downsampled, update intrinsics

        # get paths to relevant data
        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            # get subject metadata
            extrinsics_path = os.path.join(subject_path, 'camera_extrinsics.json')
            intrinsics_path = os.path.join(subject_path, 'camera_intrinsics.json')
            extrinsics = load_json(extrinsics_path)
            intrinsics = load_json(intrinsics_path)

            self.camera_scale[subject] = load_pickle(os.path.join(subject_path, 'camera_scale.pkl'))
            self.subject_projective_params[subject] = {
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }

            # annots, images, masks share the same camera directory names
            annots_path = os.path.join(subject_path, 'annots')
            images_path = os.path.join(subject_path, 'images_lr')
            masks_path = os.path.join(subject_path, 'fmask_lr')

            # for each camera
            for camera in os.listdir(masks_path): # NOTE: listdir is arbitrary order
                # retrieve data for each timestep
                if not os.path.isdir(os.path.join(masks_path, camera)):
                    continue
                for timestep in os.listdir(os.path.join(masks_path, camera)):
                    if timestep.endswith('.png'):
                        self.image_paths.append(os.path.join(images_path, camera, timestep.replace('_fmask.png', '.jpg')))
                        self.mask_paths.append(os.path.join(masks_path, camera, timestep))
                        self.annots_paths.append(os.path.join(annots_path, camera, timestep.replace('_fmask.png', '.json')))
    

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        annots_path = self.annots_paths[idx]
        subject = img_path.split('/')[1]
        camera = img_path.split('/')[-2]

        extrinsics = self.subject_projective_params[subject]['extrinsics'][f"1_{camera}.png"]
        transform_matrix = get_mvhumannet_extrinsics(extrinsics, scale=self.camera_scale[subject])
        intrinsics = torch.tensor(self.subject_projective_params[subject]['intrinsics']['intrinsics'])
        if self.pre_scale != 1:
            intrinsics = update_intrinsics_resize(intrinsics, scale=self.pre_scale)

        # load in and mask image
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        annots = load_json(annots_path)

        # bbox params useful for cropping
        H, W = int(annots['height'] * self.pre_scale), int(annots['width'] * self.pre_scale) # images not yet scaled down
        bbox = torch.tensor(annots['annots'][0]['bbox'][:4]) * self.pre_scale # [x1, y1, x2, y2]
        (center_x, center_y), (size_x, size_y) = get_bbox_center_and_size(bbox)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Create masked image by using PIL's composite
        background = Image.new('RGB', img.size, (0, 0, 0))
        masked_img = Image.composite(img, background, mask)

        # distributions to sample
        # TODO: make these arguments to the dataloader
        # additionally, add the option to not crop
        def center_sampler(batch_size):
            # mean at center of bbox
            mean = torch.tensor([x1 + size_x // 2, y1 + size_y // 2], dtype=torch.float32)
            cov = torch.tensor([[size_x, 0], [0, size_y]], dtype=torch.float32)
            weights = torch.tensor([0.7, 0.3])
            # return generate_gaussian_samples(mean, cov, batch_size)
            return generate_gaussian_mixture_samples([mean, (x1 + size_x // 2, y1)], [cov, cov], weights, batch_size)

        def length_sampler(batch_size):
            # Example: Sample from a 1D Gaussian for crop size
            mean = torch.tensor([(size_x + size_y) // 1.5], dtype=torch.float32)  # mean is the smallest dim
            cov = torch.tensor([[max(size_x, size_y)]], dtype=torch.float32)
            return generate_gaussian_samples(mean, cov, batch_size)

        # random crop the image
        random_cropper = RandomBBoxCrop(center_sampler, length_sampler)
        cropped_image, updated_K = random_cropper(
            T.functional.to_tensor(masked_img).detach().clone(),
            (x1, y1, x2, y2),
            intrinsics
        )

        # apply transforms
        if self.transforms is not None:
            cropped_image = self.transforms(cropped_image)

        return cropped_image, updated_K, transform_matrix

# transform = T.Compose([
#     T.Resize(576), # whatever final resolution we want here
#     T.ToTensor(),
# ]) should be something like this^

class MVHumanNetDataDictWrapper(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __getitem__(self, i):
        img, K, transform_matrix = self.dset[i]
        return {"jpg": img, "K": K, "transform_matrix": transform_matrix}

    def __len__(self):
        return len(self.dset)


class MVHumanNetLoader(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        image_size: int = 576
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize(image_size), # whatever final resolution we want here
            T.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MVHumanNetDataDictWrapper(
                MVHumanNetDataset(
                    root_dir=os.path.join(self.root_dir),
                    transforms=self.transform
                )
            )
            # self.val_dataset = MVHumanNetDataDictWrapper(
            #     MVHumanNetDataset(
            #         root_dir=os.path.join(self.root_dir, "val"),
            #         transforms=self.transform
            #     )
            # )
        if stage == "test" or stage is None:
            self.test_dataset = MVHumanNetDataDictWrapper(
                MVHumanNetDataset(
                    root_dir=os.path.join(self.root_dir, "test"),
                    transforms=self.transform
                )
            )

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )