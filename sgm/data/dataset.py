from typing import Optional
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from torch.utils.data import Dataset
import os
import json
from PIL import Image
import torch
import cv2 as cv
import numpy as np


from seva.geometry import get_plucker_coordinates
from seva.modules.autoencoder import AutoEncoder
from sgm.data.read_write_model import read_model
from sgm.data.utils_camera import (
    read_intrinsics_colmap,
    read_extrinsics_colmap,
    read_intrinsics_nerfstudio,
    read_extrinsics_nerfstudio,
    opencv_to_opengl,
    colmap_to_nerfstudio,
    nerfstudio_to_colmap
)
from einops import rearrange, repeat 
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms

# used in "generative-models" repo
# however, instructions to retrieve stable-datasets doesn't seem to work
# try: # TODO: fix torchdata version
#     from sdata import create_dataset, create_dummy_dataset, create_loader
# except ImportError as e:
#     print("#" * 100)
#     print("Datasets not yet available")
#     print("to enable, we need to add stable-datasets as a submodule")
#     print("please use ``git submodule update --init --recursive``")
#     print("and do ``pip install -e stable-datasets/`` from the root of this repo")
#     print("#" * 100)
#     exit(1)

# _worker_ae = None

class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self):
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_datapipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_datapipeline, **self.test_config.loader)


def center_cameras(all_c2ws, c2ws):
    # finds mean position of all_c2ws, then centers cameras by subtracting the mean
    ref_c2ws = all_c2ws
    camera_dist_2med = torch.norm(
        ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    c2ws[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
    

def scale_cameras(c2ws, camera_scale=2.0):
    camera_dists = c2ws[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    c2ws[:, :3, 3] *= translation_scaling_factor


# def worker_init_fn(worker_id):
#     global _worker_ae
#     if _worker_ae is None:
#         _worker_ae = AutoEncoder().eval()
#         # _worker_ae = AutoEncoder().eval().cuda()

class DL3DVDataset(Dataset):
    def __init__(self, dataset_dir, colmap_dir, num_images, latents_dir=None, transform=None, levels=None):
        self.dataset_dir = dataset_dir
        self.colmap_dir = colmap_dir
        self.latents_dir = latents_dir
        self.transform = transform
        self.levels = levels if levels else []
        self.num_images = num_images
        self.scenes = self._load_scenes()
        self.adjacent_frame_sampling_prob = 0.2

        print("=" * 30)
        print("\n \n")
        print("\n dataset_dir:\n", dataset_dir)
        print("\n colmap_dir:\n", colmap_dir)
        print("\n levels:\n", levels)
        print("\n num_images:\n", num_images)
        print("\n \n")
        print("=" * 30)
        
        if "480P" in dataset_dir:
            self.image_shape = (270, 480)
            self.images_folder = "images_8"
        elif "960P" in dataset_dir:
            self.image_shape = (540, 960)
            self.images_folder = "images_4"

        self.target_shape = (576, 576)
        self.transform = transforms.Compose([
            transforms.CenterCrop(self.image_shape[0]), # Center crop to square
            transforms.Resize(self.target_shape),       # Resize to target shape
            transforms.ToTensor(),                      # Convert to tensor
            transforms.Normalize([0.5], [0.5])          # Normalize to [-1, 1]
        ])

        # Values fro SD 2.1 autoencoder
        self.donwsample_factor = 8
        self.scale_factor = 0.18215              
        self.latent_shape = (self.num_images, 4, self.target_shape[0] // self.donwsample_factor, 
                          self.target_shape[1] // self.donwsample_factor)


    def _load_scenes(self):
        scenes = []
        for level in os.listdir(self.dataset_dir):
            if self.levels and level not in self.levels:
                continue
            level_path = os.path.join(self.dataset_dir, level)
            if os.path.isdir(level_path):
                for scene in os.listdir(level_path):
                    scene_path = os.path.join(level_path, scene)
                    if os.path.isdir(scene_path):
                        scenes.append(scene_path)
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_path = self.scenes[idx]

        # Load colmap data
        colmap_scene_path = os.path.join(
            self.colmap_dir, os.path.relpath(scene_path, self.dataset_dir), "colmap", "sparse", "0"
        )
        cameras_metas, images_metas, _ = read_model(colmap_scene_path)
        # below are defined in 'read_write_model.py'
        # camera_metas is a dict {1: intrinsics} based on the camera used to take them
        # ex. Camera(id=1, model="OPENCV", w, h, params=8 element array)
        # images_metas is a dict {1...num_images, sequential but varying, & values are of:
        # Image(id=1, qvec=np.array, tvec=np.array, camera_id=1, name, xys, point3D_ids=array([], dtype=int32))

        images_metas = list(sorted(images_metas.values(), key=lambda x: x.name))

        # Load images files
        images_dir = os.path.join(scene_path, self.images_folder)
        images_files = [os.path.join(images_dir, image.name) for image in images_metas]
        # ex. images_8/frame_00001.png
        # should be able to use transforms.json instead of using this

        # Sample frames indices
        if np.random.rand() <= self.adjacent_frame_sampling_prob: # for trajectory NVS
            start_idx = np.random.randint(0, len(images_files) - self.num_images)
            images_idxs = np.arange(start_idx, start_idx + self.num_images)
        else: # for set NVS
            images_idxs = np.random.choice(len(images_files), self.num_images, replace=False)

        images_files = [images_files[i] for i in images_idxs]

        clean_latents = None
        if self.latents_dir is not None:
            # Load latents
            latents_dir = images_dir.replace(self.dataset_dir, self.latents_dir)
            latents_files = sorted([f for f in os.listdir(latents_dir) if f.endswith(".pt")])
            latents_files = [latents_files[i] for i in images_idxs]
            clean_latents = torch.stack([torch.load(os.path.join(latents_dir, latents_files[i])) for i in range(len(latents_files))])

        # print("clean_latents(shape): ", clean_latents.shape)

        # Load frames from image paths
        # (T,3,H,W)
        frames = torch.zeros((self.num_images, 3, self.target_shape[0],  self.target_shape[1]))
        for i, img_file in enumerate(images_files):
            img_path = os.path.join(images_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image) # images converted to square, [-1, 1] normalization
            frames[i] = image

        # Read extrinsics from COLMAP
        # takes in tvec and qvec to return homogeneous c2w matrix
        all_c2ws = np.array([read_extrinsics_colmap(image_meta, mode="c2w") for image_meta in images_metas])
        all_c2ws = torch.from_numpy(all_c2ws).float()
        c2ws = all_c2ws[images_idxs] # choose sampled ones only
        center_cameras(all_c2ws, c2ws) # mean center
        scale_cameras(c2ws)
        
        # Read intrinsics from COLMAP
        # key of camera: 1
        intrinsics = read_intrinsics_colmap(cameras_metas[1], normalize=True)
        # CURRENTLY, assumes all images have same intrinsics/taken with same camera
        # TODO: accept multiple camera intrinsics
        Ks = repeat(intrinsics, 'd1 d2 -> n d1 d2', n=self.num_images)
        Ks = torch.from_numpy(Ks).float()

        # Sample input and target frames
        num_input_frames = np.random.randint(1, self.num_images)  # Randomly select number of input frames
        input_frames_indices = np.random.choice(self.num_images, num_input_frames, replace=False) # Randomly select the input frames
        # num_input_frames: choose between 1 and max image frames
        # input_frames_indices: indices to make input frames

        # Create masks (for above 1: inputs/ 0: targets)
        input_frames_mask = torch.zeros(self.num_images, dtype=torch.bool)
        input_frames_mask[input_frames_indices] = True

        camera_mask = torch.ones(self.num_images, dtype=torch.bool)

        w2cs = torch.linalg.inv(c2ws)
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[input_frames_indices[0]],
            extrinsics=w2cs,
            intrinsics=Ks.clone(),
            target_size=(self.target_shape[0] // self.donwsample_factor, 
                         self.target_shape[1] // self.donwsample_factor),
        )

        concat = torch.cat( # binary mask and plcukers
            [
                repeat(
                    input_frames_mask,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
            ],
            dim=1,
        ) # (T, 6 + 1, 72, 72), where 6 is for plucker coords and 1 for binary mask


        latent_shape = (self.num_images, 4, self.target_shape[0] // self.donwsample_factor, 
                        self.target_shape[1] // self.donwsample_factor)
        clean_latents = torch.zeros(latent_shape)

        # NOTE: instead of using on-the-fly encoding,
        # use precomptued latents from latent_dir
        # with torch.no_grad():
        #     input_frames = frames[input_frames_indices]
        #     temp = self.ae.encode(input_frames_to_encode)  # AutoEncoder will handle device placement
        #     clean_latents[input_frames_indices] = temp

        replace = torch.cat( # clean latents and binary mask
            [
                clean_latents * self.scale_factor,
                repeat(
                    input_frames_mask,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
            ],
            dim=1,
        )

        try:
            output_dict = {
                "clean_latent": clean_latents,
                "mask": input_frames_mask,
                "plucker": pluckers,
                "camera_mask": camera_mask,
                "concat": concat,
                "frames": frames,
                "replace": replace,
            }
        except Exception as e:
            print(f"Error creating output_dict: {e}")
            raise

        # print("Before printing shapes", flush=True)
        # for key, value in output_dict.items():
        #     print(f"Printing shape for {key}", flush=True)
        #     print(f"{key}: {value.shape}", flush=True)
        # print("After printing shapes", flush=True)

        return output_dict


class DL3DVDataModuleFromConfig(LightningDataModule):
    def __init__(
            self,
            dataset_dir,
            colmap_dir, 
            batch_size, 
            latents_dir=None,
            num_workers=0,
            num_images=21,
            shuffle=True):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.colmap_dir = colmap_dir
        self.latents_dir = latents_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = DL3DVDataset(
            dataset_dir,
            colmap_dir,
            latents_dir=latents_dir,
            num_images=num_images,
            levels=["1K"],
        )

    def prepare_data(self):
        pass

    # def _collate_fn(self, batch):
    #     # batch dict from __getitem__
    #     # ['clean_latent', 'mask', 'plucker', 'camera_mask', 'concat', 'frames', 'replace']
    #     # collate first
    #     print("\nin collate!\n")
    #     collated = {}
    #     for key in batch[0].keys():
    #         if isinstance(batch[0][key], torch.Tensor):
    #             collated[key] = torch.stack([b[key] for b in batch])
    #         else:
    #             collated[key] = [b[key] for b in batch]
    #     print("collated.keys(): ", collated.keys())
    #     for key, value in collated.items():
    #         if isinstance(value, torch.Tensor):
    #             print(f"collated[{key}].shape: {value.shape}")
    #         else:
    #             print(f"collated[{key}] is not a tensor, length: {len(value)}")

    #     # If we need to compute latents
    #     # global _worker_ae
    #     # if _worker_ae is not None and 'frames' in collated:
    #     #     with torch.no_grad():
    #     #         # Get all input frames that need encoding
    #     #         input_frames = collated['frames'][collated['mask']]
                
    #     #         # move to GPU for compute
    #     #         # input_frames = input_frames.to(torch.device('cuda'))
    #     #         encoded = _worker_ae.encode(input_frames)
    #     #         # encoded = encoded.cpu()

    #     #         # Place encoded latents back in the correct positions
    #     #         collated['clean_latent'][collated['mask']] = encoded

    #     print("\n collate done!\n")

    #     return collated

    def train_dataloader(self):
        print("\nin train_dataloader!\n")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            # multiprocessing_context='spawn',
            # persistent_workers=True,
            # worker_init_fn=self.worker_init_fn,
            # collate_fn=self._collate_fn
        )


