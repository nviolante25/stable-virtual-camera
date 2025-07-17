import os
import json
import pickle
import glob
from einops import rearrange, repeat 
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Union, Callable

from seva.geometry import get_plucker_coordinates
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
    create_transform_matrix,
    get_bbox_center_and_size,
    get_mvhumannet_extrinsics,
    load_json,
    load_pickle,
    update_intrinsics_resize,
    generate_gaussian_mixture_samples,
    generate_gaussian_samples,
    normalize_intrinsics
)
import time
from seva.data.cropper import RandomBBoxCropper

# NOTE: hardcoded camera order for each camera elevation (counter clockwise)
# use for trajectory NVS training!
# Camera IDs organized by rung elevation
TOP_RUNG = [
    'CC32871A043', 'CC32871A018', 'CC32871A012', 'CC32871A021',
    'CC32871A060', 'CC32871A006', 'CC32871A042', 'CC32871A041', 
    'CC32871A049', 'CC32871A036', 'CC32871A047', 'CC32871A019',
    'CC32871A020', 'CC32871A056', 'CC32871A009', 'CC32871A014'
]

MIDDLE_RUNG = [
    'CC32871A005', 'CC32871A033', 'CC32871A050', 'CC32871A059',
    'CC32871A017', 'CC32871A034', 'CC32871A032', 'CC32871A052',
    'CC32871A039', 'CC32871A058', 'CC32871A013', 'CC32871A004',
    'CC32871A044', 'CC32871A031', 'CC32871A055', 'CC32871A029'
]

BOTTOM_RUNG = [
    'CC32871A035', 'CC32871A016', 'CC32871A030', 'CC32871A038',
    'CC32871A023', 'CC32871A027', 'CC32871A051', 'CC32871A015',
    'CC32871A022', 'CC32871A057', 'CC32871A048', 'CC32871A008',
    'CC32871A046', 'CC32871A010', 'CC32871A040', 'CC32871A037'
]

CAMERA_RUNGS = [TOP_RUNG, MIDDLE_RUNG, BOTTOM_RUNG]
ALL_CAMERAS = sorted([cam for rung in CAMERA_RUNGS for cam in rung])
CAMERA_TO_INDEX = {cam: idx for idx, cam in enumerate(ALL_CAMERAS)}

# borrowed from @dataset.py
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

class MVHumanNetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_images,
        latents_dir=None,
        transforms=None,
        pre_scale=0.5,
        data_limit=None,
        only_include=None,
        random_crop=True,
        white_background=False,
    ):
        self.root_dir = root_dir             # directory of all subject directories
        self.latents_dir = latents_dir       # directory of all latents
        self.num_images = num_images         # context window T
        self.transforms = transforms         # transforms for the random crop
        self.pre_scale = pre_scale           # since MVHumanNet is downsampled, update intrinsics
        self.only_include = only_include     # TEMP -- include only these subjects (as List of strings)
        self.data_limit = data_limit         # TEMP -- only get the first 'data_limit' (int) subjects
        self.random_crop = random_crop       # NOTE: this is the toggle for probabilistic cropping 
                                             # ! unrelated to initial crop from crop_params.json
                                             # ! (human-centered 576x576 image crop) 
        self.adjacent_frame_sampling_prob = 0.2 # Trajectory NVS acceptance rate
        self.white_background = white_background
        self._autoencoder = None
        if num_images >= 16: # LIMIT to 16 images
            self.num_images = 16

        # actual data
        self.cam_params = {} # Dict[subject: (extrinsics, intrinsics, camera_scale)]
        self.scenes = self._load_scenes()
        self.image_shape = (1500, 2048) # MVHumanNet images are 2048x1500

        # from SD 2.1 VAE
        self.downsample_factor = 8
        self.scale_factor = 0.18215              
        self.target_shape = (576, 576)
        self.latent_shape = (self.num_images, 4, self.target_shape[0] // self.downsample_factor, 
                          self.target_shape[1] // self.downsample_factor)

        if self.transforms is None:
            # default (no probabilistic crop), only CenterCrop
            self.transform = T.Compose([
                T.CenterCrop(self.image_shape[0]), # Center crop to square
                T.Resize(self.target_shape),       # Resize to target shape
                T.ToTensor(),                      # Convert to tensor
                T.Normalize([0.5], [0.5])          # Normalize to [-1, 1]
            ])

        if self.random_crop:
            self.cropper = RandomBBoxCropper()
            self.transform = T.Compose([
                T.Resize(self.target_shape),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ])

        if self.pre_scale != 0.5:
            print("WARNING: pre_scale is not 0.5, which is expected for MVHumanNet!")


    def _clean_camera_keys(self, data):
        # Create new dictionary with cleaned keys
        cleaned_data = {}
        for key, value in data.items():
            # Extract just the camera ID number
            camera_id = key[2:-4] # remove "1_" and ".png" from camera_extrinsics.json
            cleaned_data[camera_id] = value
        return cleaned_data

    def init_autoencoder(self, autoencoder):
        self._autoencoder = autoencoder


    def _load_scenes(self):
        """
        For each subject in MVHumanNet, load dict:
        - frames_info: list of dicts, each with keys:
            - image_path
            - mask_path
            - annots_path
        - subject_id
        (Implicitly also updates self.cam_params)
        """
        scenes = []
        for i, subject in enumerate(os.listdir(self.root_dir)):
            if self.data_limit is not None and i >= self.data_limit:
                break
            subject_path = os.path.join(self.root_dir, subject)  
            if not os.path.isdir(subject_path): # ignore non-directories
                continue
            if self.only_include is not None and subject not in self.only_include:
                continue  # include the given subjects only

            # get subject metadata
            # NOTE: for MVHumanNet, all cameras have the same intrinsics
            # if different dataset, then may need to generalize this!
            extrinsics_path = os.path.join(subject_path, 'camera_extrinsics.json')
            intrinsics_path = os.path.join(subject_path, 'camera_intrinsics.json')
            extrinsics = self._clean_camera_keys(load_json(extrinsics_path))
            intrinsics = load_json(intrinsics_path)['intrinsics'] # same for all cameras
            camera_scale = load_pickle(os.path.join(subject_path, 'camera_scale.pkl'))

            # for each subject, store camera parameters separately
            # ! NOTE: intrinsics need to be downscaled by 2 later!
            # ! AND extrinsics [t] needs to be scaled by camera_scale later!
            self.cam_params[subject] = {
                'extrinsics': extrinsics, # Dict[camera_id: extrinsic params]
                'intrinsics': intrinsics, # List[List] (turn to matrix)
                'camera_scale': camera_scale # float
            }

            # annots, images, masks share the same camera directory names
            annots_path = os.path.join(subject_path, 'annots')
            images_path = os.path.join(subject_path, 'images_lr')
            masks_path = os.path.join(subject_path, 'fmask_lr')

            # NOTE: assumes the same cameras exist for each subject
            camera_dirs = [d for d in os.listdir(masks_path) if os.path.isdir(os.path.join(masks_path, d))]

            # Get first directory (to get number of timesteps)
            first_dir = camera_dirs[0]
            first_dir_path = os.path.join(masks_path, first_dir)
            timesteps = len([f for f in os.listdir(first_dir_path)])

            for i in range(1, timesteps + 1):
                # for each timestep, record all camera views (48)
                timestep = f"{i * 5:04d}"
                frames = {}
                for camera in camera_dirs:
                    frame = {
                        camera : {
                            'image_path': os.path.join(images_path, camera, f"{timestep}_img.jpg"),
                            'mask_path': os.path.join(masks_path, camera, f"{timestep}_img_fmask.png"),
                            'annots_path': os.path.join(annots_path, camera, f"{timestep}_img.json")
                        }
                    }
                    frames.update(frame)

                scenes.append({
                    'subject_id': subject, # string ID
                    'frames_info': frames,  # dict of {camera: image data}
                    'timestep': timestep
                })
                
        return scenes

    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        subject_id = scene['subject_id'] # ex. 100001
        timestep = scene['timestep'] # ex. 0005
        frames_info = dict(sorted(scene['frames_info'].items())) # camera dict
        subject_path = os.path.join(self.root_dir, subject_id) 

        # get camera parameters
        extrinsics = self.cam_params[subject_id]['extrinsics']
        intrinsics = np.array(self.cam_params[subject_id]['intrinsics'])
        camera_scale = self.cam_params[subject_id]['camera_scale'] 

        if self.pre_scale != 1: # update intrinsics (for MVHumanNet default 0.5x prescaling)
            intrinsics = update_intrinsics_resize(intrinsics, scale=self.pre_scale)

        # Sample frames indices
        camera_order = [cam for cam in list(frames_info.keys())] # 48 sorted camera IDs
        sampled_image_paths = [frames_info[cam]['image_path'] for cam in camera_order]
        sampled_image_mask_paths = [frames_info[cam]['mask_path'] for cam in camera_order]

        # NOTE: if num_images>16, then trajectory NVS will default to using all in rung
        if np.random.rand() <= self.adjacent_frame_sampling_prob: # for trajectory NVS
            # print("trajectory NVS")
            # choose which rung of cameras to sample from (top/mid/bot)
            # this is only because these paths are the most apparently continuous
            which_rung = np.random.randint(0, len(CAMERA_RUNGS))
            rung_of_cameras = CAMERA_RUNGS[which_rung]
            start_idx = np.random.randint(0, len(rung_of_cameras)) # out of 16 cameras
            images_permutation = np.roll(np.arange(len(rung_of_cameras)), -start_idx)[:self.num_images]
            images_permutation = [CAMERA_TO_INDEX[rung_of_cameras[i]] for i in images_permutation]
        else: # for set NVS
            # sample random indices
            # print("set NVS")
            images_permutation = np.random.choice(len(sampled_image_paths), self.num_images, replace=False)

        camera_order = [camera_order[i] for i in images_permutation] # ordered subset of 'num_images' sampled cameras
        sampled_image_paths = [sampled_image_paths[i] for i in images_permutation]
        sampled_image_mask_paths = [sampled_image_mask_paths[i] for i in images_permutation]

        
        # Load frames from image paths
        # (T,3,H,W)
        frames = torch.zeros((self.num_images, 3, self.target_shape[0],  self.target_shape[1]))
        for i, (img_path, mask_path) in enumerate(zip(sampled_image_paths, sampled_image_mask_paths)):
            image = Image.open(img_path).convert("RGB")
            img_mask = Image.open(mask_path)

            # Create masked image by compositing with black background
            if self.white_background:
                background = Image.new('RGB', image.size, (255, 255, 255))
            else: # black
                background = Image.new('RGB', image.size, (0, 0, 0))

            masked_image = Image.composite(image, background, img_mask)
            # Apply transforms after masking
            # NOTE: if using non-cropped latents, then transforms is just the default as in @dataset.py
            # masked_image = self.transform(masked_image) # ! moved transform to after random crop
            frames[i] = masked_image

        # Sample input/target frame split
        num_input_frames = np.random.randint(1, self.num_images) # at least 1 input frame
        input_frames_indices = np.random.choice(self.num_images, num_input_frames, replace=False) 

        # Create input/target masks (1: input/ 0: target)
        input_frames_mask = torch.zeros(self.num_images, dtype=torch.bool)
        input_frames_mask[input_frames_indices] = True

        camera_mask = torch.ones(self.num_images, dtype=torch.bool)

        def get_c2w(cam):
            tf_matrix = create_transform_matrix(
                np.array(extrinsics[cam]['rotation']),
                np.array(extrinsics[cam]['translation']) * camera_scale,
                homogeneous=True
            )

            return np.linalg.inv(tf_matrix) # w2c -> c2w

        # Read extrinsics (w2c -> c2w)
        all_c2ws = np.array([
            get_c2w(cam) for cam in frames_info.keys() # these keys are SORTED
        ])

        all_c2ws = torch.from_numpy(all_c2ws).float() # (total_cameras=48, 4, 4)
        c2ws = all_c2ws[images_permutation]    # choose previously sampled ones only (NOTE: order is unknown right now, need to edit!)
        center_cameras(all_c2ws, c2ws)  # mean center
        scale_cameras(c2ws)

        # create intrinsics tensor, update intrinsics
        # TODO: accept multiple camera intrinsics (for random cropping)
        # NOTE: for preliminary fine-tune testing, we currently account for the uniform center crop

        if self.random_crop:
            # get random crop parameters for the subject
            crop_params = np.load(os.path.join(subject_path, "crop_params.npz"))
            crop_params = [crop_params[f"{cam}.{timestep}"] for cam in camera_order]
            crop_params = torch.stack([torch.from_numpy(bbox) for bbox in crop_params])
            cropped_imgs, Ks = self.cropper(frames, crop_params, torch.from_numpy(intrinsics).float())
            Ks = normalize_intrinsics(Ks, self.image_shape[0], self.image_shape[1]) # normalize intrinsics (H,W)
            if Ks.dim() == 2: # if one shared intrinsic matrix, then repeat it for all
                Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images)
            frames = cropped_imgs

        else: # simply CenterCrop
            crop_amount  = (self.image_shape[1] - self.target_shape[0]) // 2 # (W - H) // 2: assumes W > H
            scale_amount = (self.target_shape[0] / self.image_shape[0])
            Ks = update_intrinsics(np.array(intrinsics), crop_x=crop_amount, crop_y=0, scale=scale_amount)
            Ks = normalize_intrinsics(Ks, self.image_shape[0], self.image_shape[1]) # normalize intrinsics (H,W)
            Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images) # assumes all intrinsics are the same for now 
            Ks = torch.from_numpy(Ks).float()

        # NOTE: the behavior of transform will change depending on whether random crop is used
        frames = self.transform(frames) # 576x576 image tensors

        # load latents if we provided a path
        if self.latents_dir is not None:
            latents_dir = subject_path.replace(self.root_dir, self.latents_dir)
            npz_file = os.path.join(latents_dir, f"{subject_id}.npz")
            npz_data = np.load(npz_file) # this is already for the current subject

            latent_tensors = [npz_data[f"{sample_cam}.{timestep}"] for sample_cam in camera_order]
            clean_latents = torch.stack([torch.from_numpy(latent_tensor) for latent_tensor in latent_tensors])
            # (batch_size, 4, 72, 72)
        else: # encode frames on the fly
            if self._autoencoder is not None:
                clean_latents = torch.zeros((self.num_images, 4, self.target_shape[0], self.target_shape[1]), device="cuda")
                # this automatically applies the scale factor if using seva AE
                # ! - HACK: reset the scale factor applied within the seva AE
                with torch.no_grad():
                    frames = frames.to("cuda")
                    clean_latents = (self._autoencoder.encode(frames, chunk_size=1) / self.scale_factor).to("cpu")
                    frames = frames.to("cpu") # back to CPU
            else:
                raise ValueError("Need to call _init_autoencoder() to encode latents on the fly. Otherwise, precomputed latents are required.")


        w2cs = torch.linalg.inv(c2ws)
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[input_frames_indices[0]],
            extrinsics=w2cs,
            intrinsics=Ks.clone(),
            target_size=(self.target_shape[0] // self.downsample_factor, 
                         self.target_shape[1] // self.downsample_factor),
        )

        # print("pluckers.shape: ", pluckers.shape)

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

        # print("concat.shape: ", concat.shape)

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

        # bbox params useful for cropping
        # H, W = int(annots['height'] * self.pre_scale), int(annots['width'] * self.pre_scale) # images not yet scaled down
        # bbox = torch.tensor(annots['annots'][0]['bbox'][:4])  # [x1, y1, x2, y2] (already scaled)
        # (center_x, center_y), (size_x, size_y) = get_bbox_center_and_size(bbox)
        # x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # # distributions to sample
        # # TODO: do this ONCE in init instead
        # # additionally, add the option to not crop
        # def center_sampler(batch_size):
        #     # mean at center of bbox
        #     mean = torch.tensor([x1 + size_x // 2, y1 + size_y // 2], dtype=torch.float32)
        #     cov = torch.tensor([[size_x, 0], [0, size_y]], dtype=torch.float32)
        #     weights = torch.tensor([0.7, 0.3])
        #     # return generate_gaussian_samples(mean, cov, batch_size)
        #     return generate_gaussian_mixture_samples([mean, (x1 + size_x // 2, y1)], [cov, cov], weights, batch_size)

        # def length_sampler(batch_size):
        #     # Example: Sample from a 1D Gaussian for crop size
        #     mean = torch.tensor([(size_x + size_y) // 1.5], dtype=torch.float32)  # mean is the smallest dim
        #     cov = torch.tensor([[max(size_x, size_y)]], dtype=torch.float32)
        #     return generate_gaussian_samples(mean, cov, batch_size)

        # # random crop the image
        # random_cropper = RandomBBoxCrop(center_sampler, length_sampler)
        # cropped_image, updated_K = random_cropper(
        #     T.functional.to_tensor(masked_img).detach().clone(),
        #     (x1, y1, x2, y2),
        #     intrinsics
        # )

        try:
            output_dict = {
                "clean_latent": clean_latents,
                "mask": input_frames_mask,
                "plucker": pluckers,
                "camera_mask": camera_mask,
                "concat": concat,
                "frames": frames,
                "replace": replace,
                "c2w": c2ws,
                "K": Ks,
            }
        except Exception as e:
            print(f"Error creating output_dict: {e}")
            raise

        return output_dict


class MVHumanNetLoader(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        latents_dir: str,
        num_images: int,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        image_size: int = 576,
        data_limit: int = None,
        only_include: list = None,
    ):
        super().__init__()
        
        self.root_dir = root_dir
        self.latents_dir = latents_dir
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_limit = data_limit
        self.only_include = only_include
        # Define transforms
        # self.transform = T.Compose([
        #     T.Resize(image_size), # whatever final resolution we want here
        #     T.ToTensor(),
        # ])
        self.transform = None # let corresponding Dataset handle this

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir),
                latents_dir=os.path.join(self.latents_dir),
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.only_include
            )

            # self.val_dataset = MVHumanNetDataDictWrapper(
            #     MVHumanNetDataset(
            #         root_dir=os.path.join(self.root_dir, "val"),
            #         transforms=self.transform
            #     )
            # )
        if stage == "test" or stage is None:
            self.test_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir, "test"),
                latents_dir=os.path.join(self.latents_dir),
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.only_include
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

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=True
    #     )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
