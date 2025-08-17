import os
import json
import pickle
import glob
from collections import defaultdict
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
from seva.modules.autoencoder import AutoEncoder

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
        random_crop=False,
        white_background=False,
        step_size=20,
        preload_path=None,
        synthetic_dataset_path=None,
    ):
        self.root_dir = root_dir             # directory of all subject directories
        self.latents_dir = latents_dir       # directory of all latents
        self.num_images = num_images         # context window T
        self.transforms = transforms         # transforms for the random crop
        self.pre_scale = pre_scale           # since MVHumanNet is downsampled, update intrinsics
        self.only_include = only_include     # TEMP -- include only these subjects (as List of strings)
        self.data_limit = data_limit         # TEMP -- only get the first 'data_limit' (int) subjects
        self.step_size = step_size           # only processes every 'step_size' frames (timesteps)
        self.random_crop = random_crop       # NOTE: this is the toggle for probabilistic cropping 
                                             # ! unrelated to initial crop from crop_params.json
                                             # ! (human-centered 576x576 image crop) 
        self.adjacent_frame_sampling_prob = 0.2 # Trajectory NVS acceptance rate
        self.white_background = white_background
        self.preload_path = preload_path
        self.synthetic_dataset_path = synthetic_dataset_path # IC-light/InfU output directory
        # if not None, will use the "phase 2" expected training process

        if self.num_images > 16: # if more than 16, disable trajectory NVS batching
            self.adjacent_frame_sampling_prob = 0.0

        # actual data
        self.cam_params = {} # Dict[subject: (extrinsics, intrinsics, camera_scale)]
        self.scenes = self._load_scenes() if preload_path is None else self._load_preloaded_filepaths()
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
        print("MVHN::init done!")


    def _clean_camera_keys(self, data):
        # Create new dictionary with cleaned keys
        cleaned_data = {}
        for key, value in data.items():
            # Extract just the camera ID number
            camera_id = key[2:-4] # remove "1_" and ".png" from camera_extrinsics.json
            cleaned_data[camera_id] = value
        return cleaned_data

    def _load_preloaded_filepaths(self):
        """
        Load preloaded filepaths from a custom json file structured as:
        WARNING: relies on accurate priors that all subjects have the exact same structure!
        {
            "subject_id": {
                "timesteps": <int: number of timesteps within first camera> (or list)
                "cameras": <list: all camera IDs> (validated prior)
                "extrinsics": <dict: camera extrinsics>
                "intrinsics": <list: camera intrinsics>
                "camera_scale": <float: camera scale>
                "annots": {"bbox": <list: bbox coords.>, "bbox_face": <list: face bbox coords.>}
            }
        }
        The above structure gives all that we need to load the filepaths (reducing metadata reads by a lot!)
        """
        assert self.preload_path is not None, "Preload path must be provided!"
        print("Loading preloaded filepaths...")
        preload_path = self.preload_path
        subjects = load_json(preload_path) # preloaded subject data
        scenes = []

        subjects_with_latents = None
        if self.latents_dir is not None:
            subjects_with_latents = set([subject for subject in os.listdir(self.latents_dir) if os.path.exists(os.path.join(self.latents_dir, subject, f"{subject}.npz"))])
            print(f"Found {len(subjects_with_latents)} subjects with latents")

        # for each subject (key) in the preloaded data (should be of available latents and metadata):
        for i, subject in tqdm(enumerate(subjects), total=len(subjects), desc="Loading scenes"):
            if subject == "metadata":
                continue
            subject_path = os.path.join(self.root_dir, subject)
            if self.only_include is not None and subject not in self.only_include:
                continue
            if self.data_limit is not None and i >= self.data_limit:
                break
            if (subjects_with_latents is not None and subject not in subjects_with_latents) or len(subjects[subject]['cameras']) != 48:
                print(f"Skipping subject {subject} because it does not have all 48 cameras or latents precomputed!")
                continue # if no latents precomputed for this subject, or if not all cameras are present, then skip this

            extrinsics = subjects[subject]['extrinsics'] # should be pre-cleaned!
            intrinsics = subjects[subject]['intrinsics']['intrinsics']
            camera_scale = subjects[subject]['camera_scale']
            annots = subjects[subject]['annots']

            # for each subject, store camera parameters separately
            # ! NOTE: intrinsics need to be downscaled by 2 later!
            # ! AND extrinsics [t] needs to be scaled by camera_scale later!
            self.cam_params[subject] = {
                'extrinsics': extrinsics, # Dict[camera_id: extrinsic params]
                'intrinsics': intrinsics, # List[List] (turn to matrix)
                'camera_scale': camera_scale # float
            }

            # get image, mask, annots
            num_timesteps = subjects[subject]['timesteps'] # usually int, but this may be a LIST!
            # ensure only camera dirs were captured
            cameras = [cam for cam in subjects[subject]['cameras'] if cam in annots['bbox']]
            step_size = subjects["metadata"]["step_size"]
            subject_map = defaultdict(dict)

            # build frames_info for each timestep, camera combination
            # NOTE: num_timesteps is based on the FIRST camera; some subjects have differing numbers of timesteps for their cameras!
            iterator = range(1, num_timesteps, step_size) if isinstance(num_timesteps, int) else num_timesteps
            is_list_type = isinstance(num_timesteps, list)
            for timestep in iterator:
                try: # to get all cameras for this timestep (and ENSURE all cameras are present)
                    for camera in cameras:
                        time_id = timestep if is_list_type else f"{timestep * 5:04d}"
                        image_path = os.path.join(subject_path, "images_lr", camera, f"{time_id}_img.jpg")
                        mask_path = os.path.join(subject_path, "fmask_lr", camera, f"{time_id}_img_fmask.png")
                        # annots_path = os.path.join(subject_path, "annots", camera, f"{time_id}_img.json")

                        subject_map[time_id][camera] = {
                                    'image_path': image_path,
                                    'mask_path': mask_path,
                                    'annots': {
                                        'bbox': annots['bbox'][camera][time_id],
                                        'bbox_face': annots['bbox_face2d'][camera][time_id]
                                    }
                                }
                except Exception as e: # NOTE: this is a hack to ignore missing timesteps
                    print(f"Error loading subject {subject} camera {camera} timestep {timestep}: {e}")
                    subject_map.pop(time_id, None) # remove this timestep from the subject_map
                    break 

            sorted_timesteps = sorted(subject_map.keys())
            for i in range(0, len(sorted_timesteps), step_size):
                timestep = sorted_timesteps[i]
                frames_info = subject_map[timestep]

                if len(frames_info.keys()) < self.num_images: # not enough cameras for this timestep to sample
                    continue # then skip this timestep

                scenes.append({
                    'subject_id': subject,
                    'frames_info': frames_info,
                    'timestep': timestep
                })
        print("Loading preloaded filepaths completed!")
        return scenes


    def _load_scenes(self):
        """
        NEW -- compact loading using priors:
        - all subjects are continuous between timesteps (no gaps)
        - all subjects have the same cameras
        - all subjects have the same number of timesteps
        - 
        For each subject in MVHumanNet, load dict:
        - frames_info: list of dicts, each with keys:
            - image_path
            - mask_path
            - annots (bbox, bbox_face)
        (Implicitly also updates self.cam_params)
        """
        scenes = []
        valid_latent_scenes = None
        if self.latents_dir is not None:    
            valid_latent_scenes = [
                subject for subject in os.listdir(self.latents_dir)
                if subject in os.listdir(self.root_dir)
            ]
        else:
            valid_latent_scenes = os.listdir(self.root_dir)

        for i, subject in tqdm(enumerate(valid_latent_scenes), total=len(valid_latent_scenes), desc="Loading scenes"):
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
            camera_dirs = [d for d in os.listdir(masks_path)]
            subject_map = defaultdict(dict)

            for camera in camera_dirs:
                cam_path = os.path.join(images_path, camera)

                try:
                    for entry in os.scandir(cam_path):
                        if entry.is_file() and entry.name.endswith('_img.jpg'):
                            timestep = entry.name.split('_')[0]
                            annots_json = load_json(os.path.join(annots_path, camera, f"{timestep}_img.json"))['annots'][0]
                            bbox = annots_json['bbox']
                            bbox_face = annots_json['bbox_face2d']
                            subject_map[timestep][camera] = {
                                'image_path': entry.path,
                                'mask_path': os.path.join(masks_path, camera, f"{timestep}_img_fmask.png"),
                                'annots': {
                                    'bbox': bbox,
                                    'bbox_face': bbox_face
                                }
                            }
                except Exception as e:
                    print(f"Error loading subject {subject} camera {camera}: {e}")
                    continue
            
            sorted_timesteps = sorted(subject_map.keys())
            for i in range(0, len(sorted_timesteps), self.step_size):
                timestep = sorted_timesteps[i]
                frames_info = subject_map[timestep]

                if len(frames_info.keys()) < self.num_images: # not enough cameras for this timestep to sample
                    continue # then skip this timestep

                scenes.append({
                    'subject_id': subject,
                    'frames_info': frames_info,
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
        print(f"subject id: {subject_id}, timestep: {timestep}")

        # get camera parameters
        extrinsics = self.cam_params[subject_id]['extrinsics']
        intrinsics = np.array(self.cam_params[subject_id]['intrinsics'])
        camera_scale = self.cam_params[subject_id]['camera_scale'] 

        if self.pre_scale != 1: # update intrinsics (required for MVHumanNet default 0.5x prescaling)
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
        # (T,3,H',W'), this will be scaled later to 'target_shape'
        frames = torch.zeros((self.num_images, 3, self.image_shape[0],  self.image_shape[1]))
        for i, (img_path, mask_path) in enumerate(zip(sampled_image_paths, sampled_image_mask_paths)):
            image = Image.open(img_path).convert("RGB")
            img_mask = Image.open(mask_path)

            if img_mask.size != image.size: # ensure matching size! (note: 100681 has different sizes!)
                image = image.resize(img_mask.size, Image.BILINEAR)

            # Create masked image by compositing with black background
            if self.white_background:
                background = Image.new('RGB', image.size, (255, 255, 255))
            else: # black
                background = Image.new('RGB', image.size, (0, 0, 0))

            masked_image = Image.composite(image, background, img_mask)
            # Apply transforms after masking
            # NOTE: if using non-cropped latents, then transforms is just the default as in @dataset.py
            # masked_image = self.transform(masked_image) # ! moved transform to after random crop
            frames[i] = T.ToTensor()(masked_image)

        # Sample input/target frame split
        num_input_frames = np.random.randint(1, self.num_images) # at least 1 input frame
        input_frames_indices = np.random.choice(self.num_images, num_input_frames, replace=False) 

        # Create input/target masks (1: input/ 0: target)
        input_frames_mask = torch.zeros(self.num_images, dtype=torch.bool)
        input_frames_mask[input_frames_indices] = True

        # reference mask for SimVS
        ref_mask = torch.zeros(self.num_images, dtype=torch.bool)
        fix_frame_idx = input_frames_indices[np.random.choice(len(input_frames_indices), 1).item()]
        ref_mask[fix_frame_idx] = True # this becomes the fixed frame
        ic_paths = [path.replace("mv_captures", "relit_images").replace(".jpg", ".png") for path in sampled_image_paths]
        # ! NOTE: only works with IC-light; need to combine with InfU later.

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
        if self.random_crop:
            annots_jsons = [frames_info[cam]["annots"] for cam in camera_order]
            crop_params = []
            for annots_json in annots_jsons:
                bbox = annots_json['bbox'][:4]
                face_bbox = annots_json['bbox_face'][:4]
                crop_params.append((bbox, face_bbox))
            # account for mvhn downsampling (hence the 0.5)
            bbox_params = torch.stack([torch.tensor(bbox) * 0.5 for bbox, _ in crop_params])
            # face_params = torch.stack([torch.tensor(face_bbox) * 0.5 for _, face_bbox in crop_params])
            # apparently, face bbox data isn't really        
            # crop_config = {
            #     "center_mean": torch.mean(face_params.reshape(-1,2,2).permute(0,2,1), dim=2),
            # }
            frames, Ks = self.cropper(frames, bbox_params, torch.from_numpy(intrinsics).float())
            # later, we resize using transform, so we update cropped intrinsics here accordingly
            scale = np.array([self.target_shape[0] / cropped_img.shape[-2] for cropped_img in frames])
            Ks = update_intrinsics_resize(Ks, scale)
            Ks = normalize_intrinsics(Ks, self.target_shape[0], self.target_shape[1]) # normalize intrinsics (H,W)
            if len(Ks.shape) == 2: # if one shared intrinsic matrix, then repeat it for all
                Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images)
            Ks = torch.from_numpy(Ks).float()
        else: # simply CenterCrop
            # if CenterCrop, then crop original size image to square
            min_dim = min(*self.image_shape)
            max_dim = max(*self.image_shape)
            crop_amount  = (max_dim - min_dim) // 2 # (W-H)/2, cropped from each side (only left-part relevant)
            scale_amount = (self.target_shape[0] / min_dim)
            Ks = update_intrinsics(np.array(intrinsics), crop_x=crop_amount, crop_y=0, scale=scale_amount)
            Ks = normalize_intrinsics(Ks, self.target_shape[0], self.target_shape[1]) # normalize intrinsics (H,W)
            Ks = repeat(Ks, 'd1 d2 -> n d1 d2', n=self.num_images) # assumes all intrinsics are the same 
            Ks = torch.from_numpy(Ks).float()

        # NOTE: the behavior of transform will change depending on whether random crop is used
        # frames = [self.transform(frame) for frame in frames]
        frames = self.transform(frames)
        # frames = torch.stack(frames, dim=0) # resize to 576x576 normalized [-1, 1] image tensorss

        # load latents if we provided a path
        if self.latents_dir is not None and os.path.exists(os.path.join(self.latents_dir, subject_id, f"{subject_id}.npz")):
            npz_file = os.path.join(self.latents_dir, subject_id, f"{subject_id}.npz")
            # npz_data = np.load(npz_file) # this is already for the current subject
            with np.load(npz_file) as npz_data:
                latent_tensors = [npz_data[f"{sample_cam}.{timestep}"] for sample_cam in camera_order]
                clean_latents = torch.stack([torch.from_numpy(latent_tensor) for latent_tensor in latent_tensors]) # (B, 4, 72, 72)
        else: # encode frames on the fly (DO NOT DO THIS IN DATASET)
            clean_latents = 0  # just use 'frames' (already pre-masked and cropped) in SevaWrapper
            # clean_latents = torch.zeros((self.num_images, 4, self.target_shape[0], self.target_shape[1]))

        w2cs = torch.linalg.inv(c2ws)
        pluckers = get_plucker_coordinates(
            extrinsics_src=w2cs[input_frames_indices[0]],
            extrinsics=w2cs,
            intrinsics=Ks.clone(),
            target_size=(self.target_shape[0] // self.downsample_factor, 
                         self.target_shape[1] // self.downsample_factor),
        )

        concat = torch.cat( # binary masks (inp/tgt + ref) and pluckers
            [
                repeat(
                    input_frames_mask,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                repeat(
                    ref_mask, 
                    "n -> n 1 h w", 
                    h=pluckers.shape[2],
                    w=pluckers.shape[3] 
                ),
                pluckers,
            ],
            dim=1,
        ) # (T, 6 + 1, 72, 72), where 6 is for plucker coords and 1 for binary mask

        if type(clean_latents) == int and clean_latents == 0:
            replace = 0
        else:
            replace = torch.cat(
                [
                    clean_latents * self.scale_factor,
                    # repeat( -- old Seva
                    #     input_frames_mask,
                    #     "n -> n 1 h w",
                    #     h=pluckers.shape[2],
                    #     w=pluckers.shape[3],
                    # ),
                    repeat(
                    ref_mask, 
                    "n -> n 1 h w", 
                    h=pluckers.shape[2],
                    w=pluckers.shape[3] 
                    ),
                ],
                dim=1,
            )

        try:
            # ensure in shared_step:
            # - clean_latents gets encoded on the fly (if not found)
            # - update concat with ic latents
            # - replace gets updated
            output_dict = {
                "clean_latent": clean_latents, # unscaled clean latents
                "mask": input_frames_mask,
                "ref_mask": ref_mask, # "one hot" mask for reference images 
                "ic_paths": ic_paths, # synthetic data paths
                "plucker": pluckers,
                "camera_mask": camera_mask,
                "concat": concat,
                "frames": frames,
                "replace": replace, # contains pre-scaled clean latents!
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
        num_images: int,
        batch_size: int,
        latents_dir: str = None,
        num_workers: int = 0,
        shuffle: bool = True,
        image_size: int = 576,
        data_limit: int = None,
        only_include: list = None,
        step_size: int = 150,
        preload_path: str = None,
        synthetic_dataset_path: str = None,
    ):
        super().__init__()
        print("init of DATALOADER")
        self.root_dir = root_dir
        self.latents_dir = os.path.join(self.latents_dir) if latents_dir is not None else None
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_limit = data_limit
        self.only_include = only_include
        self.step_size = step_size
        self.preload_path = preload_path
        self.synthetic_dataset_path = synthetic_dataset_path
        # Define transforms
        # self.transform = T.Compose([
        #     T.Resize(image_size), # whatever final resolution we want here
        #     T.ToTensor(),
        # ])
        self.transform = None # let corresponding Dataset handle this

    def setup(self, stage: Optional[str] = None):
        print("setup of DATALOADER")
        print("stage: ", stage)
        if stage == "fit" or stage is None:
            print("train is reached")
            self.train_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir),
                latents_dir=self.latents_dir,
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.only_include,
                step_size=self.step_size,
                preload_path=self.preload_path,
                synthetic_dataset_path=self.synthetic_dataset_path
            )
            print("train_dataset loaded")

            # self.val_dataset = MVHumanNetDataDictWrapper(
            #     MVHumanNetDataset(
            #         root_dir=os.path.join(self.root_dir, "val"),
            #         transforms=self.transform
            #     )
            # )
        if stage == "test" or stage is None:
            self.test_dataset = MVHumanNetDataset(
                root_dir=os.path.join(self.root_dir, "test"),
                latents_dir=self.latents_dir,
                num_images=self.num_images,
                transforms=self.transform,
                data_limit=self.data_limit,
                only_include=self.only_include,
                step_size=self.step_size,
                preload_path=self.preload_path,
                synthetic_dataset_path=self.synthetic_dataset_path
            )
            

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        print("dataloader train_dataloader")
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