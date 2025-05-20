# hacky crop script
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from PIL import Image, ImageOps
import sys
import json
import cv2
import argparse
from tqdm import tqdm
from copy import deepcopy
from glob import glob
import pickle
import torch

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# def reverse_camera_directions(camera_poses):
#     """
#     Reverses the direction of camera poses by flipping the z-axis.
    
#     Args:
#         camera_poses (list): List of 4x4 camera pose matrices
        
#     Returns:
#         list: List of camera poses with reversed directions
#     """
#     reversed_poses = []
    
#     for pose in camera_poses:
#         # Create a copy of the pose to avoid modifying the original
#         reversed_pose = pose.copy()
#         # Flip the z-axis direction (third column of rotation matrix)
#         reversed_pose[:3, 2] = -reversed_pose[:3, 2]
#         reversed_poses.append(reversed_pose)
    
#     return reversed_poses

# Function to normalize camera poses to a smaller scale
def normalize_camera_poses(camera_poses, scale_factor=1.0):
    """
    Normalize camera poses to a smaller scale while preserving their relative positions.
    
    Args:
        camera_poses (list): List of camera pose matrices (4x4)
        scale_factor (float): Scale factor to apply after normalization (default: 1.0)
                             Smaller values make the visualization more compact
    
    Returns:
        list: Normalized camera pose matrices
    """
    # Extract camera positions
    positions = np.array([pose[:3, 3] for pose in camera_poses])
    # Calculate the centroid
    centroid = np.mean(positions, axis=0)
    # Calculate the maximum distance from the centroid
    max_distance = np.max(np.linalg.norm(positions - centroid, axis=1))
    
    # Create normalized poses
    normalized_poses = []
    for pose in camera_poses:
        new_pose = pose.copy()
        # Center the position
        new_pose[:3, 3] = new_pose[:3, 3] - centroid
        # Scale the position
        new_pose[:3, 3] = new_pose[:3, 3] / max_distance * scale_factor
        normalized_poses.append(new_pose)
    
    return normalized_poses
 
def generate_gaussian_samples(mean, cov, n_samples=1000, device=None, dtype=None):
    mean = torch.as_tensor(mean, device=device, dtype=dtype)
    cov = torch.as_tensor(cov, device=device, dtype=dtype)
    dim = mean.shape[0]
    z = torch.randn(n_samples, dim, device=mean.device, dtype=mean.dtype)
    L = torch.linalg.cholesky(cov)
    x = z @ L.T + mean
    return x

def generate_gaussian_mixture_samples(means, covs, weights=None, n_samples=1000):
    """
    Generate samples from a Gaussian mixture model.
    
    Args:
        means: List of mean vectors for each component
        covs: List of covariance matrices for each component
        weights: List of weights for each component (if None, equal weights are used)
        n_samples: Number of samples to generate
        
    Returns:
        samples: Generated samples (shape: (n_samples, D))
    """
    n_components = len(means)
    
    if weights is None: # equal weight
        weights = torch.ones(n_components) / n_components
    else:
        # Normalize weights to sum to 1
        weights = torch.tensor(weights) / torch.sum(torch.tensor(weights))
        weights = weights.detach().clone()
    
    # Convert weights to numpy for multinomial sampling
    weights_np = weights.cpu().numpy() if isinstance(weights, torch.Tensor) else weights
    component_samples = np.random.multinomial(n_samples, weights_np)
    
    # Generate samples from each component
    samples = []
    for i in range(n_components):
        if component_samples[i] > 0:
            component_data = generate_gaussian_samples(means[i], covs[i], component_samples[i])
            samples.append(component_data)
    
    # Combine and shuffle the samples
    combined_samples = torch.cat(samples, dim=0)
    
    # Shuffle the samples
    idx = torch.randperm(combined_samples.shape[0])
    combined_samples = combined_samples[idx]
    return combined_samples


def find_index_for_camera_id(camera_ids, transforms_file):
    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)
        frames = transforms_data['frames']
        indices = []
        for camera_id in camera_ids:
            for i, frame in enumerate(frames):
                if camera_id in frame['file_path']:
                    indices.append(i)
        return indices

def read_camera_ids_from_file(file_path, step=1):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()[::step]]

def match_camera_ids_to_index(cam_id_file, transforms_file, step=1):
    camera_ids = read_camera_ids_from_file(cam_id_file, step=step)
    indices = find_index_for_camera_id(camera_ids, transforms_file)
    return indices

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_transform_matrix(R, t, homogeneous=True):
    """Create a 3x4 (4x4 if homogeneous) transform matrix from translation and rotation."""
    transform = np.eye(4) if homogeneous else np.zeros((3, 4))
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform # .tolist() outside if store in json

def apply_mask(img, mask):
    """ Applies a binary mask to an image. The mask must have values [0,1]. """
    # binary mask (make sure it's reshaped to match)
    masked_img = img * mask.reshape(*mask.shape, 1)
    return masked_img.astype(np.uint8)

def get_camera_center(transformation_matrix):
    return -transformation_matrix[:3, :3].T @ transformation_matrix[:3, 3]

def get_multiview_sample(image_path, mask_path, timestep: int, from_cameras: None):
    """
    Following the MVHumanNet dataset structure, get multi-view images (from all cameras)
     of a subject as well as their binary masks.
     NOTE: files will be output as {cameraID_img}.jpg and {camera_ID_img_fmask}.png respectively.
    """
    camera_dirs = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
    
    # if specific cameras requested, filter to only those
    # otherwise use all cameras
    if from_cameras is not None:
        camera_dirs = [d for d in camera_dirs if d in from_cameras]
        
    images = {}
    masks = {}
    
    for camera in camera_dirs:
        # Construct paths for this timestep
        img_file = f"{str(timestep).zfill(4)}_img.jpg"
        mask_file = f"{str(timestep).zfill(4)}_img_fmask.png"
        
        img_path = os.path.join(image_path, camera, img_file)
        mask_path_full = os.path.join(mask_path, camera, mask_file)
        
        # Load if files exist
        if os.path.exists(img_path) and os.path.exists(mask_path_full):
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path_full))
            
            # Store in dictionaries
            images[camera] = img
            masks[camera] = mask
            
    return images, masks

# intrinsic update functions + cropping
def get_bbox_from_annot(annot):
    """Extract bbox from annotation and convert to [x1, y1, x2, y2] format."""
    bbox = annot['annots'][0]['bbox']  # Assuming single person
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

def get_bbox_center_and_size(bbox):
    """Get center point and size of bbox."""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return (center_x, center_y), (width, height)


def get_mvhumannet_extrinsics(extrinsics_dict, scale):
    """Get extrinsics for all cameras in a subject directory."""
    extrinsics = create_transform_matrix(
        extrinsics_dict['rotation'], extrinsics_dict['translation'],
        homogeneous=False
    )
    extrinsics[:3, 3] = extrinsics[:3, 3] * scale
    return extrinsics

def update_intrinsics(K, crop_x=0, crop_y=0, scale=1, crop_first=True, padding_mode=False):
    """Update intrinsic matrix for the crop and resizes."""
    K_new = K.copy() if type(K) == np.ndarray else K.clone()
    if crop_first:
        K_new = update_intrinsics_crop(K, crop_x, crop_y, padding_mode)
        if scale != 1:
            K_new = update_intrinsics_resize(K_new, scale)
    else:
        if scale != 1:
            K_new = update_intrinsics_resize(K, scale)
        K_new = update_intrinsics_crop(K_new, crop_x, crop_y, padding_mode)
    return K_new

def update_intrinsics_crop(K, crop_x, crop_y, padding_mode=False):
    """
    Update intrinsic matrix for the crop.
    If negative, then this is considered a padding (need to set padding_mode=True).
    """
    K_new = K.copy() if type(K) == np.ndarray else K.clone()
    # padding mode enables negative "crops" to act as padding
    if crop_x > 0 or padding_mode:
        K_new[0, 2] = K[0, 2] - crop_x
    if crop_y > 0 or padding_mode:
        K_new[1, 2] = K[1, 2] - crop_y

    # if padding_mode == False: # this would need to be positive
    #     assert K_new[0, 2] >= 0, f"crop_x is negative: {K_new[0, 2]}"
    #     assert K_new[1, 2] >= 0, f"crop_y is negative: {K_new[1, 2]}"
    
    return K_new

def update_intrinsics_resize(K, scale):
    """Update intrinsic matrix for the resize."""
    # Create new intrinsic matrix
    K_new = K.clone() if type(K) == torch.Tensor else K
    K_new[0, 0] = K[0, 0] * scale
    K_new[1, 1] = K[1, 1] * scale
    K_new[0, 2] = K[0, 2] * scale
    K_new[1, 2] = K[1, 2] * scale
    return K_new


def crop_image(bbox, image, mask=None):
    """
    Crop image based on bounding box.
    """
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    if mask is not None:
        cropped_image = apply_mask(cropped_image, mask[y1:y2, x1:x2])
    return cropped_image


def get_crop_coordinates(center_x, center_y, crop_size, original_width, original_height, image=None, mask=None):
    """
    Crop image centered on a point, handling edge cases where the crop would go out of bounds.
    NOTE: image is optional; meaning that this function only returns the crop bbox coordinates if not given.
    
    Args:
        center_x, center_y: Center point coordinates
        crop_size: Size of the square crop
        original_width, original_height: Original image dimensions
        image (optional): Input image array (gives actual cropped image)
        mask (optional): Corresponding mask array for the input image.
        
    Returns:
        Crop coordinates (x1, y1, x2, y2) and cropped image (if image is not None, otherwise None)
    """

    crop_size = int(crop_size)
    half_size_left = crop_size // 2
    half_size_right = crop_size - half_size_left  # This handles odd crop_size correctly
    
       # Calculate initial coordinates
    x1 = int(round(center_x - half_size_left))
    y1 = int(round(center_y - half_size_left))
    x2 = int(round(center_x + half_size_right))
    y2 = int(round(center_y + half_size_right))
    
    # Adjust if crop goes out of bounds
    if x1 < 0:
        x2 -= x1  # Shift right edge
        x1 = 0
    if y1 < 0:
        y2 -= y1  # Shift bottom edge
        y1 = 0
    if x2 > original_width:
        x1 -= (x2 - original_width)  # Shift left edge
        x2 = original_width
    if y2 > original_height:
        y1 -= (y2 - original_height)  # Shift top edge
        y2 = original_height
    
    # Final adjustment if we still have issues after shifting
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(original_width, x2)
    y2 = min(original_height, y2)
    bbox = (x1, y1, x2, y2)

    # Return crop coordinates and cropped image if provided
    if image is not None:
        cropped_image = crop_image(bbox, image, mask)
        return bbox, cropped_image
    else:
        return bbox, None


def process_image_crops_and_intrinsics(base_dir, image_id="0005", scale=0.5, homogenous_image_size=None):
    """
    Process all images and update intrinsics based on the smallest image dimension.
    'scale' is default 0.5 for MVHumanNet, where images are downsampled by 2x,
    while intrinsics are based on the original size.
    If `homogenous_image_size` is given (tuple of (width, height)), then assumes all images are of the same given resolution.
    Additionally, if homogenous_image_size is given, then the scale is ignored.
    """
    if homogenous_image_size is not None:
        assert len(homogenous_image_size) == 2

    # Load camera intrinsics
    intrinsics = load_json(os.path.join(base_dir, 'camera_intrinsics.json'))
    K = np.array(intrinsics['intrinsics'])
    K = update_intrinsics(K, scale=scale)
    
    # Get all camera directories in annots
    annot_dir = os.path.join(base_dir, 'annots')
    camera_dirs = [d for d in os.listdir(annot_dir) if os.path.isdir(os.path.join(annot_dir, d))]
    
    bbox_info = {}
    image_dimensions = {}
    
    # First pass: collect image dimensions and bbox centers
    for camera in camera_dirs:
        camera_annot_dir = os.path.join(annot_dir, camera)
        annot_path = os.path.join(camera_annot_dir, f"{image_id}_img.json")
        annot = load_json(annot_path)
        
        if homogenous_image_size is None:
            # Get image dimensions from annot file
            original_width = annot['width'] * scale
            original_height = annot['height'] * scale
            image_dimensions[camera] = (original_width, original_height)
        else:
            original_width, original_height = homogenous_image_size
            
        # Get bbox center for centering the crop
        bbox = get_bbox_from_annot(annot)
        (center_x, center_y), _ = get_bbox_center_and_size(bbox)
        center_x = center_x * scale
        center_y = center_y * scale
        
        bbox_info[camera] = {
            'center': (center_x, center_y),
            'original_width': original_width,
            'original_height': original_height
        }
    
    if homogenous_image_size is None:
        # Find the smallest dimension across all images
        min_dimension = float('inf')
        for camera in image_dimensions:
            width, height = image_dimensions[camera]
        min_dimension = min(min_dimension, min(width, height))
    else: # image size is already known
        min_dimension = min(*homogenous_image_size)
    
    # Use the smallest dimension as the crop size
    crop_size = int(min_dimension)
    results = {}
    
    # Second pass: crop images and update intrinsics
    for camera in camera_dirs:
        center_x, center_y = bbox_info[camera]['center']
        original_width = bbox_info[camera]['original_width']
        original_height = bbox_info[camera]['original_height']
        
        # Use get_crop_coordinates function to get crop coordinates
        (x1, y1, x2, y2), _ = get_crop_coordinates(
            center_x, 
            center_y, 
            crop_size, 
            original_width, 
            original_height, 
            image=None,
            mask=None
        )

        # Update intrinsics K post-crop
        print("OLD")
        print(K)
        K_new = update_intrinsics(K, x1, y1)
        print("NEW")
        print(K_new)
        
        results[camera] = {
            'crop': (x1, y1, x2, y2),
            'K': K_new
        }
    
    return results


def crop_and_save_image(
    base_dir,
    camera_id,
    timestep,
    output_dir,
    images_dir='images_lr',
    mask_dir='fmask_lr',
    homogenous_image_size=None,
    scale=0.5
):
    """
    Crop and save images for a given camera ID and timestep.
    Save to `output_dir`.
    
    Args:
        base_dir (str): Base directory of the dataset
        camera_id (str): ID of the camera to process
        timestep (str): Timestep of the image to process
        output_dir (str): Directory to save the cropped images
        images_dir (str): Directory containing the images
        mask_dir (str): Directory containing the masks
    """
    
    # Load and crop image
    try:
        # NOTE: if we later change the file structure format, then add a "subject_id" directory between base_dir and images_lr
        # however, since this we don't need this yet, keep as is
        img_path = os.path.join(base_dir, images_dir, camera_id, f"{timestep}_img.jpg")  # Assuming same timestep
        mask_path = os.path.join(base_dir, mask_dir, camera_id, f"{timestep}_img_fmask.png")

        # Get all camera directories in annots
        annot_dir = os.path.join(base_dir, 'annots', camera_id, f"{timestep}_img.json")
        annot = load_json(annot_dir)
        bbox = get_bbox_from_annot(annot)
        (center_x, center_y), _ = get_bbox_center_and_size(bbox)

        if os.path.exists(img_path):
            # mask the image
            img = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path)) / 255.0
            original_height, original_width = img.shape[:2]

            # crop the image
            crop_bbox, cropped_img = get_crop_coordinates(
                center_x * scale, center_y * scale, original_height, # homogeneous image size here
                original_width, original_height,
                image=img,
                mask=mask
            )

            # save the cropped image
            os.makedirs(output_dir, exist_ok=True)
            cropped_img = cropped_img.astype(np.uint8)
            cropped_img_pil = Image.fromarray(cropped_img, mode='RGB')
            cropped_img_pil.save(os.path.join(output_dir, f'{camera_id}_cropped_{timestep}.png'))
        else:
            print(f"Warning: Image not found at {img_path}")

    except Exception as e:
        print(f"Error processing image for camera {camera_id}: {e}")
        print(f"Error may be due to base_dir not being structured properly. The images directory must be 'images_lr' and the masks directory must be 'fmask_lr'.")


def crop_dataset_images(
    dataset_dir,
    images_dir='images_lr',
    mask_dir='fmask_lr',
    selected_dirs=None,
    new_dir_name="cropped_images",
    homogenous_image_size=None):
    """
    Crops all images for all cameras and timesteps in all subject directories.
    Args:
        dataset_dir (str): Top-level directory containing multiple subject directories
        images_dir (str): Directory containing the images (relative to each subject directory)
        mask_dir (str): Directory containing the masks (relative to each subject directory)
        selected_dirs (list/set, optional): List/Set of specific subject directories to process. If None, process all directories.
            These are number IDs corresponding their tar.gz files; we can list of IDs to crop.
    """
    from glob import glob
    # Get all subject directories (each containing a complete mvhumannet_format_dir structure)
    all_subject_dirs = [d for d in os.listdir(dataset_dir) 
                   if os.path.isdir(os.path.join(dataset_dir, d))]
    
    # If selected_dirs is provided, filter to only those directories
    if selected_dirs is not None:
        subject_dirs = [d for d in all_subject_dirs if d.split('.')[0] in selected_dirs]
        if len(subject_dirs) < len(selected_dirs):
            missing = set(selected_dirs) - set(subject_dirs)
            print(f"Warning: Some selected directories were not found: {missing}")
            # this OK and to be expected if tar.gz files are not contiguous
    else:
        subject_dirs = all_subject_dirs

    print(subject_dirs)
    
    for subject_dir in tqdm(subject_dirs, desc="Processing instances", ncols=80):
        base_dir = os.path.join(dataset_dir, subject_dir)
        subject_output_dir = os.path.join(base_dir, new_dir_name)

        # if os.path.exists(subject_output_dir):
        #     print(f"Skipping {subject_dir}: already exists")
        #     continue
        
        # Check if this directory has the expected structure
        if not os.path.exists(os.path.join(base_dir, images_dir)):
            print(f"Skipping {subject_dir}: missing {images_dir} directory")
            continue
            
        # Get all camera directories for this subject
        camera_dirs = [d for d in os.listdir(os.path.join(base_dir, images_dir)) 
                      if os.path.isdir(os.path.join(base_dir, images_dir, d))]
        
        for camera_id in tqdm(camera_dirs, desc=f"Processing {subject_dir} camera directiories", ncols=120):
            # Get all timesteps for this camera
            image_pattern = os.path.join(base_dir, images_dir, camera_id, "*_img.jpg")
            image_files = glob(image_pattern)
            
            # Create camera-specific output directory
            camera_output_dir = os.path.join(subject_output_dir, camera_id)
            if os.path.exists(camera_output_dir): # if already exists, we skip
                print(f"Skipping {camera_id} {timestep}: already exists")
                continue
            os.makedirs(camera_output_dir, exist_ok=True)
            
            for img_path in tqdm(image_files, desc=f"Processing camera {camera_id} timesteps", ncols=120):
                # Extract timestep from filename (e.g., "0015_img.jpg" -> "0015")
                filename = os.path.basename(img_path)
                timestep = filename.split('_')[0]
                
                # Process this image
                crop_and_save_image(
                    base_dir=base_dir,
                    camera_id=camera_id,
                    timestep=timestep,
                    output_dir=camera_output_dir,
                    images_dir=images_dir,
                    mask_dir=mask_dir,
                    homogenous_image_size=homogenous_image_size,
                )

    print(f"Processed all subjects and saved cropped images in their respective directories!")

def create_transforms_json(
    base_dir,
    output_dir,
    timestep,
    subject_id,
    save=True,
    tag_frames=False,
    homogeneous=True,
    num_orbital_frames=0,
    background=None,
    invert_rotations=True,
    homogenous_image_size=None,
    crop=True,
    normalize=False
):
    """
    Create transforms.json with cropped images and updated intrinsics.
    This function is geared towards SEVA forward inference.

    NOTE: assumes camera_extrinsics.json is w2c and inverts to c2w. If already c2w, then set invert_rotations=False.
    If normalize is True, will normalize camera poses and update intrinsics accordingly.
    """
    # currently, subject_id isn't used, but when the dataset is decompressed, then it should be used for inference.
    # Load crop parameters
    if crop:
        crop_params = process_image_crops_and_intrinsics(
            base_dir, timestep, scale=0.5, homogenous_image_size=homogenous_image_size)
    else:
        # keep default intrinsics instead (hacky!)
        intrinsics_path = os.path.join(base_dir, 'camera_intrinsics.json')
        if os.path.exists(intrinsics_path):
            intrinsics_data = load_json(intrinsics_path)
            K = np.array(intrinsics_data['intrinsics'])
            K[:2, :] = K[:2, :] / 2
        else:
            print(f"Warning: No intrinsics found at {intrinsics_path}. Using default values.")
            # hardcoded
            K = np.array([
                [2365.45 / 2, 0, 2047.5 / 2],
                [0, 2365.45 / 2, 1499.5 / 2],
                [0, 0, 1]
            ]) 
        
        # Create crop_params without actually cropping
        crop_params = {}
        # Get list of camera IDs from extrinsics
        extrinsics = load_json(os.path.join(base_dir, 'camera_extrinsics.json'))
        for camera_key in extrinsics.keys():
            camera_id = camera_key.split('_')[1].split('.')[0]  # Extract camera ID from "1_XXXX.png"
            
            # Set full image dimensions (no cropping)
            crop_params[camera_id] = {
                'crop': [0, 0, 2048, 1500],  # Full image, no cropping
                'K': K.tolist(),       # Original intrinsics
                'w': 2048,
                'h': 1500
            }
    
    # Load camera extrinsics
    extrinsics = load_json(os.path.join(base_dir, 'camera_extrinsics.json'))

    with open(os.path.join(base_dir, 'camera_scale.pkl'), 'rb') as f:
        camera_scale = pickle.load(f)

    if background: # may need to be an environment map (DEPRECATED)
        background_img = np.array(Image.open(background))
    
    # Create output directory for cropped images if it doesn't exist
    os.makedirs(output_dir, exist_ok=True) # follows the SEVA demo structure
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True) # puts images in directory
    # Prepare transforms.json structure
    transforms = {
        "frames": []
    }
    
    # Process each camera
    tag_id = 0
    for camera_id, camera_params in tqdm(crop_params.items(), desc="constructing transforms.json", ncols=80):
        # Get camera extrinsics
        camera_data = extrinsics.get(f"1_{camera_id}.png")
        if not camera_data:
            print(f"Warning: No extrinsics found for camera {camera_id}. This indicates a BIG error.")
            continue
            
        # Get crop parameters
        crop_params = camera_params['crop'] # at this point, this is a square
        K = np.array(camera_params['K']) # 0.5 scaled intrinsics

        if invert_rotations:
            transform_matrix = create_transform_matrix(
                np.linalg.inv(camera_data['rotation']), # get into c2w
                camera_data['camera_pos'] * np.array(camera_scale), # scale by camera_scale
                homogeneous=homogeneous
            )
            # NOTE: 'camera_pos' is essentially -R.T @ t, this skips some computation
        else:
            transform_matrix = create_transform_matrix(
                camera_data['rotation'], # already c2w
                camera_data['camera_pos'] * np.array(camera_scale), # scale by camera_scale
                homogeneous=homogeneous
            )

        # Rotate the transform_matrix 180 degrees around the z-axis
        # Create a rotation matrix for 180 degrees (π radians) around z-axis
        rotation_z_180 = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        # Apply the rotation to the transform matrix
        transform_matrix = rotation_z_180 @ transform_matrix @ rotation_z_180

        # transform_matrix[:, 0] = transform_matrix[:, 0]
        # transform_matrix[:, 1] = transform_matrix[:, 1]
        # transform_matrix[:, 2] = transform_matrix[:, 2]

        # Create frame entry
        crop_size = (crop_params[2] - crop_params[0], crop_params[3] - crop_params[1])
        crop_size = (int(crop_size[0]), int(crop_size[1]))
        frame = {
            "file_path": f"images/{camera_id}_cropped_{timestep}.png", # relative to output dir
            "transform_matrix": transform_matrix.tolist(),
            "w": crop_size[0],
            "h": crop_size[1],
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            # downsample due to dataset downsampling
            # extrinsics and intrinsics are NOT downsampled by default in their jsons.
        }
        
        if tag_frames:
            frame["tag_id"] = tag_id
            tag_id += 1
        
        transforms["frames"].append(frame)
        
        # Load and crop image
        if crop:
            crop_and_save_image(
                base_dir, camera_id, timestep, os.path.join(output_dir, 'images'),
                homogenous_image_size=homogenous_image_size)
        else:
            # Save the image without cropping
            img_path = os.path.join(base_dir, 'images_lr', camera_id, f"{timestep}_img.jpg")
            mask_path = os.path.join(base_dir, 'fmask_lr', camera_id, f"{timestep}_img_fmask.png")
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_path)) / 255.0
                
                # Apply mask to image
                masked_img = img * mask[..., None]
                
                # Save the masked image without cropping
                os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
                Image.fromarray(masked_img.astype(np.uint8)).save(
                    os.path.join(output_dir, 'images', f'{camera_id}_cropped_{timestep}.png'))
            else:
                print(f"Warning: Image or mask not found at {img_path} or {mask_path}")

    if num_orbital_frames > 0: # append any test frames (blank)
        center = get_central_position(transforms) # get center before extending test points
        orbital_transforms = generate_orbital_path(
            transforms, center,
            timestep=timestep,
            num_points=num_orbital_frames,
            radius=2000,
            direction_towards=center - np.array([0, 0, 650]) # TODO: un-hardcode this
        )
        transforms["frames"].extend(orbital_transforms)
        create_black_frames(
            orbital_transforms,
            os.path.join(output_dir, 'images'),
            timestep,
            image_size=(crop_size[1], crop_size[0])
        )

    # Save transforms.json
    if save:
        with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=4)
        print(f"Created transforms.json with {len(transforms['frames'])} frames")

    return transforms

def get_central_position(camera_extrinsics, w2c=False):
    """Calculate the approximate central position that cameras are looking at
    
    Args:
        camera_extrinsics (dict): Dictionary containing camera poses and parameters
        w2c (bool): Whether the transforms are world-to-camera (w2c) or camera-to-world (c2w)
        
    Returns:
        np.ndarray: 3D point representing the approximate center position
    """
    # Get all camera positions and directions
    positions = []
    directions = []

    if "frames" in camera_extrinsics:
        for frame in camera_extrinsics["frames"]:
            # should be c2w by default
            mat = np.array(frame["transform_matrix"])
            R = mat[:3, :3]
            t = mat[:3, 3]
            
            if w2c:
                pos = -R.T @ t
                forward = -R.T @ np.array([0, 0, 1])
            else:
                pos = t
                forward = -R @ np.array([0, 0, 1])
            positions.append(pos)
            directions.append(forward)
    else:
        # legacy format, w2c should be True
        for cam_data in camera_extrinsics.values():
            pos = np.array(cam_data['camera_pos'])
            positions.append(pos)
            R = np.array(cam_data['rotation'])
            if w2c:
                forward = -R.T @ np.array([0, 0, 1])
            else:
                forward = -R @ np.array([0, 0, 1])
                
            directions.append(forward)
    
    positions = np.array(positions)
    directions = np.array(directions)
    center = np.mean(positions, axis=0)
    
    # Could be made more sophisticated by finding intersection points
    # of camera direction rays, but mean position is a good approximation
    # when cameras are roughly arranged in a circle/sphere around subject
    return center


def generate_split_json(
    num_train_ids, 
    num_test_ids=None, 
    output_dir=".",
    output_file_prefix='train_test_split',
    transforms_file='transforms.json',
    train_ids=[],
    test_ids=[],
    step=1
):
    """
    Generate a train/test split JSON file with configurable ID lists.
    
    Args:
        num_train_ids (int): Number of training IDs to generate
        num_test_ids (int, optional): Number of test IDs to generate. If None, uses remaining frames
        output_file_prefix (str): Prefix for output JSON filename
        transforms_file (str): Path to transforms.json file containing frame data
    """
    # Get total number of available frames
    with open(transforms_file, 'r') as f:
        transforms_data = json.load(f)
        frames = transforms_data['frames']
        total_frames = len(frames)
        
        # Extract camera IDs from frame paths
        camera_ids = []
        for frame in frames:
            # Extract ID from file path like "images/CC32871A034_cropped_0150.png"
            camera_id = frame['file_path'].split('/')[1].split('_')[0]
            camera_ids.append(camera_id)

    # Validate inputs
    if num_train_ids <= 0:
        raise ValueError("num_train_ids must be positive")
    if num_train_ids > total_frames:
        raise ValueError(f"num_train_ids ({num_train_ids}) exceeds total frames ({total_frames})")
        
    # Calculate num_test_ids if not provided
    if num_test_ids is None:
        num_test_ids = total_frames - num_train_ids
    elif num_test_ids < 0:
        raise ValueError("num_test_ids must be non-negative")
        
    # Adjust test IDs if they would exceed total frames
    if num_train_ids + num_test_ids > total_frames:
        num_test_ids = total_frames - num_train_ids
        print(f"Warning: Reducing num_test_ids to {num_test_ids} to fit within total frames")

    # Generate train and test IDs
    if len(train_ids) == 0: # if empty use the first 'N' frames
        train_ids = list(range(num_train_ids)) # ordinal

    if len(test_ids) == 0: # if none were included, use all the others
        test_ids = list(range(num_train_ids, num_train_ids + num_test_ids))
    
    split_data = {
        "train_ids": train_ids[::step],
        "test_ids": test_ids
    }

    # Write split data to JSON file
    output_file = f"{output_dir}/{output_file_prefix}_{(num_train_ids if len(train_ids) == 0 else len(train_ids[::step]))}.json"
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=4)
        
    return split_data


def create_black_frames(transforms, output_dir, timestep, image_size=(1500, 1500)):
    """
    Creates black/mock frames with transforms from camera extrinsics needed for SEVA testing on
    a generated path with no available corresponding camera views.
    
    Args:
        transforms (dict): Dictionary containing camera poses and parameters (generated)
        image_size (tuple): Size of output images (width, height)
        intrinsics (dict): Dictionary containing camera intrinsics ('focal_length' and 'image_center')
        
    Returns:
        list: List of dictionaries containing frame data with transforms
    """
    plt.figure(figsize=(10,10))
    black_img = np.zeros((image_size[0], image_size[1])) # NOTE: may need to scale height/width such that intrinscs are integers
    plt.imshow(black_img, cmap='gray')
    plt.axis('off')
    for i, frame in tqdm(enumerate(transforms), desc="creating orbital test frames", ncols=100):
        plt.savefig(os.path.join(output_dir, f"orbital_{str(i).zfill(4)}_{timestep}.png"), bbox_inches='tight', pad_inches=0)


def generate_average_intrinsics(transforms):
    # computes average intrinsics from all frames such that 
    # test images are rendered with homogeneous intrinsics
    intrinsics = []
    for frame in transforms["frames"]:
        intrinsics.append({
            'w': frame['w'],
            'h': frame['h'],
            'fl_x': frame['fl_x'],
            'fl_y': frame['fl_y'],
            'cx': frame['cx'],
            'cy': frame['cy']
        })

    return {
        'w': np.mean([i['w'] for i in intrinsics]),
        'h': np.mean([i['h'] for i in intrinsics]),
        'fl_x': np.mean([i['fl_x'] for i in intrinsics]),
        'fl_y': np.mean([i['fl_y'] for i in intrinsics]),
        'cx': np.mean([i['cx'] for i in intrinsics]),
        'cy': np.mean([i['cy'] for i in intrinsics])
    }

def generate_orbital_extrinsics(
    center_point, 
    radius, 
    num_points, 
    orbit_normal=[0, 0, 1], 
    up_reference=[0, 0, 1],
    direction_towards=None
):
    """
    Generate extrinsic matrices (transformation matrices) for a circular orbital path around a center point.
    
    Parameters:
    - center_point: 3D list/array [x, y, z] - the point to orbit around
    - radius: float - orbital radius
    - num_points: int - number of points in the orbit
    - orbit_normal: 3D list/array - defines the orientation of the orbital plane (default is Y-axis normal)
    - up_reference: 3D list/array - reference vector for camera up direction (default is Z-axis up)
    - direction_towards: 3D list/array - if provided, cameras will look towards this point instead of center_point
    
    Returns:
    - List of 4x4 numpy arrays representing the transformation matrices at each orbital point
    """
    
    # Convert inputs to numpy arrays
    center = np.array(center_point, dtype=np.float32)
    normal = np.array(orbit_normal, dtype=np.float32)
    up_ref = np.array(up_reference, dtype=np.float32)
    
    # Normalize the orbit normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Find two orthogonal vectors in the orbital plane
    # First, find a vector not parallel to normal to create the first axis
    if not np.allclose(normal, [0, 0, 1]):
        axis1 = np.cross(normal, [0, 0, 1])
    else:
        axis1 = np.cross(normal, [1, 0, 0])
    axis1 = axis1 / np.linalg.norm(axis1)
    
    # Second axis is perpendicular to both normal and axis1
    axis2 = np.cross(normal, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    
    # Determine the look-at point
    look_at_point = np.array(direction_towards, dtype=np.float32) if direction_towards is not None else center
    
    extrinsics = []
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Position in world coordinates
        position = center + x * axis1 + y * axis2
        
        # Calculate view direction (from position to look_at_point)
        view_dir = look_at_point - position
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Calculate right vector (in orbital plane)
        right = np.cross(view_dir, normal)
        # If right vector is too small, find another perpendicular vector
        if np.linalg.norm(right) < 1e-6:
            # Find any vector perpendicular to view_dir
            if not np.allclose(view_dir, [1, 0, 0]):
                right = np.cross(view_dir, [1, 0, 0])
            else:
                right = np.cross(view_dir, [0, 1, 0])
        right = right / np.linalg.norm(right)
        
        # Recalculate up vector to ensure orthogonality
        up = np.cross(right, view_dir)
        up = up / np.linalg.norm(up)
        
        # If up reference is provided, adjust orientation to match as closely as possible
        if up_ref is not None:
            # Project reference up onto the camera's up plane
            up_ref_proj = up_ref - np.dot(up_ref, view_dir) * view_dir
            if np.linalg.norm(up_ref_proj) > 1e-6:
                up_ref_proj = up_ref_proj / np.linalg.norm(up_ref_proj)
                # Find the angle between current up and reference up
                angle = np.arctan2(np.dot(np.cross(up, up_ref_proj), view_dir), np.dot(up, up_ref_proj))
                # Rotate around view direction to align with reference up
                rot_angle = angle
                c, s = np.cos(rot_angle), np.sin(rot_angle)
                right = c * right + s * np.cross(view_dir, right)
                up = np.cross(right, view_dir)
        
        # Create the transformation matrix
        transform = np.eye(4)
        transform[:3, 0] = right      
        transform[:3, 1] = up         
        transform[:3, 2] = -view_dir  
        transform[:3, 3] = position   
        extrinsics.append(transform)
    
    return extrinsics

def generate_orbital_path(transforms, center, timestep, num_points, radius=1000, direction_towards=None):
    """ Generates orbital path around the center with num_points around it. Returns List of transforms."""
    # transforms to get average intrinscs
    avg_intrinsics = generate_average_intrinsics(transforms)
    # TODO: if radius is None, then use the length from the center to the closest cameras
    orbital_extrinsics = generate_orbital_extrinsics(center, radius=radius, num_points=num_points, direction_towards=direction_towards)

    # create orbital transforms
    orbital_transforms = []
    for i, extrinsic in enumerate(orbital_extrinsics):
        orbital_transforms.append({
            "file_path": f"images/orbital_{str(i).zfill(4)}_{timestep}.png",
            "transform_matrix": extrinsic.tolist(),
            **avg_intrinsics
        })

    return orbital_transforms

def apply_to_all_subjects(base_dir, function, *args, **kwargs):
    """
    Applies a function to all subjects in the mvset directory.
    
    Args:
        base_dir (str): Base directory containing all subject folders
        function (callable): Function to apply to each subject
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        dict: Dictionary mapping subject IDs to function results
    """
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d)) 
                   and not d.startswith('.')]

    print(subject_dirs)
    
    results = {}
    
    for subject_id in subject_dirs:
        subject_path = os.path.join(base_dir, subject_id)
        print(f"Processing subject: {subject_id}")
        
        try:
            # Apply the function to this subject
            result = function(subject_path, *args, **kwargs)
            results[subject_id] = result
        except Exception as e:
            print(f"Error processing subject {subject_id}: {str(e)}")
            results[subject_id] = None
    
    return results

def save_crop_params(subject_dir, homogenous_image_size=(2048, 1500)):
    crop_dict = process_image_crops_and_intrinsics(
        subject_dir,
        homogenous_image_size=homogenous_image_size
    )
    # Save crop_dict to a JSON file
    crop_dict_path = os.path.join(subject_dir, 'crop_dict.json')

    serializable_crop_dict = {}
    for camera, data in crop_dict.items():
        serializable_crop_dict[camera] = {
            'crop': data['crop'],
            'K': data['K'].tolist() if isinstance(data['K'], np.ndarray) else data['K']
        }

    os.makedirs(subject_dir, exist_ok=True)
    with open(crop_dict_path, 'w') as f:
        json.dump(serializable_crop_dict, f, indent=4)

    print(f"Crop dictionary saved to {crop_dict_path}")


def run_main(args):
    # Update global variables based on arguments
    assert args.timestep % 5 == 0, "Timestep must be a multiple of 5 lower than the maximum timestep."
    IMAGE_AT_TIME = f"{str(args.timestep).zfill(4)}"

    # TEMPORARY: chooses the first range from 100001 to 101000 subjects.
    if args.crop_only: # data preprocessing (training)
        # crop_dataset_images(
        #     args.base_dir,
        #     # selected_dirs=["mvhumannet_format_dir"],
        #     selected_dirs=[str(i) for i in range(100001, 101001)],
        #     homogenous_image_size=(2048, 1500)
        # )

        # save-only, so we don't need to return anything
        apply_to_all_subjects(args.base_dir, save_crop_params, homogenous_image_size=(2048, 1500))
        return

    # compute number of orbital frames (w.r.t seconds and fps)
    generate_path = False
    if args.num_orbital_frames <= -1:
        num_orbital_frames = int(args.seconds * args.fps)
        generate_path = True
    else:
        num_orbital_frames = args.num_orbital_frames

    # beyond this point is SEVA INFERENCE preprocessing:
    if not args.apply_split_only:  # then re-generate transforms.json based on existing
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        try:
            transforms = create_transforms_json(
                args.base_dir,
                args.output_dir,
                IMAGE_AT_TIME,
                args.subject_id,
                save=True,
                tag_frames=args.tag_frames,
                homogeneous=args.nonhomogeneous,
                num_orbital_frames=num_orbital_frames,
                background=args.background_path,
                invert_rotations=args.c2w,
                crop=args.no_crop)
                
        except Exception as e:
            os.rmdir(args.output_dir) 
            raise e

        if args.normalize:
            # Load all poses and intrinsics
            poses = []
            intrinsics = []
            for frame in transforms["frames"]:
                poses.append(np.array(frame["transform_matrix"]))
                intrinsics.append(np.array([
                    [frame["fl_x"], 0, frame["cx"]],
                    [0, frame["fl_y"], frame["cy"]],
                    [0, 0, 1]
                ]))

            # Normalize all poses and intrinsics together
            normalized_poses = normalize_camera_poses(
                poses, scale_factor=1.0)

            # Update transforms.json with normalized values
            for i, frame in enumerate(transforms["frames"]):
                frame["transform_matrix"] = normalized_poses[i].tolist()
                frame["fl_x"] = float(intrinsics[i][0, 0])
                frame["fl_y"] = float(intrinsics[i][1, 1])
                # cx, cy and image dimensions remain unchanged
        
        # if we apply any post-transformations (SEVA will handle this internally)
        # then we append a transformation matrix to apply to every matrix
        if args.transform_coords is not None:
            post_apply_transform_mat = np.loadtxt(args.transform_coords)
            transforms["applied_transform"] = post_apply_transform_mat.tolist()
        
        # write transforms.json to file
        with open(os.path.join(args.output_dir, 'transforms.json'), 'w') as f:
            json.dump(transforms, f, indent=4)
        print(f"Cropped images saved in {args.output_dir}/")

    train_ids = [] # used if we want to manually select training IDs for train part of the split
    if args.train_ids_path is not None:
        train_ids = match_camera_ids_to_index(args.train_ids_path, os.path.join(args.output_dir, 'transforms.json'), step=1)

    if generate_path:
        try: # this will error with --apply_split_only
            test_ids = list(range(len(transforms["frames"]) - num_orbital_frames, len(transforms["frames"])))
        except:
            with open(os.path.join(args.output_dir, 'transforms.json'), 'r') as f:
                transforms_data = json.load(f)
                num_frames = len(transforms_data.get("frames", []))
                test_ids = list(range(num_frames - num_orbital_frames, num_frames)) if num_frames > 0 else []
    else:
        test_ids = args.test_ids

    # generate train_test_split.json file
    generate_split_json(
        args.num_train_frames,
        output_dir=args.output_dir,
        transforms_file=os.path.join(args.output_dir, 'transforms.json'),
        train_ids=train_ids,
        test_ids=test_ids,
        step=args.train_ids_step)  # Creates file with train_ids [0,1,2] and empty test_ids
    print(f"Generated train_test_split_{len(train_ids[::args.train_ids_step]) if len(train_ids) > 0 else args.num_train_frames}.json")

if __name__ == "__main__":
    # TODO: should split this up into different cases for different preprocessing actions

    parser = argparse.ArgumentParser(description='Process and crop multi-view images')
    parser.add_argument('--base_dir', type=str, default='.', required=True, 
                      help='Base directory containing images and camera data. This should be where mvhumannet resides. (When cropping dataset, this becomes the dir that holds all subjects.)\
                      \nNOTE: requires images_lr and fmask_lr directories to be present within the directory. \
                      \nAlso requires camera_intrinsics.json and camera_extrinsics.json to be present within the directory.')
    parser.add_argument('--output_dir', type=str, default='./processed_imgs', required=True,
                      help='Output directory for cropped images, updated transforms.json')
    parser.add_argument('--timestep', type=int, default=5, required=True,
                      help='Timestep/frame to choose.')

    parser.add_argument('--crop_only', action='store_true', help="If true, then images will be cropped and saved within their respective subject directories.")
    parser.add_argument('--subject_id', type=int, default=5, required=False,
                      help='Which human ID to choose from the dataset. This corresponds to the tar.gz filename in which the images are extracted from. If not given, chooses all.')
    parser.add_argument('--transform_coords', type=str, default=None, required=False,
                      help='Converts transforms.json output for SEVA. (Either OPENCV or OPENGL.) Expects a 3x4 numpy array txt file.')
    parser.add_argument('--num_train_frames', type=int, default=1, required=False,
                      help="Number of train frames to include from the transforms.json file for SEVA inference. The rest will be used for testing. Currently, these are selected ordinally.")
    parser.add_argument('--nonhomogeneous', action='store_false',
                      help="If true, the transform matrix will be homogeneous (4x4). Otherwise, it will be 3x4.")
    parser.add_argument('--tag_frames', action='store_true',
                      help="Adds id key to each frame in transforms.json. Not required and will be unused in SEVA. Helpful for indexing.")
    parser.add_argument('--test_ids', nargs='+', type=int, default=[], help="List of test IDs to include in the train/test split. If not provided, will use all remaining frames, except in the case of generated test poses (which will all be included in test)")
    parser.add_argument('--train_ids_path', type=str, help="Path to a file containing a list of train IDs to include in the train/test split. Will overwrite num_train_frames.")
    parser.add_argument('--train_ids_step', type=int, default=1, help="Step size to use when reading train IDs from --train_ids_path or sequential from num_train_frames.")
    parser.add_argument('--apply_split_only', action='store_true', help="If true, will only apply the split to the existing transforms.json file and skip generating.")
    parser.add_argument('--background_path', type=str, help="Path to a background image to add to the images. Otherwise, images are not modified.")
    parser.add_argument('--c2w', action='store_false', help="If flagged, then camera_extrinsics.json is assumed to be c2w and will not go through conversion.")
    parser.add_argument('--num_orbital_frames', type=int, default=-1, required=False,
                      help="Number of orbital frames to add to the end of the transforms.json file. Will ignore seconds and fps arguments.")
    parser.add_argument('--seconds', type=float, default=4.0, help="Number of seconds per path.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second.")
    parser.add_argument('--no_crop', action='store_false', help="If true, will not crop the images.")
    parser.add_argument('--normalize', action='store_true', help="If true, will normalize the poses to a smaller scale.")
    
    args = parser.parse_args()
    run_main(args)


