import os
import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm

# Load the VAE model of Stable Diffusion 2.1
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").cuda()
vae.eval()

# Define the dataset directory and target directory
DATASET_DIR = "/workspace/datasetvol/mvhuman_data/mv_captures"
TARGET_DIR = "/workspace/datasetvol/mvhuman_data/mv_latents"

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Precompute latents from images')
parser.add_argument('--max_subjects', type=int, default=None,
                   help='Number of subjects to process (default: process all)')
parser.add_argument('--overwrite', action='store_false',
                   help='Overwrite existing latents (default: False)')

args = parser.parse_args()


# Define image preprocessing
# NOTE: remember to update intrinsics for crop!
# preprocess = transforms.Compose([
#     transforms.CenterCrop(540),         # Center crop to square (DL3DV images are 940x540)
#     transforms.Resize((576, 576)),      # Resize to 512x512
#     transforms.ToTensor(),              # Convert to tensor
#     transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
# ])
preprocess = transforms.Compose([
    transforms.CenterCrop(1500),        # Center crop to square (MVHumanNet images are 2048x1500)
    transforms.Resize((576, 576)),      # Resize to 576x576
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Function to process images and save latents
# DL3DV
# def process_and_save_latents(scene_path, target_scene_path):
#     os.makedirs(target_scene_path, exist_ok=True)
#     images_dir = os.path.join(scene_path, "images_4")
#     target_images_dir = os.path.join(target_scene_path, "images_4")
#     os.makedirs(target_images_dir, exist_ok=True)

#     image_names = [name for name in os.listdir(images_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     batch_size = 32

#     for i in range(0, len(image_names), batch_size):
#         batch_names = image_names[i:i + batch_size]
#         batch_images = []

#         # Load and preprocess images in the batch
#         for image_name in batch_names:
#             image_path = os.path.join(images_dir, image_name)
#             image = Image.open(image_path).convert("RGB")
#             image_tensor = preprocess(image)
#             batch_images.append(image_tensor)

#         # Stack images into a batch tensor
#         batch_tensor = torch.stack(batch_images).cuda()

#         # Encode the batch to latents
#         with torch.no_grad():
#             latents = vae.encode(batch_tensor).latent_dist.sample()
#             latents = latents.cpu()  # Move to CPU

#         # Save each latent in the batch
#         for j, image_name in enumerate(batch_names):
#             latent_path = os.path.join(target_images_dir, f"{os.path.splitext(image_name)[0]}.pt")
#             torch.save(latents[j], latent_path)

# # Iterate through the dataset and process each scene
# for subfolder_name in tqdm(os.listdir(DATASET_DIR), desc="Processing subfolders"):
#     subfolder_path = os.path.join(DATASET_DIR, subfolder_name)
#     if os.path.isdir(subfolder_path):
#         for scene_name in tqdm(os.listdir(subfolder_path), desc=f"Processing scenes in {subfolder_name}", leave=False):
#             scene_path = os.path.join(subfolder_path, scene_name)
#             if os.path.isdir(scene_path):
#                 target_scene_path = os.path.join(TARGET_DIR, subfolder_name, scene_name)
#                 process_and_save_latents(scene_path, target_scene_path)

# MVHumanNet
def process_and_save_latents(scene_path, target_scene_path):
    print(f"\nProcessing scene: {scene_path}")
    os.makedirs(target_scene_path, exist_ok=True)
    
    # Create images_lr directory in target path
    target_images_dir = os.path.join(target_scene_path, "images_lr")
    os.makedirs(target_images_dir, exist_ok=True)
    
    # Get all camera directories from images_lr
    images_lr_path = os.path.join(scene_path, "images_lr")
    if not os.path.exists(images_lr_path):
        print(f"No images_lr directory found in {scene_path}")
        return
        
    camera_dirs = [d for d in os.listdir(images_lr_path) if os.path.isdir(os.path.join(images_lr_path, d)) and d.startswith('CC')]
    print(f"Found camera directories: {camera_dirs}")
    batch_size = 16

    for camera_dir in camera_dirs:
        # Create camera-specific directory in target
        target_camera_dir = os.path.join(target_images_dir, camera_dir)
        os.makedirs(target_camera_dir, exist_ok=True)
        
        # Get all images for this camera
        camera_path = os.path.join(images_lr_path, camera_dir)
        print(f"Checking camera path: {camera_path}")
        image_names = [name for name in os.listdir(camera_path) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_names)} images in {camera_dir}")
        
        # Sort images by frame number
        image_names.sort(key=lambda x: int(x.split('_')[0]))

        # Filter out existing latents if not overwriting
        if not args.overwrite:
            image_names = [name for name in image_names if not os.path.exists(
                os.path.join(target_camera_dir, f"{name.split('_')[0]}_latent.pt")
            )]
        # else: implicit full overwrite (uses full image_names list)

        if not image_names: # empty list
            print(f"Skipping {camera_dir} - all latents already exist")
            continue

        print(f"Processing {len(image_names)} images in {camera_dir}")
        for i in range(0, len(image_names), batch_size):
            batch_names = image_names[i:i + batch_size]
            batch_images = []

            # Load and preprocess images in the batch
            for image_name in batch_names:
                image_path = os.path.join(camera_path, image_name)
                print(f"Loading image: {image_path}")
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)

            # Stack images into a batch tensor
            batch_tensor = torch.stack(batch_images).cuda()
            batch_tensor.requires_grad = False

            # Encode the batch to latents
            with torch.no_grad():
                latents = vae.encode(batch_tensor).latent_dist.sample()
                latents = latents.cpu()  # Move to CPU

            # Save each latent in the batch
            for j, image_name in enumerate(batch_names):
                # Extract frame number and create new filename
                frame_num = image_name.split('_')[0]
                latent_path = os.path.join(target_camera_dir, f"{frame_num}_latent.pt")
                torch.save(latents[j], latent_path)
                print(f"Saved latent for {frame_num}")
            
            # Clear GPU memory after each batch
            del batch_tensor, latents
            torch.cuda.empty_cache()

# Iterate through the dataset and process each subject
i = 0
print(f"Dataset directory: {DATASET_DIR}")
print(f"Target directory: {TARGET_DIR}")
for subject in tqdm(os.listdir(DATASET_DIR), desc="Processing subjects"):
    if args.max_subjects is not None and i >= args.max_subjects:
        break
    i += 1

    subject_path = os.path.join(DATASET_DIR, subject)
    if os.path.isdir(subject_path):
        print(f"\nFound subject directory: {subject_path}")
        target_subject_path = os.path.join(TARGET_DIR, subject)
        process_and_save_latents(subject_path, target_subject_path)

print("Processing complete. Latents saved.")

