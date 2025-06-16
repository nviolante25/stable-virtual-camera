import os
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm
import argparse
from functools import partial
import numpy as np

# Load the VAE model of Stable Diffusion 2.1
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").cuda()
vae.eval()

# Define the dataset directory and target directory
DATASET_DIR = "/workspace/datasetvol/mvhuman_data/mv_captures"
TARGET_DIR = "/workspace/datasetvol/mvhuman_data/mv_latents"

# Set up argument parser
parser = argparse.ArgumentParser(description='Precompute latents from images')
parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                   help='Dataset directory (default: /workspace/datasetvol/mvhuman_data/mv_captures)')
parser.add_argument('--target_dir', type=str, default=TARGET_DIR,
                   help='Target directory to store computed latents (default: /workspace/datasetvol/mvhuman_data/mv_latents)')
parser.add_argument('--max_subjects', type=int, default=None,
                   help='Number of subjects to process (default: process all)')
parser.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing latents (default: False)')
parser.add_argument('--batch_size', type=int, default=22,
                   help='Batch size for encoding latents (default: 16)')

args = parser.parse_args()
DATASET_DIR = args.dataset_dir
TARGET_DIR = args.target_dir
BATCH_SIZE = args.batch_size

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.CenterCrop(1500),        # Center crop to square (MVHumanNet images are 2048x1500)
    transforms.Resize((576, 576)),      # Resize to 576x576
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

def process_batch(model, batch_images, device):
    """Process a batch of images and return latents"""
    batch_tensor = torch.stack(batch_images).to(device)
    with torch.no_grad():
        latents = model.encode(batch_tensor).latent_dist.sample()
    return latents.cpu()

def process_camera_dir(model, camera_path, target_camera_dir, device, batch_size=BATCH_SIZE):
    """Process all images in a camera directory"""
    # Get and sort images
    image_names = sorted([
        name for name in os.listdir(camera_path) 
        if name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=lambda x: int(x.split('_')[0]))
    
    # Filter existing latents if not overwriting
    if not args.overwrite:
        image_names = [
            name for name in image_names 
            if not os.path.exists(os.path.join(target_camera_dir, f"{name.split('_')[0]}_latent.npz"))
        ]
    
    # Process in batches
    for i in range(0, len(image_names), batch_size):
        batch_names = image_names[i:i + batch_size]
        batch_images = []
        
        # Load and preprocess images
        for image_name in batch_names:
            try:
                image_path = os.path.join(camera_path, image_name)
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Process batch
        try:
            latents = process_batch(model, batch_images, device)
            
            # Save latents
            for j, image_name in enumerate(batch_names):
                frame_num = image_name.split('_')[0]
                latent_path = os.path.join(target_camera_dir, f"{frame_num}_latent.npz")
                # Convert to numpy and save compressed
                latent_np = latents[j].cpu().numpy()
                np.savez_compressed(latent_path, latent=latent_np)
                # if legacy format exists, delete it
                if os.path.exists(os.path.join(target_camera_dir, f"{frame_num}_latent.pt")):
                    os.remove(os.path.join(target_camera_dir, f"{frame_num}_latent.pt"))
            
            # Clear GPU memory
            del latents
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

def worker(rank, world_size, subject_chunks):
    """Worker process for handling a subset of subjects"""
    # Set device for this process
    device = f"cuda:{rank}"
    print(f"Worker {rank}: Using device {device}")
    torch.cuda.set_device(device)
    
    # Load model on current GPU
    model = vae.to(device)
    model.eval()
    
    # Get this worker's subjects
    subjects = subject_chunks[rank]
    print(f"Worker {rank}: Processing {len(subjects)} subjects")
    
    # Process assigned subjects
    for subject in subjects:
        subject_path = os.path.join(DATASET_DIR, subject)
        if not os.path.isdir(subject_path):
            continue
            
        target_subject_path = os.path.join(TARGET_DIR, subject)
        if os.path.exists(target_subject_path) and not args.overwrite:
            print(f"Worker {rank}: Skipping {subject} - already exists")
            continue
            
        print(f"Worker {rank}: Processing subject {subject}")
        
        # Create directories
        os.makedirs(target_subject_path, exist_ok=True)
        target_images_dir = os.path.join(target_subject_path, "images_lr")
        os.makedirs(target_images_dir, exist_ok=True)
        
        # Get camera directories
        images_lr_path = os.path.join(subject_path, "images_lr")
        if not os.path.exists(images_lr_path):
            print(f"Worker {rank}: No images_lr directory found in {subject_path}")
            continue
            
        camera_dirs = sorted([
            d for d in os.listdir(images_lr_path) 
            if os.path.isdir(os.path.join(images_lr_path, d)) and d.startswith('CC')
        ])
        
        # Process each camera directory
        for camera_dir in camera_dirs:
            camera_path = os.path.join(images_lr_path, camera_dir)
            target_camera_dir = os.path.join(target_images_dir, camera_dir)
            os.makedirs(target_camera_dir, exist_ok=True)
            
            print(f"Worker {rank}: Processing camera {camera_dir} in {subject}")
            process_camera_dir(model, camera_path, target_camera_dir, device)
            
        print(f"Worker {rank}: Finished subject {subject}")

def main():
    # Get list of subjects
    subjects = sorted([
        d for d in os.listdir(DATASET_DIR) 
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    
    if args.max_subjects is not None:
        subjects = subjects[:args.max_subjects]
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    # Split subjects among GPUs
    chunk_size = (len(subjects) + world_size - 1) // world_size
    subject_chunks = [
        subjects[i:i + chunk_size]
        for i in range(0, len(subjects), chunk_size)
    ]
    
    # Ensure we have enough chunks for all GPUs
    while len(subject_chunks) < world_size:
        subject_chunks.append([])
    
    # Launch processes
    mp.spawn(
        worker,
        args=(world_size, subject_chunks),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()