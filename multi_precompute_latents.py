import os
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
import numpy as np
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Tuple
import torch.nn.functional as F

# Set up multiprocessing with spawn method
mp.set_start_method('spawn', force=True)

# Constants for image processing
IMAGE_SIZE = 576
CENTER_CROP_SIZE = 1500

# Define image preprocessing with optimized operations
# NOTE: for now, center crop, but afterwards, should do random crop (for SimVS)
preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((
        (img.width - CENTER_CROP_SIZE) // 2,  # left
        (img.height - CENTER_CROP_SIZE) // 2,  # top
        (img.width + CENTER_CROP_SIZE) // 2,   # right
        (img.height + CENTER_CROP_SIZE) // 2   # bottom
    ))),
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Define mask preprocessing (same as image but without normalization)
mask_preprocess = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((
        (img.width - CENTER_CROP_SIZE) // 2,  # left
        (img.height - CENTER_CROP_SIZE) // 2,  # top
        (img.width + CENTER_CROP_SIZE) // 2,   # right
        (img.height + CENTER_CROP_SIZE) // 2   # bottom
    ))),
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

class VAEWorker:
    def __init__(self, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = f"cuda:{rank}"
        
        # Initialize model on this GPU
        self.model = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", 
            subfolder="vae"
        ).to(self.device)
        self.model.eval()
        
        # Warm up the model
        self._warmup_model()
    
    def _warmup_model(self):
        """Warm up the model with a dummy input"""
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        with torch.no_grad():
            _ = self.model.encode(dummy_input).latent_dist.sample()
        torch.cuda.synchronize()
    
    def process_batch(self, batch_images):
        """Process a batch of images and return latents"""
        with torch.no_grad():
            batch_tensor = torch.stack(batch_images).to(self.device, non_blocking=True)
            latents = self.model.encode(batch_tensor).latent_dist.sample()
            return latents.detach().cpu()
    
    def _load_and_preprocess_batch(self, image_paths: List[str], mask_paths: List[str]) -> List[torch.Tensor]:
        """Load and preprocess a batch of images with their masks using vectorized operations"""
        batch_images = []
        
        # Load all images and masks in parallel with increased workers
        with ThreadPoolExecutor(max_workers=self.args.io_workers) as executor:
            # Load images and masks concurrently
            image_futures = [executor.submit(self._load_image, path) for path in image_paths]
            mask_futures = [executor.submit(self._load_mask, path) for path in mask_paths]
            
            # Collect results as they complete
            images = [future.result() for future in image_futures]
            masks = [future.result() for future in mask_futures]
        
        # Filter out None results
        valid_pairs = [(img, mask) for img, mask in zip(images, masks) if img is not None and mask is not None]
        
        if not valid_pairs:
            return []
        
        # Unzip valid pairs
        valid_images, valid_masks = zip(*valid_pairs)
        
        # Convert to tensors and stack for vectorized operations
        # Use pin_memory=True for faster CPU->GPU transfer
        image_tensors = torch.stack(valid_images).pin_memory()
        mask_tensors = torch.stack(valid_masks).pin_memory()
        
        # Vectorized masking operation with optimized broadcasting
        # Expand mask from (B, 1, H, W) to (B, 3, H, W) to match image channels
        # Use repeat instead of expand for better memory efficiency
        mask_tensors = mask_tensors.repeat(1, 3, 1, 1)
        masked_images = image_tensors * mask_tensors
        
        # Convert back to list for compatibility
        return [masked_images[i] for i in range(masked_images.shape[0])]
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess a single image with optimized I/O"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB immediately to avoid repeated conversions
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                return preprocess(img)
        except Exception as e:
            print(f"Worker {self.rank}: Error loading image {image_path}: {e}")
            return None
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess a single mask with optimized I/O"""
        try:
            with Image.open(mask_path) as mask:
                # Convert to grayscale immediately
                if mask.mode != 'L':
                    mask = mask.convert("L")
                return mask_preprocess(mask)
        except Exception as e:
            print(f"Worker {self.rank}: Error loading mask {mask_path}: {e}")
            return None
    
    def _load_and_preprocess(self, image_path, mask_path, save_test_image=False):
        """Legacy method - kept for compatibility but not used in optimized version"""
        try:
            # Load image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_tensor = preprocess(img)
            
            # Load mask
            with Image.open(mask_path) as mask:
                mask = mask.convert("L")  # Convert to grayscale
                # Use the same preprocessing pipeline as images (center crop + resize)
                mask_tensor = mask_preprocess(mask)
            
            # Apply mask (mask is 0-1, where 1 is foreground)
            # For VAE encoding, we typically want to mask out background (set to 0)
            # So we multiply the image by the mask
            masked_image = img_tensor * mask_tensor
            
            # Save test image if requested
            if save_test_image:
                self._save_test_masked_image(image_path, img_tensor, mask_tensor, masked_image)
            
            return masked_image
            
        except Exception as e:
            print(f"Worker {self.rank}: Error processing image {image_path}: {e}")
            return None
    
    def _save_test_masked_image(self, image_path, original_tensor, mask_tensor, masked_tensor):
        """Save test images to verify masking"""
        try:
            # Create test directory
            test_dir = "/workspace/stable-virtual-camera/mask_test"
            os.makedirs(test_dir, exist_ok=True)
            
            # Get base filename
            base_name = os.path.basename(image_path).split('.')[0]
            
            # Convert tensors back to PIL images for saving
            def tensor_to_pil(tensor):
                print(tensor.shape)
                # Denormalize from [-1, 1] to [0, 255]
                tensor = (tensor + 1) / 2
                tensor = torch.clamp(tensor, 0, 1)
                # Convert to PIL - tensors are already in (C, H, W) format
                if tensor.shape[0] == 3:  # RGB
                    # No need to permute - ToPILImage expects (C, H, W)
                    return transforms.ToPILImage()(tensor)
                else:  # Grayscale
                    tensor = tensor.squeeze()
                    return transforms.ToPILImage()(tensor)
            
            # Save original image
            original_pil = tensor_to_pil(original_tensor)
            original_pil.save(os.path.join(test_dir, f"{base_name}_original.png"))
            
            # Save mask
            mask_pil = tensor_to_pil(mask_tensor)
            mask_pil.save(os.path.join(test_dir, f"{base_name}_mask.png"))
            
            # Save masked image
            masked_pil = tensor_to_pil(masked_tensor)
            masked_pil.save(os.path.join(test_dir, f"{base_name}_masked.png"))
            
            print(f"Worker {self.rank}: Saved test images for {base_name}")
            
        except Exception as e:
            print(f"Worker {self.rank}: Error saving test image: {e}")
    
    def process_camera_dir(self, camera_path, target_camera_dir, mask_camera_path):
        """Process all images in a camera directory with optimized batch processing and prefetching"""
        # Get and sort images with thread-safe glob
        image_paths = sorted(
            [os.path.join(camera_path, name) for name in os.listdir(camera_path) 
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        # mask images as well
        mask_paths = sorted(
            [os.path.join(mask_camera_path, name) for name in os.listdir(mask_camera_path) 
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        # Filter existing latents if not overwriting
        if not self.args.overwrite:
            image_paths = [
                path for path in image_paths 
                if not os.path.exists(os.path.join(
                    target_camera_dir, 
                    f"{os.path.basename(path).split('_')[0]}_latent.npz"
                ))
            ]
            
            # Filter corresponding mask paths
            mask_paths = [
                mask_paths[i] for i, img_path in enumerate(image_paths)
                if img_path in image_paths
            ]
        
        # Prefetch next batch while processing current batch
        def prefetch_batch(batch_image_paths, batch_mask_paths):
            """Prefetch the next batch of images and masks"""
            return self._load_and_preprocess_batch(batch_image_paths, batch_mask_paths)
        
        # Process in batches with prefetching
        for i in range(0, len(image_paths), self.args.batch_size):
            batch_image_paths = image_paths[i:i + self.args.batch_size]
            batch_mask_paths = mask_paths[i:i + self.args.batch_size]
            
            # Start prefetching next batch if available
            next_batch_future = None
            if i + self.args.batch_size < len(image_paths):
                next_batch_image_paths = image_paths[i + self.args.batch_size:i + 2 * self.args.batch_size]
                next_batch_mask_paths = mask_paths[i + self.args.batch_size:i + 2 * self.args.batch_size]
                with ThreadPoolExecutor(max_workers=1) as executor:
                    next_batch_future = executor.submit(prefetch_batch, next_batch_image_paths, next_batch_mask_paths)
            
            # Load current batch
            batch_images = self._load_and_preprocess_batch(batch_image_paths, batch_mask_paths)
            
            if not batch_images:
                continue
            
            try:
                # Process batch
                start_time = time.time()
                latents = self.process_batch(batch_images)
                
                # Save latents in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    executor.map(self._save_latent, batch_image_paths, latents)
                
                # Log performance
                batch_time = time.time() - start_time
                print(f"Worker {self.rank}: Processed {len(batch_images)} images in {batch_time:.2f}s "
                      f"({len(batch_images)/batch_time:.2f} img/s)")
                
                # Clean up
                del latents, batch_images
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Worker {self.rank}: Error processing batch: {e}")
                continue
    
    def _save_latent(self, image_path, latent):
        """Save a single latent tensor"""
        # mask images as well
        try:
            frame_num = os.path.basename(image_path).split('_')[0]

            latent_path = os.path.join(
                os.path.dirname(image_path).replace(self.args.dataset_dir, self.args.target_dir),
                f"{frame_num}_latent.npz"
            )
            np.savez_compressed(latent_path, latent=latent.numpy())
        except Exception as e:
            print(f"Worker {self.rank}: Error saving latent for {image_path}: {e}")
    
    def process_subjects(self, subjects):
        """Process a list of subjects"""
        for subject in subjects:
            subject_path = os.path.join(self.args.dataset_dir, subject)
            if not os.path.isdir(subject_path):
                continue
                
            target_subject_path = os.path.join(self.args.target_dir, subject)
            print(target_subject_path)
            if os.path.exists(target_subject_path) and not self.args.overwrite:
                print(f"Worker {self.rank}: Skipping {subject} - already exists")
                continue
                
            print(f"Worker {self.rank}: Processing subject {subject}")
            
            # Create directories
            os.makedirs(target_subject_path, exist_ok=True)
            target_images_dir = os.path.join(target_subject_path, "images_lr")
            os.makedirs(target_images_dir, exist_ok=True)
            
            # Get camera directories
            images_lr_path = os.path.join(subject_path, "images_lr")
            mask_lr_path = os.path.join(subject_path, "fmask_lr")
            if not os.path.exists(images_lr_path):
                print(f"Worker {self.rank}: No images_lr directory found in {subject_path}")
                continue
                
            image_camera_dirs = sorted([
                d for d in os.listdir(images_lr_path) 
                if os.path.isdir(os.path.join(images_lr_path, d)) and d.startswith('CC')
            ])

            mask_camera_dirs = sorted([
                d for d in os.listdir(mask_lr_path) 
                if os.path.isdir(os.path.join(mask_lr_path, d)) and d.startswith('CC')
            ])
            
            # Process each camera directory
            for image_camera_dir, mask_camera_dir in zip(image_camera_dirs, mask_camera_dirs):
                image_camera_path = os.path.join(images_lr_path, image_camera_dir)
                mask_camera_path = os.path.join(mask_lr_path, mask_camera_dir)
                target_image_camera_dir = os.path.join(target_images_dir, image_camera_dir)
                os.makedirs(target_image_camera_dir, exist_ok=True)
                
                print(f"Worker {self.rank}: Processing camera {image_camera_dir} in {subject}")
                self.process_camera_dir(image_camera_path, target_image_camera_dir, mask_camera_path)
                
            print(f"Worker {self.rank}: Finished subject {subject}")

def worker(rank, world_size, args, subject_chunks):
    """Worker process wrapper"""
    worker = VAEWorker(rank, world_size, args)
    worker.process_subjects(subject_chunks[rank])

def main():
    parser = argparse.ArgumentParser(description='Precompute latents from images')
    parser.add_argument('--dataset_dir', type=str, default="/workspace/datasetvol/mvhuman_data/mv_captures",
                       help='Dataset directory')
    parser.add_argument('--target_dir', type=str, default="/workspace/datasetvol/mvhuman_data/mv_latents",
                       help='Target directory for computed latents')
    parser.add_argument('--max_subjects', type=int, default=None,
                       help='Number of subjects to process')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing latents')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for encoding latents')
    parser.add_argument('--io_workers', type=int, default=16,
                       help='Number of I/O workers for loading images/masks')
    
    args = parser.parse_args()
    
    # Get list of subjects
    subjects = sorted([
        d for d in os.listdir(args.dataset_dir) 
        if os.path.isdir(os.path.join(args.dataset_dir, d))
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
        args=(world_size, args, subject_chunks),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()