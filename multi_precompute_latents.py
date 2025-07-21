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
import tqdm

# Set up multiprocessing with spawn method
mp.set_start_method('spawn', force=True)

# Constants for image processing
IMAGE_SIZE = 576
CENTER_CROP_SIZE = 1500

# Model configuration
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
MODEL_SUBFOLDER = "vae"
LOCAL_MODEL_PATH = "./local_vae_model"

def download_model_locally():
    """Download the VAE model from HuggingFace and save it locally"""
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"Model already exists at {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
    
    print(f"Downloading VAE model from {MODEL_ID}...")
    try:
        model = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder=MODEL_SUBFOLDER,
            force_download=True,
            low_cpu_mem_usage=False,
        )
        
        # Save the model locally
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model.save_pretrained(LOCAL_MODEL_PATH)
        print(f"Model saved to {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

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
    # transforms.Normalize([0.5], [0.5])
])

# # Define mask preprocessing (same as image but without normalization)
# mask_preprocess = transforms.Compose([
#     transforms.Lambda(lambda img: img.crop((
#         (img.width - CENTER_CROP_SIZE) // 2,  # left
#         (img.height - CENTER_CROP_SIZE) // 2,  # top
#         (img.width + CENTER_CROP_SIZE) // 2,   # right
#         (img.height + CENTER_CROP_SIZE) // 2   # bottom
#     ))),
#     transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
#     transforms.ToTensor(),
# ])

class VAEWorker:
    def __init__(self, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = f"cuda:{rank}"
        
        # Initialize model on this GPU from local path
        print(f"Worker {self.rank}: Loading VAE model from local path...")
        self.model = AutoencoderKL.from_pretrained(
            LOCAL_MODEL_PATH,
            force_download=False,
            low_cpu_mem_usage=False,
        ).to(self.device)
        self.model.eval()
        print(f"Worker {self.rank}: VAE model loaded successfully")

        # Initialize I/O executor for prefetching 
        self.prefetch_exec = ThreadPoolExecutor(max_workers=1)
        
        # Warm up the model
        self._warmup_model()
    
    def _warmup_model(self):
        """Warm up the model with a dummy input"""
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)
        with torch.no_grad():
            _ = self.model.encode(dummy_input).latent_dist.sample()
        torch.cuda.synchronize()

    def shutdown_worker_executors(self):
        """Shutdown ThreadPoolExecutors when finished."""
        print(f"Worker {self.rank}: Shutting down I/O executor...")
        self.prefetch_exec.shutdown(wait=True)
        print(f"Worker {self.rank}: I/O executor shut down.")

    
    def process_batch(self, batch_images):
        """Process a batch of images and return latents"""
        with torch.no_grad():
            # batch_tensor = torch.stack(batch_images).to(self.device, non_blocking=True)
            batch_tensor = batch_images.to(self.device, non_blocking=True)
            latents = self.model.encode(batch_tensor).latent_dist.sample()
            return latents.detach().cpu()
    
    def _load_and_preprocess_batch(self, image_paths: List[str], mask_paths: List[str]) -> List[torch.Tensor]:
        """Load and preprocess a batch of images with their masks using vectorized operations"""
        
        # Load all images and masks in parallel
        with ThreadPoolExecutor(max_workers=self.args.io_workers) as executor:
            # Submit all image and mask loading tasks concurrently
            image_futures = [executor.submit(self._load_image, path) for path in image_paths]
            mask_futures =  [executor.submit(self._load_mask,  path) for path in mask_paths]
            
            # Collect results as they complete
            images = [future.result() for future in image_futures]
            masks = [future.result() for future in mask_futures]

        assert len(images) == len(mask_paths)

        # Convert to tensors and stack for vectorized operations
        # Use pin_memory=True for faster CPU->GPU transfer
        images_tensors = torch.stack(images).pin_memory()
        masks_tensors = torch.stack(masks).pin_memory()

        # print("in load and preprocess batch")
        # print("images:", images_tensors.shape)
        # print("masks:", masks_tensors.shape)
        
        
        # Vectorized masking operation with optimized broadcasting
        # Expand mask from (B, 1, H, W) to (B, 3, H, W) to match image channels
        # Use repeat instead of expand for better memory efficiency
        # masks_tensors = masks_tensors.repeat(1, 3, 1, 1)
        masked_images = images_tensors * masks_tensors
        normalize = transforms.Normalize([0.5], [0.5])
        masked_images = normalize(masked_images)

        # # Save test images for visualization
        # if len(masked_images) > 0:
        #     test_img = images_tensors[0]
        #     test_mask = masks_tensors[0, 0:1]
        #     test_masked = masked_images[0]
            
        #     # Convert tensors to PIL images
        #     to_pil = transforms.ToPILImage()
            
        #     # Create test output directory
        #     test_dir = "test_masked_outputs"
        #     os.makedirs(test_dir, exist_ok=True)
            
        #     # Generate unique filename based on timestamp
        #     timestamp = int(time.time() * 1000)
        #     base_name = f"test_{timestamp}"
            
        #     # Save original image
        #     test_img_pil = to_pil(test_img)
        #     test_img_pil.save(os.path.join(test_dir, f"{base_name}_original.png"))
            
        #     # Save mask 
        #     test_mask_pil = to_pil(test_mask)
        #     test_mask_pil.save(os.path.join(test_dir, f"{base_name}_mask.png"))
            
        #     # Save masked result
        #     test_masked_pil = to_pil(test_masked)
        #     test_masked_pil.save(os.path.join(test_dir, f"{base_name}_masked.png"))
        
        # Convert back to list for compatibility
        return masked_images
    
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
                    mask = mask.convert(mode="L")
                return preprocess(mask)
        except Exception as e:
            print(f"Worker {self.rank}: Error loading mask {mask_path}: {e}")
            return None
    
    # def _load_and_preprocess(self, image_path, mask_path, save_test_image=False):
    #     """Legacy method - kept for compatibility but not used in optimized version"""
    #     try:
    #         # Load image
    #         with Image.open(image_path) as img:
    #             img = img.convert("RGB")
    #             img_tensor = preprocess(img)
            
    #         # Load mask
    #         with Image.open(mask_path) as mask:
    #             mask = mask.convert("L")  # Convert to grayscale
    #             # Use the same preprocessing pipeline as images (center crop + resize)
    #             mask_tensor = mask_preprocess(mask)
            
    #         # Apply mask (mask is 0-1, where 1 is foreground)
    #         # For VAE encoding, we typically want to mask out background (set to 0)
    #         # So we multiply the image by the mask
    #         masked_image = img_tensor * mask_tensor
            
    #         # Save test image if requested
    #         if save_test_image:
    #             self._save_test_masked_image(image_path, img_tensor, mask_tensor, masked_image)
            
    #         return masked_image
            
    #     except Exception as e:
    #         print(f"Worker {self.rank}: Error processing image {image_path}: {e}")
    #         return None
    
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
                # print(tensor.shape)
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
    
    def process_camera_dir(self, 
        camera_path,
        mask_camera_path,
    ):
        """Process all images in a camera directory with optimized batch processing and prefetching"""
        # print("inside process_camera_dir")
        # print("camera_path:", camera_path)
        target_latent_dir = camera_path.replace("mv_captures", "mv_latents").split("/images_lr")[0]
        # print("target_latent_dir:", target_latent_dir)
        # GOAL: fill up this dict with all images inside camera
        camera_latents_dict = {}

        image_paths = []
        mask_paths  = []

        img_camera_paths = sorted(
            [name for name in os.listdir(camera_path) 
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.basename(x).split('_')[0]))

        mask_camera_paths = sorted(
            [name for name in os.listdir(mask_camera_path) 
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.basename(x).split('_')[0]))

        for image_path, mask_path in zip(img_camera_paths, mask_camera_paths):
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')) and mask_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(camera_path, image_path))
                mask_paths.append(os.path.join(mask_camera_path, mask_path))
                assert image_path.split('_')[0] == mask_path.split('_')[0] # timestep must match, otherwise skip (in process_subjects)        
        
        # Prefetch next batch while processing current batch
        def prefetch_batch(batch_image_paths, batch_mask_paths):
            """Prefetch the next batch of images and masks"""
            return self._load_and_preprocess_batch(batch_image_paths, batch_mask_paths)

        current_batch_future = None
        assert len(image_paths) == len(mask_paths) # if not synced, skip. (Aggressive, but can be changed later.)
        if len(image_paths) > 0:
            init_image_paths = image_paths[0:self.args.batch_size]
            init_mask_paths  =  mask_paths[0:self.args.batch_size]
            current_batch_future = self.prefetch_exec.submit(
                prefetch_batch, init_image_paths, init_mask_paths
            )

        # Process in batches with prefetching
        for i in range(0, len(image_paths), self.args.batch_size):
            # retrieve current batch
            if current_batch_future:
                batch_images = current_batch_future.result() # these are MASKED
                # print("batch_images:", batch_images.shape)
            else:
                batch_images = []

            if batch_images is None:
                print(f"Worker {self.rank}: No valid images in batch starting at index {i}. Skipping.")
                current_batch_future = None # reset
                continue

            # prefetch next batch
            if i + self.args.batch_size < len(image_paths):
                next_batch_image_paths = image_paths[i + self.args.batch_size:i + 2 * self.args.batch_size]
                next_batch_mask_paths  =  mask_paths[i + self.args.batch_size:i + 2 * self.args.batch_size]
                current_batch_future = self.prefetch_exec.submit(
                    prefetch_batch, next_batch_image_paths, next_batch_mask_paths
                )
            else:
                current_batch_future = None

            try:
                # Process batch
                start_time = time.time()
                latents = self.process_batch(batch_images)
                
                # OLD -- Save latents in parallel
                # with ThreadPoolExecutor(max_workers=self.args.save_workers) as executor:
                #     executor.map(self._save_latent, batch_image_paths, latents)

                # NEW -- Store latents in dict, then save after all subjects processed
                try: 
                    j = 0
                    for j in range(len(latents)):
                        image_path = image_paths[i + j] # inter-batch idx 'i' + infra-batch idx 'j'
                        cam_id = os.path.basename(os.path.dirname(image_path))
                        timestep = os.path.basename(image_path).split('_')[0]
                        # print("cam_id:", cam_id)
                        # print("timestep:", timestep)
                        # print("latent shape:", latents[j].shape)
                        key = f"{cam_id}.{timestep}"
                        camera_latents_dict[key] = latents[j].numpy()
                except Exception as e:
                    j = j - 1 # adjust behavior to accurately keep track of batch size
                    print(f"Worker {self.rank}: Last batch processed.")

                # Log performance
                batch_time = time.time() - start_time
                print(f"Worker {self.rank}: Processed {min(j + 1, len(batch_images))} images in {batch_time:.2f}s "
                      f"({len(batch_images)/batch_time:.2f} img/s)")
                
                # Clean up
                del latents, batch_images
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Worker {self.rank}: Error processing batch: {e}")
                continue

        return camera_latents_dict
    
    def _save_latent(self, image_path, latent):
        """DEPRECATED -- Save a single latent tensor"""
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
            subject_latents_dict = {}
            subject_path = os.path.join(self.args.dataset_dir, subject)
            if not os.path.isdir(subject_path):
                continue
                
            # ex. output target structure as such: ...mv_latents/100001/ 
            target_subject_path = os.path.join(self.args.target_dir, subject)
            target_subject_npz_path = os.path.join(target_subject_path, f"{subject}.npz")

            if os.path.exists(target_subject_npz_path) and not self.args.overwrite:
                print(f"Worker {self.rank}: Skipping {subject} - already exists")
                continue
                
            # Create temporary file to indicate processing is in progress
            # tmp_file = os.path.join(target_subject_path, ".processing")
            # if os.path.exists(tmp_file):
            #     print(f"Worker {self.rank}: Skipping {subject} - being processed by another worker. ")
            #     continue

            # Create directory FIRST, then create the processing file
            os.makedirs(target_subject_path, exist_ok=True)
            
            # with open(tmp_file, 'w') as f:
            #     f.write(str(os.getpid()))

            print(f"Worker {self.rank}: Processing subject {subject}")
            
            # target_images_dir = os.path.join(target_subject_path, "images_lr")
            # os.makedirs(target_images_dir, exist_ok=True)
            
            # Get camera directories (images & masks)
            images_lr_path = os.path.join(subject_path, "images_lr")
            mask_lr_path = os.path.join(subject_path, "fmask_lr")
            if not os.path.exists(images_lr_path):
                print(f"Worker {self.rank}: No images_lr directory found in {subject_path}")
                continue
                
            if not os.path.exists(mask_lr_path):
                print(f"Worker {self.rank}: No fmask_lr directory found in {subject_path}")
                continue
                
            # Sort cameras
            image_camera_dirs = sorted([
                d for d in os.listdir(images_lr_path) 
                if os.path.isdir(os.path.join(images_lr_path, d)) and d.startswith('CC')
            ])

            mask_camera_dirs = sorted([
                d for d in os.listdir(mask_lr_path) 
                if os.path.isdir(os.path.join(mask_lr_path, d)) and d.startswith('CC')
            ])
            
            # Process each camera directory
            # NOTE: it seems at least one file in a subject is missing and can destroy the entire process
            # çurrently, just skips over the subject if this occurs, and writes its name into a file.
            try:
                for image_camera_dir, mask_camera_dir in zip(image_camera_dirs, mask_camera_dirs):
                    assert image_camera_dir == mask_camera_dir # if cameras not matching, skip.
                    image_camera_path = os.path.join(images_lr_path, image_camera_dir)
                    mask_camera_path = os.path.join(mask_lr_path, mask_camera_dir)
                    # target_image_camera_dir = os.path.join(target_subject_path, image_camera_dir)
                    # print("target_image_camera_dir:", target_image_camera_dir)
                    # os.makedirs(target_image_camera_dir, exist_ok=True)
                    
                    print(f"Worker {self.rank}: Processing camera {image_camera_dir} in {subject}")
                    camera_latents_dict = self.process_camera_dir(image_camera_path, mask_camera_path)
                    subject_latents_dict.update(camera_latents_dict)
            except Exception as e:
                print(f"Worker {self.rank}: Error processing subject {subject}: {e}")
                # write subject name to file for later processing
                with open(os.path.join(target_subject_path, "missing_files.txt"), "a") as f:
                    f.write(f"{subject}: {e}\n")
                continue
                
            # save all latents to singular {subject}.npz file
            try:
                if subject_latents_dict:
                    np.savez_compressed(os.path.join(target_subject_path, f"{subject}.npz"), **subject_latents_dict)
                    # Remove .processing indicator file safely
                    # try:
                    #     # os.remove(tmp_file)
                    # except FileNotFoundError:
                    #     pass  # File was already removed
                    del subject_latents_dict
                    print(f"Worker {self.rank}: Finished subject {subject}.")
                else:
                    print(f"Worker {self.rank}: No latents processed for subject {subject}")
            except Exception as e:
                # Remove .processing indicator file safely on error
                # try:
                #     # os.remove(tmp_file)
                # except FileNotFoundError:
                    # pass  # File was already removed
                print(f"Worker {self.rank}: Error saving subject NPZ for {subject}: {e}")

        print("Finished processing all subjects.")


def worker(rank, world_size, args, subject_chunks):
    """Worker process wrapper"""
    worker = VAEWorker(rank, world_size, args)
    try: 
        worker.process_subjects(subject_chunks[rank])
    finally:
        worker.shutdown_worker_executors()


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
    parser.add_argument('--io_workers', type=int, default=8,
                       help='Number of I/O workers for loading images/masks')
    parser.add_argument('--save_workers', type=int, default=1,
                       help='Number of workers for saving latents (total threads = io_workers + save_workers)')
    parser.add_argument('--download_model', action='store_true',
                       help='Download model from HuggingFace before processing')
    parser.add_argument('--model_path', type=str, default="./local_vae_model",
                       help='Path to local model directory')
    
    args = parser.parse_args()
    
    # Update global model path if specified
    global LOCAL_MODEL_PATH
    LOCAL_MODEL_PATH = args.model_path
    
    # Download model if requested or if it doesn't exist
    if args.download_model or not os.path.exists(LOCAL_MODEL_PATH):
        print("Model download requested or model not found locally")
        download_model_locally()
    else:
        print(f"Using existing model at {LOCAL_MODEL_PATH}")
    
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

    print(subject_chunks)
    
    # Launch processes
    mp.spawn(
        worker,
        args=(world_size, args, subject_chunks),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()