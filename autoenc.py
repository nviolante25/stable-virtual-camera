import torch
from diffusers.models import AutoencoderKL  # type: ignore
from torch import nn
import lightning as L
from PIL import Image
import torchvision.transforms as transforms
import wandb
import numpy as np

import torch
from diffusers.models import AutoencoderKL  # type: ignore
from torch import nn
import pytorch_lightning as pl
from sgm.util import instantiate_from_config
from sgm.models.autoencoder import AbstractAutoencoder
from typing import Dict, Optional
import torchmetrics

class AutoEncoder(nn.Module):
    scale_factor: float = 0.18215
    downsample: int = 8

    def __init__(self, chunk_size: int | None = None):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="vae",
            force_download=False,
            low_cpu_mem_usage=False,
        )
        self.module.eval().requires_grad_(False)  # type: ignore
        # self.module.train()
        self.chunk_size = chunk_size

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # print("SEVA VAE::_encode")
        # print("data input shape: ", x.shape)
        # print("encoder conv_in weight shape: ", self.module.encoder.conv_in.weight.shape)
        return (
            self.module.encode(x).latent_dist.mean  # type: ignore
            * self.scale_factor
        )

    def encode(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        # print("SEVA VAE::encode")
        if chunk_size is not None:
            # print("SEVA VAE::encode::chunk_size: ", chunk_size)
            return torch.cat(
                [self._encode(x_chunk) for x_chunk in x.split(chunk_size)],
                dim=0,
            )
        else:
            # print("SEVA VAE::encode:: ELSE STMT ")
            return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.module.decode(z / self.scale_factor).sample  # type: ignore

    def decode(self, z: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        # print("SEVA VAE::decode::z: ", z)
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._decode(z_chunk) for z_chunk in z.split(chunk_size)],
                dim=0,
            )
        else:
            return self._decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # this is indeed a tensor
        return self.decode(self.encode(x))

## NOTE: this is the SEVA AutoEncoder, used as a test

if __name__ == "__main__":
    import argparse
    import os
    import wandb
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    # to image
    parser = argparse.ArgumentParser(description="Test autoencoder reconstruction")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--latent_path", type=str, help="Path to the latent encoding")
    parser.add_argument("--npzkey", type=str, help="Key to the latent encoding in the npz file")

    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="autoencoder-reconstruction")
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((576, 576)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    elif args.latent_path is not None:
        latent_tensor = torch.from_numpy(np.load(args.latent_path)[args.npzkey])
        input_tensor = latent_tensor.unsqueeze(0)
    else:
        raise ValueError("Either --image_path or --latent_path must be provided")
    
    # Initialize model and run inference
    model = AutoEncoder()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Generate reconstruction
    with torch.no_grad():
        if args.latent_path is not None:
            latents = input_tensor * 0.18215 # scale the latents
            reconstructed = model.decode(latents)
        else:
            latents = model.encode(input_tensor)
            reconstructed = model.decode(latents)
    
    # Log tensor memory sizes
    def get_tensor_size(tensor):
        return tensor.element_size() * tensor.nelement()
    
    tensor_sizes = {
        "latents": get_tensor_size(latents),
        "reconstructed": get_tensor_size(reconstructed)
    }
    if args.image_path is not None:
        tensor_sizes["original"] = get_tensor_size(input_tensor)
    
    # Convert tensors back to images for visualization
    def tensor_to_image(tensor):
        # Denormalize and convert to PIL image
        tensor = tensor.cpu().squeeze(0)
        tensor = tensor * 0.5 + 0.5  # Denormalize
        tensor = torch.clamp(tensor, 0, 1)
        img = transforms.ToPILImage()(tensor)
        return img
    
    reconstructed_img = tensor_to_image(reconstructed)
    print("latents min, max: ", latents.min(), latents.max())
    print("reconstructed min, max: ", reconstructed.min(), reconstructed.max())
    print(reconstructed_img.size)
    
    if args.image_path is not None:
        original_img = tensor_to_image(input_tensor)
        print(original_img.size)
        # Create a new image with double width for side-by-side comparison
        combined_width = original_img.width * 2
        combined_height = original_img.height
        combined_img = Image.new('RGB', (combined_width, combined_height))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(reconstructed_img, (original_img.width, 0))
    
    # Calculate memory sizes
    def get_image_size(img):
        # Convert to bytes and get size
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.tell()
    
    reconstructed_size = get_image_size(reconstructed_img)
    
    # Log to wandb
    if wandb.run is not None:
        log_dict = {
            "reconstruction": wandb.Image(reconstructed_img, caption=f"Reconstructed (Tensor: {tensor_sizes['reconstructed']/1024:.1f}KB, File: {reconstructed_size/1024:.1f}KB)")
        }
        
        if args.image_path is not None:
            original_size = get_image_size(original_img)
            combined_size = get_image_size(combined_img)
            log_dict.update({
                "original": wandb.Image(original_img, caption=f"Original (Tensor: {tensor_sizes['original']/1024:.1f}KB, File: {original_size/1024:.1f}KB)"),
                "reconstruction_comparison": wandb.Image(combined_img, caption=f"Original | Reconstructed (Combined: {combined_size/1024:.1f}KB)")
            })
            
        wandb.log(log_dict)
    
    print(f"Reconstruction complete. Results logged to wandb project 'autoencoder-reconstruction'")
    wandb.finish()

# example image path (000.png to 015.png of input GT images)
# /workspace/myvol/stable_cam_inference/seva_outputs/test123_img2img_16/demo/img2img/test123/input