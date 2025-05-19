import torch
from diffusers.models import AutoencoderKL  # type: ignore
from torch import nn
import lightning as L
from PIL import Image
import torchvision.transforms as transforms
import wandb

## NOTE: this is the SEVA AutoEncoder, used as a test

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
        self.chunk_size = chunk_size

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.module.encode(x).latent_dist.mean  # type: ignore
            * self.scale_factor
        )

    def encode(self, x: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._encode(x_chunk) for x_chunk in x.split(chunk_size)],
                dim=0,
            )
        else:
            return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.module.decode(z / self.scale_factor).sample  # type: ignore

    def decode(self, z: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        chunk_size = chunk_size or self.chunk_size
        if chunk_size is not None:
            return torch.cat(
                [self._decode(z_chunk) for z_chunk in z.split(chunk_size)],
                dim=0,
            )
        else:
            return self._decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class LightningAutoEncoder(L.LightningModule):
    def __init__(self, model: AutoEncoder, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model # AE
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    import argparse
    import os
    import wandb
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    # to image
    parser = argparse.ArgumentParser(description="Test autoencoder reconstruction")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="autoencoder-reconstruction")
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Initialize model and run inference
    model = AutoEncoder()
    lightning_model = LightningAutoEncoder(model)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model = lightning_model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Generate reconstruction
    with torch.no_grad():
        reconstructed = lightning_model(image_tensor)
    
    # Log tensor memory sizes
    def get_tensor_size(tensor):
        return tensor.element_size() * tensor.nelement()
    
    tensor_sizes = {
        "original": get_tensor_size(image_tensor),
        "reconstructed": get_tensor_size(reconstructed)
    }
    
    # Convert tensors back to images for visualization
    def tensor_to_image(tensor):
        # Denormalize and convert to PIL image
        tensor = tensor.cpu().squeeze(0)
        tensor = tensor * 0.5 + 0.5  # Denormalize
        tensor = torch.clamp(tensor, 0, 1)
        img = transforms.ToPILImage()(tensor)
        return img
    
    original_img = tensor_to_image(image_tensor)
    reconstructed_img = tensor_to_image(reconstructed)

    print(original_img.size)
    print(reconstructed_img.size)
    
    # Create a new image with double width
    combined_width = original_img.width * 2
    combined_height = original_img.height
    combined_img = Image.new('RGB', (combined_width, combined_height))
    
    # Paste images side by side
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(reconstructed_img, (original_img.width, 0))
    
    # Calculate memory sizes
    def get_image_size(img):
        # Convert to bytes and get size
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.tell()
    
    original_size = get_image_size(original_img)
    reconstructed_size = get_image_size(reconstructed_img)
    combined_size = get_image_size(combined_img)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({
            "original": wandb.Image(original_img, caption=f"Original (Tensor: {tensor_sizes['original']/1024:.1f}KB, File: {original_size/1024:.1f}KB)"),
            "reconstruction": wandb.Image(reconstructed_img, caption=f"Reconstructed (Tensor: {tensor_sizes['reconstructed']/1024:.1f}KB, File: {reconstructed_size/1024:.1f}KB)"),
            "reconstruction_comparison": wandb.Image(combined_img, caption=f"Original | Reconstructed (Combined: {combined_size/1024:.1f}KB)"),
        })
    
    print(f"Reconstruction complete. Results logged to wandb project 'autoencoder-reconstruction'")
    wandb.finish()

# example image path (000.png to 015.png of input GT images)
# /workspace/myvol/stable_cam_inference/seva_outputs/test123_img2img_16/demo/img2img/test123/input