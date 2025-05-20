import torch
from diffusers.models import AutoencoderKL  # type: ignore
from torch import nn
import pytorch_lightning as pl

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
        # self.module.eval().requires_grad_(False)  # type: ignore
        self.module.train()
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


class LightningAutoEncoder(pl.LightningModule):
    def __init__(self, model: AutoEncoder, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch
        x_hat = self(x)
        # Reconstruction loss
        loss = nn.functional.mse_loss(x_hat, x)
        # Optional: Add KL divergence loss if you want to maintain the prior
        # kl_loss = self.model.module.kl_loss()
        # loss = loss + 0.1 * kl_loss
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)