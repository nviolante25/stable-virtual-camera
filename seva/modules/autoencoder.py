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
        # self.module.eval().requires_grad_(False)  # type: ignore
        self.module.train()
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


class LightningAutoEncoder(AbstractAutoencoder):
    def __init__(self, model: AutoEncoder, learning_rate: float = 1e-4):
        super().__init__()
        print("SEVA::LightningAutoEncoder::model:learning_rate ", learning_rate)
        print("SEVA::LightningAutoEncoder::model: ", model)
        self.model = instantiate_from_config(model)
        self.model.train()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.loss_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=2.0)
        # self.loss_fn = torch.nn.functional.mse_loss

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first
        # format (e.g., bchw instead if bhwc)
        # print("batch: ", batch)
        return batch[self.input_key]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.decode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.get_input(x) if isinstance(x, dict) else x)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = self.get_input(batch)
        x_hat = self(x)

        # print("Input shapes:")
        # print("x shape:", x.shape)
        # print("x_hat shape:", x_hat.shape)
        # print("Input ranges:")
        print("x range:", x.min().item(), "to", x.max().item())
        print("x_hat range:", x_hat.min().item(), "to", x_hat.max().item())
        # print("Input types:")
        # print("x dtype:", x.dtype)
        # print("x_hat dtype:", x_hat.dtype)

        # print("x device: ", x.device)
        # print("x_hat device: ", x_hat.device)

        # force x_hat to the range [-1, 1]
        x_hat = torch.clamp(x_hat, -1.0, 1.0)

        try:
            loss = self.loss_fn(x_hat, x)
            print("Loss computed successfully:", loss.item())
        except Exception as e:
            print("Error computing loss:", str(e))
            raise e

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = self.get_input(batch)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        # loss = torchmetrics.image.StructuralSimilarityIndexMeasure(
        #     data_range=1.0)(x_hat, x)
        # loss = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
        #     data_range=1.0)(x_hat, x)
        self.log("val_loss", loss)
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = self.get_input(batch)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        # loss = torchmetrics.image.StructuralSimilarityIndexMeasure(
        #     data_range=1.0)(x_hat, x)
        # loss = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
        #     data_range=1.0)(x_hat, x)
        self.log("val_loss", loss)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def log_images(
        self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs
    ) -> dict:
        log = dict()
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        print("SEVA::LightningAutoEncoder::log_images::x: ", x)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

        _, xrec, _ = self(x, **additional_decode_kwargs)
        log["inputs"] = x
        log["reconstructions"] = xrec
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost shows location of small errors, by boosting their
        # brightness.
        log["diff_boost"] = (
            2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        )
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, xrec))
        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            log["reconstructions_ema"] = xrec_ema
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            log["diff_boost_ema"] = (
                2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
            )
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log[log_str] = xrec_add
        return log
