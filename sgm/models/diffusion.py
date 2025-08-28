import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
import torchvision.transforms.v2 as T
from einops import repeat

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img)
from seva.sampling import MultiviewCFG
import gc

def compute_psnr(pred, target):
    mse = torch.nn.functional.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / mse)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        verbose_lora_deltas: bool = False
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.verbose_lora_deltas = verbose_lora_deltas

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if hasattr(self.first_stage_model, "decoder") and \
                    isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def _encode_inconsistent_images(
        self,
        paths: List[str],
        ref_mask: torch.Tensor,
        clean_latent: torch.Tensor,
        rel_bbox: torch.Tensor,
    ) -> torch.Tensor:

        # TODO: optimize such that we only encode images that are needed using ref_mask
        # load images from paths and convert to tensors
        latents = []
        rgb_images = []
        num_images = len(paths)
        normalizer = T.Normalize([0.5], [0.5])
        with torch.no_grad():
            for i, batch in enumerate(list(zip(*paths))): # assumes these are already (576,576)
                batch_images = []
                for j, path in enumerate(batch): # for each image in the batch
                    img = Image.open(path).convert('RGB')
                    # Apply same transforms as your dataloader
                    transform = T.Compose([
                        T.Resize((576, 576)),
                        T.ToTensor(),
                        # T.Normalize([0.5], [0.5])  # Scale to [-1, 1]
                    ])
                    dx1, dy1, dx2, dy2 = (rel_bbox[i, j]).int() # ! coordinates based on 576^2
                    img_tensor = transform(img)
                    _, H, W = img_tensor.shape
                    img_tensor = img_tensor[:, 0+dy1:H+dy2, 0+dx1:W+dx2]
                    batch_images.append(normalizer(transform(img_tensor))) # resize back to 576^2
                batch_tensor = torch.stack(batch_images).to(self.device)
                rgb_images.append(batch_tensor.to(self.device)) # leave normalized
                latents.append(self.encode_first_stage(batch_tensor))
            latents = torch.cat(latents, dim=0)
            latents = latents.reshape(-1, num_images, 4, 72, 72) # ! HARDCODED
            # replace latents with clean latents for ref images
            latents[ref_mask] = clean_latent[ref_mask]
            rgb_images = torch.cat(rgb_images, dim=0).reshape(-1, num_images, 3, 576, 576)
        return latents, rgb_images

    def shared_step(self, batch: Dict) -> Any: 
        x = self.get_input(batch)
        if len(x.shape) == 1: # latents are NOT computed yet
            x = batch["frames"].to(self.device)
            batch_latents = []
            for b in x:
                batch_latents.append(self.encode_first_stage(b))
            x = torch.stack(batch_latents, dim=0)
            batch["clean_latent"] = x
        else: #latents already precomputed (same as "IdentityEncoder")
            batch["clean_latent"] = x * self.scale_factor # need to scale!

        # encode ic latents from the paths (scales)
        if torch.any(batch["use_inconsistent"]).item():
            ic, _ = self._encode_inconsistent_images(batch["ic_paths"], batch["ref_mask"], batch["clean_latent"], batch["ic_bbox"])
            # for target (not input/ref) frames, zero condition latents
            ic[~batch["mask"]] = 0
        else:
            # no conditioning (to be replaced by clean_latents for inputs)
            ic = torch.zeros_like(batch["clean_latent"], device=self.device)
            ic[batch["ref_mask"]] = batch["clean_latent"][batch["ref_mask"]]

        # ensure for ref image, ic tensors should be replaced by clean latents 
        # add ic as conditioning in concat (along with clean + plucker + masks)
        batch.update({
            "replace": torch.cat([
                batch["clean_latent"],
                repeat(
                    batch["ref_mask"],
                    "b n -> b n 1 h w",
                    h=batch["plucker"].shape[-2],
                    w=batch["plucker"].shape[-1]
                )
            ], dim=2),
            "concat": torch.cat([batch["concat"], ic], dim=2)
        }) # concat to be (B, T, 6(plucker) + 2(masks) + 4(ic))

        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        # print("\nDiffusionEngine::training_step batch:\n", batch)
        loss, loss_dict = self.shared_step(batch)

        # Debug: Check LoRA gradients after backward
        # if batch_idx % 10 == 0 and self.verbose_lora_deltas:  # Check every 10 batches
        #     lora_grads = []
        #     for name, param in self.model.named_parameters():
        #         if 'lora' in name and param.grad is not None:
        #             grad_norm = param.grad.norm().item()
        #             lora_grads.append((name, grad_norm))
            
        #     if lora_grads:
        #         print(f"Batch {batch_idx} - LoRA gradients found:")
        #         for name, grad_norm in lora_grads[:3]:  # Show first 3
        #             print(f"  {name}: grad_norm = {grad_norm:.6f}")
        #     else:
        #         print(f"Batch {batch_idx} - NO LoRA gradients found!")

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: add this in in place of training image logs (not tested yet)
        loss, loss_dict = self.shared_step(batch)
        # log averaged validation loss; keep per-step metrics off
        val_dict = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(val_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Add this debug print
        print(f"Logger save_dir: {self.logger.save_dir}")
        
        # Get ground truth images
        x = self.get_input(batch)
        
        # Generate samples
        c, uc = self.conditioner.get_unconditional_conditioning(batch)
        samples = self.sample(c, uc=uc, batch_size=x.shape[0], shape=self.encode_first_stage(x).shape[1:])
        samples = self.decode_first_stage(samples)
        
        # Compute metrics
        metrics = {
            'mse': torch.nn.functional.mse_loss(samples, x),
            'psnr': compute_psnr(samples, x),
            # 'ssim': self.compute_ssim(samples, x)
        }
        
        # Log metrics
        self.log_dict(metrics, prog_bar=True, logger=True)
        
        # Log images for visualization
        if batch_idx == 0:
            # Add debug prints
            print("Attempting to log images...")
            images = self.log_images(batch, N=min(8, x.shape[0]), sample=True)
            print(images["reconstructions"].shape)
            print("max: ", images["reconstructions"].max())
            print("min: ", images["reconstructions"].min())
            print(f"Generated image keys: {list(images.keys())}")
        
        return metrics

    def on_train_epoch_end(self, *args, **kwargs):
        # clear multiple GPU cache 
        torch.cuda.empty_cache()
        gc.collect()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")
        
        # Store initial LoRA parameter values for tracking changes
        self.initial_lora_params = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                self.initial_lora_params[name] = param.data.clone()
        print(f"Stored initial values for {len(self.initial_lora_params)} LoRA parameters")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
        
        # # Check LoRA parameter changes every 100 batches
        # if hasattr(self, 'initial_lora_params') and self.global_step % 100 == 0 and self.verbose_lora_deltas:
        #     print(f"\n=== Global Step {self.global_step} - LoRA Parameter Changes ===")
        #     total_change = 0
        #     for name, param in self.model.named_parameters():
        #         if 'lora' in name and name in self.initial_lora_params:
        #             change = (param.data - self.initial_lora_params[name]).abs().mean().item()
        #             total_change += change
        #             if change > 1e-6:  # Only show significant changes
        #                 print(f"  {name}: mean_change = {change:.8f}")
            
        #     if total_change < 1e-6:
        #         print("  WARNING: No significant LoRA parameter changes detected!")
        #     print(f"  Total mean change: {total_change:.8f}")
        #     print("=" * 60)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        # params = list(self.model.parameters())

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params = params + [param]

        # Write trainable embedders to file
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config) # AdamW
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config) # LambdaLinearScheduler
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)
        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        if isinstance(self.sampler.guider, MultiviewCFG): # or anything that accepts kwargs
            samples = self.sampler(denoiser, randn, scale=kwargs.get("scale", 2.0), cond=cond, uc=uc, **kwargs)
        else:
            samples = self.sampler(denoiser, randn, scale=kwargs.get("scale", 2.0), cond=cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        # [batch_size, num_images, channels, height, width]
        input_tensor = batch[self.input_key]
        image_h, image_w = input_tensor.shape[-2:]
            
        print(f"Input tensor shape: {input_tensor.shape}, extracting H={image_h}, W={image_w}")
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to stringa
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        # no longer using this
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders] # plucker, concat, replace, mask, None
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch) # clean_latent

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        # log["inputs"] = x
        z = self.encode_first_stage(x) # identity
        reconstructions = self.decode_first_stage(z)
        gt_images = batch["frames"]
        log["inputs"] = gt_images[:N]

        # Handle SEVA multi-view reconstructions: reshape [batch_size*num_images, C, H, W] -> [batch_size, num_images, C, H, W]
        if reconstructions.dim() == 4 and reconstructions.shape[0] == x.shape[0] * x.shape[1]:
            # This is a SEVA multi-view output
            batch_size = x.shape[0]
            num_images = x.shape[1]
            reconstructions = reconstructions.view(batch_size, num_images, *reconstructions.shape[1:])
            log["reconstructions"] = reconstructions
        else:
            log["reconstructions"] = reconstructions
            
        # log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        sampling_kwargs = {}
        if isinstance(self.sampler, MultiviewCFG):
            sampling_kwargs["c2w"] = batch.get("c2w", None)
            sampling_kwargs["K"] = batch.get("K", None)
            sampling_kwargs["input_frame_mask"] = batch.get("input_frame_mask", None)
            sampling_kwargs["scale"] = kwargs.get("scale", 2.0)

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            
            # Handle SEVA multi-view outputs: reshape [batch_size*num_images, C, H, W] -> [batch_size, num_images, C, H, W]
            if samples.dim() == 4 and samples.shape[0] == x.shape[0] * x.shape[1]:
                # This is a SEVA multi-view output
                batch_size = x.shape[0]
                num_images = x.shape[1]
                samples = samples.view(batch_size, num_images, *samples.shape[1:])
                log["samples"] = samples
            else:
                log["samples"] = samples
        print("log_images end, keys: ", log.keys())
        return log
