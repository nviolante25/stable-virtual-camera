import argparse
import datetime
import glob
import inspect
import os
import sys
from inspect import Parameter
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import wandb
from matplotlib import pyplot as plt
from natsort import natsorted
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from diffusers import AutoencoderKL
from sgm.util import exists, instantiate_from_config, isheatmap

import threading
import queue
from typing import Dict
from dataclasses import dataclass

MULTINODE_HACKS = True


def default_trainer_args():
    argspec = dict(inspect.signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {
        param: argspec[param].default
        for param in argspec
        if argspec[param] != Parameter.empty
    }
    return default_args


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--no_date",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="if True, skip date generation for logdir and only use naming via opt.base or opt.name (+ opt.postfix, optionally)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--projectname",
        type=str,
        default="stablediffusion",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--legacy_naming",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="name run based on config file name if true, else by whole path",
    )
    parser.add_argument(
        "--enable_tf32",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",
    )
    parser.add_argument(
        "--startup",
        type=str,
        default=None,
        help="Startuptime from distributed script",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,  # TODO: later default to True
        help="log to wandb",
    )
    parser.add_argument(
        "--no_base_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,  # TODO: later default to True
        help="log to wandb",
    )
    if version.parse(torch.__version__) >= version.parse("2.0.0"):
        parser.add_argument(
            "--resume_from_checkpoint",
            type=str,
            default=None,
            help="single checkpoint file to resume from",
        )
    default_args = default_trainer_args()
    for key in default_args:
        parser.add_argument("--" + key, default=default_args[key])
    return parser


def get_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "checkpoints", "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    
    if len(ckpt) > 1:
        print("got most recent checkpoint")
        ckpt = sorted(ckpt, key=lambda x: os.path.getmtime(x))[-1]
        print(f"Most recent ckpt is {ckpt}")
        with open(os.path.join(logdir, "most_recent_ckpt.txt"), "w") as f:
            f.write(ckpt + "\n")
        try:
            version = int(ckpt.split("/")[-1].split("-v")[-1].split(".")[0])
        except Exception as e:
            print("version confusion but not bad")
            print(e)
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt = ckpt[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    print(f"Current melk ckpt name: {melk_ckpt_name}")
    return ckpt, melk_ckpt_name


class SetupCallback(Callback):
    def __init__(
        self,
        resume,
        now,
        logdir,
        ckptdir,
        cfgdir,
        config,
        lightning_config,
        debug,
        ckpt_name=None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: pl.Trainer, pl_module, exception):
        if not self.debug and trainer.global_rank == 0:
            print("Summoning checkpoint.")
            if self.ckpt_name is None:
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            else:
                ckpt_path = os.path.join(self.ckptdir, self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainsteps_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


@dataclass
class LogTask:
    """Data class to hold logging task information"""
    save_dir: str
    split: str
    images: Dict[str, torch.Tensor]
    masks: torch.Tensor
    global_step: int
    current_epoch: int
    batch_idx: int
    pl_module: pl.LightningModule
    scale_factor: float

class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

        # for logging
        with torch.device("cpu"):
            self.cpu_decoder = AutoencoderKL.from_pretrained(
                    "stabilityai/stable-diffusion-2-1-base",
                    subfolder="vae",
                    force_download=False,
                    low_cpu_mem_usage=False,
                ).eval()
        self.cpu_decoder.requires_grad_(False)
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.shutdown_event = threading.Event()
        self._start_log_thread()

    def _start_log_thread(self):
        """Start the logging thread"""
        if self.log_thread is None:
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()

    def _log_worker(self):
        """Background worker that processes logging tasks from the queue"""
        print("[ImageLogger] Log worker started")
        
        # Initialize CPU decoder for the worker thread
        # with torch.device("cpu"):
        #     self.cpu_decoder = AutoencoderKL.from_pretrained(
        #         "stabilityai/stable-diffusion-2-1-base",
        #         subfolder="vae",
        #         force_download=False,
        #         low_cpu_mem_usage=False,
        #     ).eval()
        #     self.cpu_decoder.requires_grad_(False)
            
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for a task with timeout to allow checking shutdown event
                task = self.log_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                print(f"[ImageLogger] Processing log task for step {task.global_step}")
                self._process_log_task(task)
                self.log_queue.task_done()
            except queue.Empty:
                continue # polling
            except Exception as e:
                print(f"[ImageLogger] Error in log worker: {e}")
                if task is not None:
                    self.log_queue.task_done()
        
        # Process any remaining tasks before exiting
        print("[ImageLogger] Processing remaining tasks in queue...")
        while True:
            try:
                task = self.log_queue.get_nowait()
                if task is None:
                    break
                print(f"[ImageLogger] Processing final log task for step {task.global_step}")
                self._process_log_task(task)
                self.log_queue.task_done()
            except queue.Empty:
                break
        
        print("[ImageLogger] Log worker stopped")

    def _process_log_task(self, task: LogTask):
        """Process a single logging task"""
        try:
            # Move data to CPU and decode
            images = {}
            for k in task.images:
                if isinstance(task.images[k], torch.Tensor):
                    images[k] = task.images[k].to("cpu")
            masks = task.masks.to("cpu")

            # Decode latents
            for k in images:
                if k == "inputs":
                    continue  # Already in RGB space
                if isinstance(images[k], torch.Tensor):
                    # assuming we use AutoencoderKL
                    if isinstance(self.cpu_decoder, AutoencoderKL):
                        images[k] = self.cpu_decoder.decode(images[k] / task.scale_factor).sample
                    else:
                        raise Exception("Assumed AutoencoderKL as the cpu_decoder during logging images!")
                    images[k] = self._convert_valid_log_format(images[k], images["inputs"])
                    if self.clamp and not isheatmap(images[k]):
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            # Perform the actual logging
            self.log_local(
                task.save_dir, task.split, images, masks,
                task.global_step, task.current_epoch, task.batch_idx, task.pl_module
            )
            
        except Exception as e:
            print(f"[ImageLogger] Error processing log task: {e}")

    def _queue_log_task(self, save_dir, split, images, masks, global_step, current_epoch, batch_idx, pl_module):
        """Queue a logging task for background processing"""
        if self.shutdown_event.is_set():
            print("[ImageLogger] Logger is shutting down, skipping log task")
            return
            
        task = LogTask(
            save_dir=save_dir,
            split=split,
            images=images,
            masks=masks,
            global_step=global_step,
            current_epoch=current_epoch,
            batch_idx=batch_idx,
            pl_module=pl_module,
            scale_factor=pl_module.scale_factor
        )
        
        try:
            # Non-blocking put with timeout
            self.log_queue.put(task, timeout=0.1)
            print(f"[ImageLogger] Queued log task for step {global_step}")
        except queue.Full:
            print(f"[ImageLogger] Queue full, skipping log task for step {global_step}")

    def shutdown(self):
        """Gracefully shutdown the logging thread"""
        print("[ImageLogger] Shutting down logging thread...")
        self.shutdown_event.set()
        
        # Send shutdown signal to queue
        try:
            self.log_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish (with timeout)
        if self.log_thread and self.log_thread.is_alive():
            self.log_thread.join(timeout=8.0 * 60.0) # it can take 8 minutes to log 4 images
            if self.log_thread.is_alive():
                print("[ImageLogger] Warning: Log thread did not stop gracefully")
        
        print("[ImageLogger] Logging thread shutdown complete")
    

    def add_colored_border(self, image_tensor, border_color, border_width=2):
        """
        Add a colored border around an image tensor.
        
        Args:
            image_tensor: Tensor of shape [C, H, W] in range [0, 1] (VAE decoded)
            border_color: Tuple of (R, G, B) values in range [0, 255]
            border_width: Width of the border in pixels
        
        Returns:
            Tensor with colored border
        """
        C, H, W = image_tensor.shape
        
        # Normalize border color from [0, 255] to [0, 1] range
        border_color = tuple(c / 255.0 for c in border_color)
        
        # Create border tensor
        border_tensor = torch.tensor(border_color, dtype=image_tensor.dtype, device=image_tensor.device)
        border_tensor = border_tensor.view(3, 1, 1).expand(3, H + 2*border_width, W + 2*border_width)
        
        # Create new image with border
        bordered_image = border_tensor.clone()
        bordered_image[:, border_width:border_width+H, border_width:border_width+W] = image_tensor

        del border_tensor
        return bordered_image

    @torch.no_grad()
    def tensor_to_image(self, tensor):
        # Denormalize and convert to PIL image
        tensor = tensor.cpu().squeeze(0)
        tensor = tensor * 0.5 + 0.5  # Denormalize
        # tensor = torch.clamp(tensor, 0, 1)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

    @torch.no_grad()
    def _convert_valid_log_format(self, latents, x):
        if latents.dim() == 4 and latents.shape[0] == x.shape[0] * x.shape[1]:
            batch_size = x.shape[0]
            num_images = x.shape[1]
            latents = latents.view(batch_size, num_images, *latents.shape[1:])
        return latents


    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        masks,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isheatmap(images[k]):
                print("ImageLogger::log_local:in local log_local:heatmap:")
                fig, ax = plt.subplots()
                ax = ax.matshow(
                    images[k].cpu().numpy(), cmap="hot", interpolation="lanczos"
                )
                plt.colorbar(ax)
                plt.axis("off")

                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                os.makedirs(root, exist_ok=True)
                path = os.path.join(root, filename)
                plt.savefig(path)
                plt.close()
                # TODO: support wandb
            else:            
                # SEVA multi-view tensors are already flattened in log_img to [N, C, H, W]
                # Add colored borders based on image type
                bordered_images = []
                for i, img in enumerate(images[k]):
                    # Determine border color based on image key or index
                    
                    if masks[i]: # inputs
                        border_color = (247.0, 121.0, 132.0)  # red
                    else: # targets
                        border_color = (101.0, 174.0, 219.0)  # blue

                    img = self.tensor_to_image(img) # (-1, 1) ->(0, 1)
                    bordered_img = self.add_colored_border(img, border_color, border_width=24)
                    bordered_images.append(bordered_img)
                
                # Stack bordered images and create grid
                # bordered_tensor = torch.stack(bordered_images)
                # print("bordered_tensor.shape: ", bordered_tensor.shape)
                # grid = torchvision.utils.make_grid(bordered_tensor, nrow=4, padding=24)
                grid = torchvision.utils.make_grid(bordered_images, nrow=4, padding=24)
                
                # if self.rescale:
                #     grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                
                grid = grid.permute(1, 2, 0).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k, global_step, current_epoch, batch_idx
                )
                path = os.path.join(root, filename)
                print("ImageLogger::Saving image to: ", path)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[
                            img,
                        ],
                        step=pl_module.global_step,
                    )

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train", sample=True): #pl_module: DiffusionEngine
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step

        # check if we should log at this batch index
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                # set eval for logs (sample generations before logging)
                pl_module.eval()

            # OLD -- GPU based logging
            # gpu_autocast_kwargs = {
            #     "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
            #     "dtype": torch.get_autocast_gpu_dtype(),
            #     "cache_enabled": torch.is_autocast_cache_enabled(),
            # }
            # with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
            #     # this sholud be where images are logged!
            #     print("ImageLogger::Logging images")
            #     # images = pl_module.log_images(
            #     #     batch, split=split, **self.log_images_kwargs
            #     # )

            # NEW -- CPU based logging, based on DiffusionEngine.log_images
            with torch.no_grad():
                conditioner_input_keys = [e.input_key for e in pl_module.conditioner.embedders]
                if self.log_images_kwargs.get("ucg_keys"):
                    ucg_keys = self.log_images_kwargs.get("ucg_keys")
                    assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                        "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                        f"but we have {ucg_keys} vs. {conditioner_input_keys}"
                    )
                else:
                    ucg_keys = conditioner_input_keys

                log = dict()
                x = pl_module.get_input(batch) # clean_latent
                c, uc = pl_module.conditioner.get_unconditional_conditioning(
                    batch,
                    force_uc_zero_embeddings=ucg_keys
                    if len(pl_module.conditioner.embedders) > 0
                    else [],
                )
                sampling_kwargs = {}

                # keep GPU until we have the generated latents
                N = min(x.shape[0], self.max_images)
                x = x.to(pl_module.device)[:N]
                z = pl_module.encode_first_stage(x) # identity; keep encoding on GPU

                for k in c:
                    if isinstance(c[k], torch.Tensor):
                        c[k], uc[k] = map(lambda y: y[k][:N].to(pl_module.device), (c, uc))
                # sample latents for targets
                if sample:
                    with pl_module.ema_scope("Plotting"):
                        samples = pl_module.sample(
                            c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                        )

                # async decoder + log to wandb stage -- move to CPU
                z = z.to("cpu")
                samples = samples.to("cpu")
                gt_images = batch["frames"][:N].to("cpu") # choose first N from B

                # unlike in original impl., we have yet to decode the latents; hence, pre-image
                pre_images = {}
                pre_images["inputs"] = gt_images
                pre_images["reconstructions"] = z
                pre_images["samples"] = samples

                # flatten for decoder
                for k in pre_images: # images is dict{inputs, reconstructions, samples} (as in diffusion.py)
                    if isinstance(pre_images[k], torch.Tensor):
                        # Handle SEVA multi-view tensors: [batch_size, num_images, C, H, W]
                        if pre_images[k].dim() == 5:
                            batch_size, num_images = pre_images[k].shape[:2]
                            total_images = batch_size * num_images
                            N = min(total_images, self.max_images)
                            # Flatten to [batch_size*num_images, C, H, W] for easier slicing
                            pre_images[k] = pre_images[k].view(total_images, *pre_images[k].shape[2:])
                            pre_images[k] = pre_images[k][:N]
                        else:
                            N = min(pre_images[k].shape[0], self.max_images)
                            if not isheatmap(pre_images[k]):
                                pre_images[k] = pre_images[k][:N]
                        
                        pre_images[k] = pre_images[k].detach().float().cpu()
                        # move clamping POST-decoder (which has range -1, 1)
                        # if self.clamp and not isheatmap(pre_images[k]):
                        #     pre_images[k] = torch.clamp(pre_images[k], -1.0, 1.0)

                masks = batch["mask"] # (B, max_images) binary boolean tensor
                masks = masks.reshape(-1)[:N].detach().cpu()

                if is_train: # if was training previously, set it back
                    # this shouldn't interfere, since the VAE is frozen anyways
                    pl_module.train()

                # add this iteration's images to the CPU-based logger queue
                self._queue_log_task(
                    pl_module.logger.save_dir, split, pre_images, masks,
                    pl_module.global_step, pl_module.current_epoch, batch_idx, pl_module
                )
            

            # for k in images: # images is dict{inputs, reconstructions, samples} (as in diffusion.py)
            #     if isinstance(images[k], torch.Tensor):
            #         # Handle SEVA multi-view tensors: [batch_size, num_images, C, H, W]
            #         if images[k].dim() == 5:
            #             batch_size, num_images = images[k].shape[:2]
            #             total_images = batch_size * num_images
            #             N = min(total_images, self.max_images)
            #             # Flatten to [batch_size*num_images, C, H, W] for easier slicing
            #             images[k] = images[k].view(total_images, *images[k].shape[2:])
            #             images[k] = images[k][:N]
            #         else:
            #             N = min(images[k].shape[0], self.max_images)
            #             if not isheatmap(images[k]):
            #                 images[k] = images[k][:N]
                    
            #         images[k] = images[k].detach().float().cpu()
            #         if self.clamp and not isheatmap(images[k]):
            #             images[k] = torch.clamp(images[k], -1.0, 1.0)

            # # flatten masks to correspond with images[:N], detach
            # masks = masks.reshape(-1)[:N].detach().cpu()


    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
            pass

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            # self.log_img(pl_module, batch, batch_idx, split="train")
            pass

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs
    ):
        if not self.disabled and pl_module.global_step > 0:
            # self.log_img(pl_module, batch, batch_idx, split="val")
            pass
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (
                pl_module.calibrate_grad_norm and batch_idx % 25 == 0
            ) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")


@rank_zero_only
def init_wandb(save_dir, opt, config, group_name, name_str):
    print(f"setting WANDB_DIR to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    os.environ["WANDB_DIR"] = save_dir
    if opt.debug:
        wandb.init(project=opt.projectname, mode="offline", group=group_name)
    else:
        config_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(
            project=opt.projectname,
            config=config_dict,
            settings=wandb.Settings(code_dir="./sgm"),
            group=group_name,
            name=name_str,
        )


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    melk_ckpt_name = None
    name = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
            _, melk_ckpt_name = get_checkpoint_name(logdir)
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt, melk_ckpt_name = get_checkpoint_name(logdir)

        print("#" * 100)
        print(f'Resuming from checkpoint "{ckpt}"')
        print("#" * 100)

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            if opt.no_base_name:
                name = ""
            else:
                if opt.legacy_naming:
                    cfg_fname = os.path.split(opt.base[0])[-1]
                    cfg_name = os.path.splitext(cfg_fname)[0]
                else:
                    assert "configs" in os.path.split(opt.base[0])[0], os.path.split(
                        opt.base[0]
                    )[0]
                    cfg_path = os.path.split(opt.base[0])[0].split(os.sep)[
                        os.path.split(opt.base[0])[0].split(os.sep).index("configs")
                        + 1 :
                    ]  # cut away the first one (we assert all configs are in "configs")
                    cfg_name = os.path.splitext(os.path.split(opt.base[0])[-1])[0]
                    cfg_name = "-".join(cfg_path) + f"-{cfg_name}"
                name = "_" + cfg_name
        else:
            name = ""
        if not opt.no_date:
            nowname = now + name + opt.postfix
        else:
            nowname = name + opt.postfix
            if nowname.startswith("_"):
                nowname = nowname[1:]
        logdir = os.path.join(opt.logdir, nowname)
        print(f"LOGDIR: {logdir}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    # move before model init, in case a torch.compile(...) is called somewhere
    if opt.enable_tf32:
        # pt_version = version.parse(torch.__version__)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Enabling TF32 for PyTorch {torch.__version__}")
    else:
        print(f"Using default TF32 settings for PyTorch {torch.__version__}:")
        print(
            f"torch.backends.cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
        )
        print(f"torch.backends.cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}")

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # default to gpu
        trainer_config["accelerator"] = "gpu"
        #
        standard_args = default_trainer_args()
        for k in standard_args:
            if getattr(opt, k) != standard_args[k]:
                trainer_config[k] = getattr(opt, k)

        ckpt_resume_path = opt.resume_from_checkpoint

        print("ckpt_resume_path: ", ckpt_resume_path)
        print("trainer_config: ", trainer_config)

        if not "devices" in trainer_config and trainer_config["accelerator"] != "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
            print(f"Running on GPUs {gpuinfo}")
            gpuinfo = [gpuinfo] # ! - CUDA Accelerate wants this as a list; if not using, comment this out
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model) # DiffusionEngine

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    # "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                    "project": opt.projectname,
                    "log_model": False,
                    # "dir": logdir,
                },
            },
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "testtube",  # hack for sbord fanatics
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb" if opt.wandb else "csv"]
        if opt.wandb:
            # TODO change once leaving "swiffer" config directory
            try:
                group_name = nowname.split(now)[-1].split("-")[1]
            except:
                group_name = nowname
            default_logger_cfg["params"]["group"] = group_name
            init_wandb(
                os.path.join(os.getcwd(), logdir),
                opt=opt,
                group_name=group_name,
                config=config,
                name_str=nowname,
            )
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")

        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
        # default to ddp if not further specified
        default_strategy_config = {"target": "pytorch_lightning.strategies.DDPStrategy"}

        if "strategy" in lightning_config:
            strategy_cfg = lightning_config.strategy
        else:
            strategy_cfg = OmegaConf.create()
            default_strategy_config["params"] = {
                "find_unused_parameters": False,
                # "static_graph": True,
                # "ddp_comm_hook": default.fp16_compress_hook  # TODO: experiment with this, also for DDPSharded
            }
        strategy_cfg = OmegaConf.merge(default_strategy_config, strategy_cfg)
        print(
            f"strategy config: \n ++++++++++++++ \n {strategy_cfg} \n ++++++++++++++ "
        )
        trainer_kwargs["strategy"] = instantiate_from_config(strategy_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                    "ckpt_name": melk_ckpt_name,
                },
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {"batch_frequency": 1000, "max_images": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                },
            },
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
            print(
                "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
            )
            default_metrics_over_trainsteps_ckpt_dict = {
                "metrics_over_trainsteps_checkpoint": {
                    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        "save_top_k": -1,
                        "every_n_train_steps": 10000,
                        "save_weights_only": True,
                    },
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and ckpt_resume_path is not None:
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = ckpt_resume_path
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()

        # cmd line trainer args (which are in trainer_opt) have always priority over config-trainer-args (which are in trainer_kwargs)
        trainer_opt = vars(trainer_opt) # from trainer in yaml
        trainer_kwargs = {
            key: val for key, val in trainer_kwargs.items() if key not in trainer_opt
        } # logger, strategy, callbacks, etc.
        trainer = Trainer(**trainer_opt, **trainer_kwargs)

        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        # data.setup()
        print("#### Data #####")
        try:
            for k in data.datasets:
                print(
                    f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            print("datasets not yet initialized.")

        # configure learning rate
        if "batch_size" in config.data.params:
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        else:
            bs, base_lr = (
                config.data.params.train.loader.batch_size,
                config.model.base_learning_rate,
            )
        if not cpu:
            ngpu = len(lightning_config.trainer.devices.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs): # emergency checkpointing
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                if melk_ckpt_name is None:
                    ckpt_path = os.path.join(ckptdir, "last.ckpt")
                else:
                    ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs): # emergency debugger
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                print(model)
                trainer.fit(model, data, ckpt_path=ckpt_resume_path)
            except Exception:
                if not opt.debug:
                    melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as err:
        if MULTINODE_HACKS:
            import datetime
            import os
            import socket

            import requests

            device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            hostname = socket.gethostname()
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
            print(
                f"ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
                flush=True,
            )
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

        if opt.wandb:
            wandb.finish()
        # if trainer.global_rank == 0:
        #    print(trainer.profiler.summary())
