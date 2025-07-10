import math
from weakref import ref
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from seva.modules.autoencoder import AutoEncoder
import torch
import torchvision
import matplotlib.pyplot as plt
import lpips
import numpy as np

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
from seva.eval import transform_img_and_K
from sgm.data.utils_camera import read_extrinsics_nerfstudio, read_intrinsics_nerfstudio
from sgm.data.dataset import center_cameras, scale_cameras
from seva.geometry import get_plucker_coordinates
from einops import repeat
import json
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

from seva.modules.lora_wrapper import SevaLoRAWrapper
from seva.modules.conditioner import CLIPConditioner
from sgm.models.diffusion import DiffusionEngine
from sgm.modules.encoders.modules import GeneralConditioner
from sgm.modules.diffusionmodules.denoiser import DiscreteDenoiser
from sgm.modules.diffusionmodules.sampling import EulerEDMSampler
from sgm.modules.diffusionmodules.discretizer import LegacyDDPMDiscretization
from sgm.modules.diffusionmodules.denoiser_scaling import EpsScaling
from sgm.modules.diffusionmodules.guiders import VanillaCFG
from sgm.modules.encoders.modules import IdentityEncoder, SevaFrozenOpenCLIPImageEmbedder
from sgm.models.autoencoder import IdentityFirstStage
from seva.model import SGMWrapper
from sgm.util import instantiate_from_config
from seva.utils import load_model
from sgm.modules.diffusionmodules.wrappers import SevaWrapper, SevaWrapperV2

import yaml
from omegaconf import OmegaConf
from seva.sampling import EulerEDMSampler, DDPMDiscretization, MultiviewCFG, DiscreteDenoiser
from sgm.models.autoencoder import IdentityFirstStage
from sgm.modules.encoders.modules import IdentityEncoder, SevaFrozenOpenCLIPImageEmbedder, GeneralConditioner
from seva.model import SGMWrapper

# EVAL

# -- garden_flythrough --
# from demo: 18.06 mean PSNR, 0.5430 SSIM (nearest-gt)
# from here: 

# -- dl3d140 --
# from demo: 


def eval_report(samples, frames, input_masks, T, reference=None):
    # Reference can be concatenated to the right
    # such as adding the generation via the demo itself
    means = {}
    means["psnr"] = 0.0
    means["ssim"] = 0.0
    means["lpips"] = 0.0
    if reference is not None:
        means["ref_psnr"] = 0.0
        means["ref_ssim"] = 0.0
        means["ref_lpips"] = 0.0
    num_target_frames = 0
    visual_images = []
    input_images = []
    grid = None  # Initialize grid to None

    ref_cnt = 0 # need separate counter (reference images should be targets ONLY)
    for i in range(T):
        if input_masks[i] == 0: # if target frame
            num_target_frames += 1
            
            # Create a row of images for this frame
            frame_row = torch.cat([
                frames[i].permute(1,2,0),
                samples[i].permute(1,2,0),
                grayscale_to_jet(create_heatmap(samples[i].permute(1,2,0), frames[i].permute(1,2,0))),
                # seva_ref_imgs[i].permute(1,2,0)
            ], dim=1)

            if reference is not None:
                frame_row = torch.cat([
                    frame_row,
                    reference[ref_cnt].permute(1,2,0),
                    grayscale_to_jet(create_heatmap(frames[i].permute(1,2,0), reference[ref_cnt].permute(1,2,0))),
                ], dim=1)

                ref_cnt += 1

            visual_images.append(frame_row)

            curr_pnsr = compute_psnr(samples[i:i+1], frames[i:i+1])
            curr_ssim = compute_ssim(samples[i:i+1], frames[i:i+1])
            curr_lpips = compute_lpips(samples[i:i+1], frames[i:i+1], normalized=True)
            print(f"image {i}: \t [PSNR↑] {curr_pnsr:.3f}, \t [SSIM↑] {curr_ssim:.3f}, \t[LPIPS↓] {curr_lpips:.3f}")
            if reference is not None:
                print(f"ref_img: \t [PSNR↑] {compute_psnr(frames[i:i+1], reference[ref_cnt-1:ref_cnt]):.3f}, \t [SSIM↑] {compute_ssim(frames[i:i+1], reference[ref_cnt-1:ref_cnt]):.3f}, \t[LPIPS↓] {compute_lpips(frames[i:i+1], reference[ref_cnt-1:ref_cnt], normalized=True):.3f}")
                means["ref_psnr"] += compute_psnr(frames[i:i+1], reference[ref_cnt-1:ref_cnt])
                means["ref_ssim"] += compute_ssim(frames[i:i+1], reference[ref_cnt-1:ref_cnt])
                means["ref_lpips"] += compute_lpips(frames[i:i+1], reference[ref_cnt-1:ref_cnt], normalized=True)
            means["psnr"] += curr_pnsr
            means["ssim"] += curr_ssim
            means["lpips"] += curr_lpips
        else:
            input_images.append(frames[i])

    print(f"mean PSNR: {means['psnr'] / num_target_frames:.3f}, mean SSIM: {means['ssim'] / num_target_frames:.3f}, mean LPIPS: {means['lpips'] / num_target_frames:.3f}")
    if reference is not None:
        print(f"mean ref PSNR: {means['ref_psnr'] / ref_cnt:.3f}, mean ref SSIM: {means['ref_ssim'] / ref_cnt:.3f}, mean ref LPIPS: {means['ref_lpips'] / ref_cnt:.3f}")

    if input_images:
        input_grid = torchvision.utils.make_grid(
            torch.stack(input_images, dim=0), 
            nrow=5,  # One row per frame (each frame is already a row of 3 images)
            padding=10,  # Add padding between frames
            pad_value=1.0  # White padding
        )
        print("\nINPUTS:")
        input_grid = input_grid.permute(1, 2, 0)
        plt.figure(figsize=(20, 20))  # Set figure size to make it larger
        plt.imshow(input_grid)
        plt.axis('off')  # Remove axes for cleaner display
        plt.show()
    
    # Stack images vertically and use make_grid for better spacing
    if visual_images:
        print("GT | TARGET | DIFFERENCE:")
        stacked_images = torch.stack(visual_images, dim=0)  # [N, H, W, C]
        # Convert to [N, C, H, W] format for make_grid
        stacked_images = stacked_images.permute(0, 3, 1, 2)
        
        # Create grid with padding and spacing
        grid = torchvision.utils.make_grid(
            stacked_images, 
            nrow=1,  # One row per frame (each frame is already a row of 3 images)
            padding=20,  # Add padding between frames
            pad_value=1.0  # White padding
        )
        
        # Convert back to [H, W, C] for display
        grid = grid.permute(1, 2, 0)
        
        plt.figure(figsize=(40, 40))  # Set figure size to make it larger
        plt.imshow(grid)
        plt.axis('off')  # Remove axes for cleaner display
        plt.show()

    return means, grid

    # save samples to:
    # output_dir = "output_samples"
    # os.makedirs(output_dir, exist_ok=True)

    # for i in range(T):
    #     if input_masks[i] == 0:  # if target frame
    #         # Save the sample[i:i+1] as an image
    #         sample_img = samples[i:i+1]
    #         save_path = os.path.join(output_dir, f"sample_{i}.png")
    #         # Clamp to [0,1] if needed
    #         save_image(sample_img, save_path)


def create_heatmap(img1, img2, plot=False):
    # Assume we want to create a heatmap of the absolute difference between the first two images in samples and frames
    # (or you can choose any two images as needed)
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    # Compute absolute difference
    diff = abs(img1 - img2)

    # If images are multi-channel, you can sum or mean across channels for a 2D heatmap
    if diff.ndim == 3 and diff.shape[2] > 1:
        diff_map = diff.mean(axis=2)
    else:
        diff_map = diff


    if plot:
        plt.figure(figsize=(6, 6))
        plt.title("Heatmap of Absolute Difference Between Images")
        plt.imshow(diff_map,cmap="jet")
        plt.colorbar(label='Absolute Difference')
        plt.axis('off')
        plt.show()

    return repeat(torch.from_numpy(diff_map), 'h w -> h w c', c=3)

def grayscale_to_jet(img):
    """
    Converts a grayscale image (H, W, 3) where all channels are equal to a jet-colored image (H, W, 3) using the jet colormap.

    Args:
        img (torch.Tensor or np.ndarray): Input image of shape (H, W, 3) with identical channels.

    Returns:
        torch.Tensor: Jet-colored image of shape (H, W, 3), values in [0, 1].
    """
    import numpy as np

    # Convert to numpy if torch
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = img

    # Ensure shape is (H, W, 3)
    assert img_np.ndim == 3 and img_np.shape[2] == 3, "Input must be (H, W, 3)"
    # Take one channel (since all are equal)
    gray = img_np[..., 0]

    # Normalize to [0, 1]
    gray_min = gray.min()
    gray_max = gray.max()
    if gray_max > gray_min:
        gray_norm = (gray - gray_min) / (gray_max - gray_min)
    else:
        gray_norm = np.zeros_like(gray)

    # Use matplotlib's jet colormap to map to RGB
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('jet')
    jet_img = cmap(gray_norm)[:, :, :3]  # Drop alpha

    # Convert back to torch if needed
    jet_img = torch.from_numpy(jet_img).float()
    return jet_img


def load_image_directory_as_tensor(image_dir, image_ext='*.png', image_size=None, device='cpu'):
    """
    Loads all images from a directory into a single tensor batch.

    Args:
        image_dir (str): Path to the directory containing images.
        image_ext (str): Image file extension pattern (default: '*.png').
        image_size (tuple or None): If specified, resize images to this size (W, H).
        device (str): Device to load tensors onto.

    Returns:
        torch.Tensor: Batch of images as a tensor of shape [N, C, H, W].
        list: List of image file paths in the order loaded.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, image_ext)))
    transform_list = [transforms.ToTensor()]
    if image_size is not None:
        transform_list.insert(0, transforms.Resize(image_size))
    transform = transforms.Compose(transform_list)

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    if len(images) == 0:
        raise ValueError(f"No images found in {image_dir} with extension {image_ext}")
    batch = torch.stack(images, dim=0).to(device)
    return batch, image_paths


def compare_image_directories(dir1_path, dir2_path, image_ext='*.png'):
    """
    Calculate PSNR and SSIM between corresponding images in two directories.
    
    Args:
        dir1_path: Path to first directory containing images
        dir2_path: Path to second directory containing images  
        image_ext: Image file extension pattern (default: '*.png')
        
    Returns:
        dict: Dictionary containing PSNR and SSIM values for each image pair
    """
    # Get sorted lists of image files from both directories
    images1 = sorted(glob.glob(os.path.join(dir1_path, image_ext)))
    images2 = sorted(glob.glob(os.path.join(dir2_path, image_ext)))
    
    if len(images1) != len(images2):
        raise ValueError(f"Number of images don't match: {len(images1)} vs {len(images2)}")
    
    # Setup image transformation to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    results = {
        'psnr_values': [],
        'ssim_values': [],
        'image_pairs': [],
        'mean_psnr': 0.0,
        'mean_ssim': 0.0
    }
    
    for img1_path, img2_path in zip(images1, images2):
        # Load and convert images to tensors
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img1_tensor = transform(img1)
        img2_tensor = transform(img2)

        # Convert to tensors [C, H, W] in range [0, 1]
        # img1_tensor = transform(img1).unsqueeze(0)  # Add batch dimension
        # img2_tensor = transform(img2).unsqueeze(0)  # Add batch dimension

        img1_tensor, _ = transform_img_and_K(
            img1_tensor.unsqueeze(0),
            (576, 576)
        )

        img2_tensor, _ = transform_img_and_K(
            img2_tensor.unsqueeze(0),
            (576, 576)
        )

        # visual_tensor = torch.cat([img1_tensor, img2_tensor], dim=-1).squeeze(0)
        # plt.imshow(visual_tensor.permute(1, 2, 0))
        # plt.show()
        
        # Calculate PSNR and SSIM using your functions
        psnr = compute_psnr(img1_tensor, img2_tensor)
        ssim = compute_ssim(img1_tensor, img2_tensor)
        
        # Store results
        results['psnr_values'].append(psnr.item())
        results['ssim_values'].append(ssim.item())
        results['image_pairs'].append((os.path.basename(img1_path), os.path.basename(img2_path)))
    
    # Calculate mean values
    results['mean_psnr'] = sum(results['psnr_values']) / len(results['psnr_values'])
    results['mean_ssim'] = sum(results['ssim_values']) / len(results['ssim_values'])
    
    return results

def print_eval(dir1_path, dir2_path):
    results = compare_image_directories(dir1_path, dir2_path)
    print(f"Mean PSNR: {results['mean_psnr']:.2f}")
    print(f"Mean SSIM: {results['mean_ssim']:.4f}")
    for res in results['psnr_values']:
        print(res)



# helper functions
@torch.no_grad()
def encode_first_stage(x):
        n_samples = x.shape[0]
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=False):
            for n in range(n_rounds):
                out = first_stage.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = scale_factor * z
        return z

@torch.no_grad()
def decode_first_stage(z):
    # z = 1.0 / scale_factor * z -- VAE will do this automatically
    n_samples = z.shape[0]

    n_rounds = math.ceil(z.shape[0] / n_samples)
    all_out = []
    with torch.autocast("cuda", enabled=False):
        for n in range(n_rounds):
            if hasattr(first_stage, "decoder") and \
                isinstance(first_stage.decoder, VideoDecoder):
                kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
            else:
                kwargs = {}
            out = first_stage.decode(
                z[n * n_samples : (n + 1) * n_samples], **kwargs
            )
            all_out.append(out)
    out = torch.cat(all_out, dim=0)
    return out

def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    # Convert [-1, 1] to [0, 1] for display
    t = (t + 1.0) / 2.0
    return t.clamp(0, 1)

def add_colored_border(img: torch.Tensor, color: tuple, border_width: int = 8):
    c, h, w = img.shape
    bordered = torch.ones(3, h + 2 * border_width, w + 2 * border_width)
    for i in range(3):
        bordered[i] *= color[i] / 255.0
    bordered[:, border_width:-border_width, border_width:-border_width] = img
    return bordered

def show_tensor_batch(images: torch.Tensor, frames: torch.Tensor=None, masks=None, nrow=4, border_width=8):
    """
    images: Tensor [B, 3, H, W] in [-1, 1] or [0, 1]
    masks: Optional Bool Tensor [B], True=red, False=blue
    """

    if masks is not None:
        bordered_images = []
        for i, img in enumerate(images):
            img_ = img
            color = (247, 121, 132) if masks[i] else (101, 174, 219)
            if frames is not None and masks[i] == True: 
                img_ = frames[i]
            bordered_images.append(add_colored_border(img_, color, border_width))
        images = torch.stack(bordered_images)

    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=16)
    npimg = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(24, 12))
    plt.imshow(npimg)
    plt.axis('off')
    plt.show()
    return npimg

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio between predicted and target images.
    
    Args:
        pred: Predicted images tensor of shape [B, C, H, W] in range [-1, 1] or [0, 1]
        target: Target images tensor of shape [B, C, H, W] in range [-1, 1] or [0, 1]
        
    Returns:
        PSNR value as tensor
    """
    
    mse = torch.mean((pred - target) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM) between predicted and target images.
    
    Args:
        pred: Predicted images tensor of shape [B, C, H, W] in range [-1, 1] or [0, 1]
        target: Target images tensor of shape [B, C, H, W] in range [-1, 1] or [0, 1]
        window_size: Size of the sliding window. Default: 11
        
    Returns:
        SSIM value as tensor
    """
    
    # Constants for stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Generate Gaussian window
    window = torch.ones(window_size, window_size) / (window_size * window_size)
    window = window.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    window = window.expand(pred.size(1), 1, window_size, window_size)
    window = window.to(pred.device)
    
    mu1 = torch.nn.functional.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
    mu2 = torch.nn.functional.conv2d(target, window, padding=window_size//2, groups=target.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=window_size//2, groups=pred.size(1)) - mu1_mu2
    
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean()

def compute_lpips(pred, target, normalized=False):
    # should be range [-1, 1]!!!
    loss_fn = lpips.LPIPS(net="vgg", verbose=False)
    if normalized: # [0, 1] -> [-1, 1]
        pred = 2.0 * pred - 1.0
        target = 2.0 * target - 1.0
    return loss_fn(pred, target).mean().detach()

from einops import repeat

def run_seva_sample(batch, model_components, scale_factor=0.18215, cfg=2.0):
    """
    Runs the SEVA sampling pipeline on a batch using provided model components.

    Args:
        batch (dict): Input batch containing keys like 'mask', 'plucker', 'clean_latent', 'frames', 'c2ws', 'Ks'.
        model_components (dict): Dictionary containing model components:
            {
                'ae': autoencoder,
                'conditioner': conditioner,
                'sampler': sampler,
                'denoiser': denoiser,
                'model': model
            }
        scale_factor (float): Optional scaling factor for latents.

    Returns:
        samples (torch.Tensor): Decoded samples.
        frames (torch.Tensor): Input frames (reshaped).
        input_masks (torch.Tensor): Input mask tensor.
    """
    from einops import repeat
    ae = model_components['autoencoder']
    conditioner = model_components['conditioner']
    sampler = model_components['sampler']
    denoiser = model_components['denoiser']
    model = model_components['model']

    T = batch["mask"].shape[-1]
    batch = {k: v.to("cuda") for k, v in batch.items()}
    input_masks = batch["mask"].reshape(T).to("cuda")
    pluckers = batch["plucker"].reshape(T, 6, 72, 72).to("cuda")
    clean_latent = batch["clean_latent"].to("cuda")
    frames = batch["frames"].to("cuda")

    with torch.inference_mode(), torch.autocast("cuda"):
        latents = (clean_latent * scale_factor)
        latents = latents.reshape(T, 4, 72, 72)
        latents = torch.nn.functional.pad(
            latents, (0, 0, 0, 0, 0, 1), value=1.0
        )
        frames = frames.reshape(T, 3, 576, 576)
        c_crossattn = repeat(conditioner(frames).mean(0), "d -> n 1 d", n=T)
        uc_crossattn = torch.zeros_like(c_crossattn)
        c_replace = latents.new_zeros(T, *latents.shape[1:])
        c_replace[input_masks] = latents[input_masks]
        uc_replace = torch.zeros_like(c_replace)
        c_concat = torch.cat(
            [
                repeat(
                    input_masks,
                    "n -> n 1 h w",
                    h=pluckers.shape[2],
                    w=pluckers.shape[3],
                ),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers], 1
        )
        c_dense_vector = pluckers
        uc_dense_vector = c_dense_vector
        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": c_dense_vector,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": uc_dense_vector,
        }

        additional_model_inputs = {"num_frames": T}

        randn = torch.randn(T, 4, 72, 72).to("cuda")
        samples_z = sampler(
            lambda input, sigma, c: denoiser(
                model,
                input,
                sigma,
                c,
                **additional_model_inputs,
            ),
            randn,
            scale=cfg,  # cfg
            cond=c,
            uc=uc,
            **{
                "c2w": batch["c2ws"].reshape(T, 4, 4).to("cuda"),
                "K": batch["Ks"].reshape(T, 3, 3).to("cuda"),
                "input_frame_mask": input_masks.to("cuda"),
            },
        )
        samples = ae.decode(samples_z, chunk_size=1)
    return samples, frames, input_masks


def all_keys_in_dict(dict, keys):
    # keys should be a list
    for key in keys:
        if key not in dict:
            return False
    return True


def load_from_transforms_json_and_split(
    transforms_json_path,
    num_train_images,
    autoencoder, # need for encoding
    scale_factor=0.18215,
    target_shape=(576, 576),
    downsample_factor=8
):
    """
        Loads images from a transforms.json file and splits them into train and test sets.
        Returns a batch of images and their corresponding camera parameters.

        Args:
            transforms_json_path: Path to the transforms.json file.
            num_train_images: Number of train images.
            num_test_images: Number of test images.
            autoencoder: Autoencoder model.
            scale_factor: Scale factor for the images.
            target_shape: Target shape for the images.
            downsample_factor: Downsample factor for the images.

        Returns:
            batch: Batch of images and their corresponding camera parameters.
            - clean_latents: Clean latents for the images.
            - frames: Frames for the images.
            - input_mask: Input mask for the images.
            - c2ws: Camera parameters for the images.
            - Ks: Intrinsics for the images.
            - pluckers: Pluckers for the images.
            - concat: Concatenated images and their corresponding camera parameters.
    """

    # get extrinsics + intrinsics, input & targets
    transforms_dict = json.load(open(os.path.join(transforms_json_path, "transforms.json")))
    split_dict = json.load(open(os.path.join(transforms_json_path, f"train_test_split_{num_train_images}.json")))

    image_files = []
    input_mask = []
    all_Ks = []

    if all_keys_in_dict(transforms_dict, ["w", "h", "fl_x", "fl_y", "cx", "cy"]):
        all_Ks = read_intrinsics_nerfstudio(
            transforms_dict=transforms_dict,
            normalize=False
        )
        all_Ks = repeat(torch.tensor(all_Ks), "h w -> n h w", n=len(transforms_dict["frames"]))

    for i, frame in enumerate(transforms_dict["frames"]):
        if i in split_dict["train_ids"]:
            image_files.append(frame["file_path"])
            input_mask.append(1)
        elif i in split_dict ["test_ids"]:
            image_files.append(frame["file_path"])
            input_mask.append(2)
        else:
            input_mask.append(0)
        

        if all_keys_in_dict(frame, ["w", "h", "fl_x", "fl_y", "cx", "cy"]):
            all_Ks.append(read_intrinsics_nerfstudio(
                transforms_dict=frame,
                normalize=False
            ))

    input_mask = np.array(input_mask)
    images_idx = np.where(input_mask != 0)[0] # get our "T" images
    input_frames_indices = np.where(input_mask == 1)[0]
    input_frames_mask = torch.zeros(len(images_idx), dtype=torch.bool)
    input_frames_mask[input_frames_indices] = True # inputs are 1, targets 0
    Ks = torch.tensor(all_Ks)[images_idx]

    num_images = len(images_idx)

    frames = torch.zeros((num_images, 3, 576, 576))
    for i, (img_file, K) in enumerate(zip(image_files, Ks)):
        img_path = os.path.join(transforms_json_path, img_file)
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image) # may need to scale K here
        image, K = transform_img_and_K(image.unsqueeze(0), size=(576, 576), K=K[None]) # images converted to square, [-1, 1] normalization
        W, H = image.shape[-2:]
        # print("W, H:", W, H) 
        K[:, 0] /= W
        K[:, 1] /= H
        Ks[i] = K
        image = transforms.Normalize([0.5], [0.5])(image)
        frames[i] = image

    clean_latents = autoencoder.encode(frames.to("cuda"), chunk_size=1) # scales automatically
    frames = frames.to("cpu")
    clean_latents = clean_latents.to("cpu")

    all_c2ws = torch.from_numpy(read_extrinsics_nerfstudio(
        transforms_dict=transforms_dict,
        mode="c2w"
    )).float()

    all_c2ws[:, :, [1,2]] *= -1 # this is fine (OpenGL -> OpenCV)

    c2ws = all_c2ws[images_idx]
    center_cameras(all_c2ws, c2ws) # mean center
    scale_cameras(c2ws)

    camera_mask = torch.ones(len(images_idx), dtype=torch.bool)

    w2cs = torch.linalg.inv(c2ws)
    pluckers = get_plucker_coordinates(
        extrinsics_src=w2cs[input_frames_indices[0]],
        extrinsics=w2cs,
        intrinsics=Ks.float().clone(),
        target_size=(target_shape[0] // downsample_factor, 
                            target_shape[1] // downsample_factor),
    )

    concat = torch.cat(
        [
            repeat(
                input_frames_mask,
                "n -> n 1 h w",
                h=pluckers.shape[2],
                w=pluckers.shape[3],
            ),
            pluckers,
        ],
        dim=1
    )

    replace = torch.cat( # clean latents and binary mask
        [
            clean_latents,
            repeat(
                input_frames_mask,
                "n -> n 1 h w",
                h=pluckers.shape[2],
                w=pluckers.shape[3],
            ),
        ],
        dim=1,
    )

    batch = {
        "clean_latent": clean_latents / scale_factor,
        "mask": input_frames_mask,
        "plucker": pluckers,
        "camera_mask": camera_mask,
        "concat": concat,
        "frames": frames,
        "replace": replace,
        "c2ws": c2ws,
        "Ks": Ks,
    }

    return batch


def save_tensor_dict(tensor_dict, file_path):
    """
    Save a dictionary of tensors to a file using torch.
    Args:
        tensor_dict (dict): Dictionary where values are torch.Tensor.
        file_path (str): Path to save the file (should end with .pt or .pth).
    """
    torch.save(tensor_dict, file_path)

def save_image(array, file_path):
    """
    Save image to a file.
    Args:
        image (ndarray): Image numpy array.
        file_path (str): Path to save the file (should end with .png).
    """
    im = Image.fromarray((array*255).astype(np.uint8))
    im.save(file_path)


def init_seva(vanilla: bool=True):
    # * BELOW: faithful Seva process
    with torch.autocast("cuda"):
        ae = AutoEncoder(chunk_size=1).eval().to("cuda")
        conditioner = CLIPConditioner().to("cuda") # change this to GeneralConditioner with SevaFrozenOpenCLIPImageEmbedder and test for equality
        sampler = EulerEDMSampler(
            discretization=DDPMDiscretization(
                linear_start=5e-06,
                linear_end=0.012,
                num_timesteps=1000,
                log_snr_shift=2.4,
            ),
            guider=MultiviewCFG(cfg_min=1.2),
            num_steps=50,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=999.0,
            s_noise=1.0,
        )
        denoiser = DiscreteDenoiser(
            discretization=DDPMDiscretization(),
            num_idx=1000,
            device="cuda",
        )

        model = SGMWrapper( # swap to SevaWrapper during training
            instantiate_from_config(
                {
                    "target": "seva.modules.lora_wrapper.SevaLoRAWrapper",
                    "params": {
                        "seva_model_config": {
                            "target": "seva.utils.load_model",
                        },
                        "self_attn_rank": 4,
                        "cross_attn_rank": 8,
                        "alpha": 16.0,
                        "dropout": 0.0,
                        "target_modules": ["TransformerBlockTimeMix", "MultiviewTransformer", "TransformerBlock"],
                        "keys_to_lora": ["q", "k", "v"],
                    }
                }
            )
        ).to("cuda")

    model_components = {
        "autoencoder": ae,
        "conditioner": conditioner,
        "sampler": sampler,
        "denoiser": denoiser,
        "model": model
    }
    return model_components


# ! - below: config for ported Seva code
# Load the config file
# config_path = "configs/example_training/seva-true.yaml"
# config = OmegaConf.load(config_path)
# components_config = config.model.params

# # setup components from YAML config
# first_stage = instantiate_from_config(components_config.first_stage_config) # should stay IdentityFirstStage/DecoderOnly
# conditioner = instantiate_from_config(components_config.conditioner_config) # GeneralConditioner
# sampler = instantiate_from_config(components_config.sampler_config) # EulerEDMSampler (Seva implementation)
# denoiser = instantiate_from_config(components_config.denoiser_config).to("cuda") # SevaDiscreteDenoiser
# sigma_sampler = instantiate_from_config(components_config.loss_fn_config.params.sigma_sampler_config) # DiscreteSampling with DDPMDiscretization
# model = SevaWrapper(instantiate_from_config(components_config.network_config).eval()).to("cuda")

# use DDMPDiscretization over LegacyDDPM


# Legacy (put in init_model when vanilla=False)
# then, match results with the original seva model

# def run_one_scene(model, conditioner, sigma_sampler, denoiser, batch):
#     with torch.no_grad():
#         inputs = batch["clean_latent"]
#         latents = encode_first_stage(inputs) # * scale_factor (internally)
#         # latents = latents.reshape(-1, *latents.shape[2:])
#         # condition
#         cond = conditioner(batch)
#         # sample
#         sigmas = sigma_sampler(latents.shape[0]).to(latents)
#         noise = torch.randn_like(latents)
#         sigmas_bc = append_dims(sigmas, latents.ndim)
#         noised_input = latents + noise * sigmas_bc

#         print("inputs.shape:", inputs.shape)
#         print("latents.shape:", latents.shape)
#         print("cond.keys:", cond.keys())

#         sigmas = sigmas.to("cuda")
#         latents = latents.to("cuda")
#         noised_input = noised_input.to("cuda")
#         cond = {k: v.to("cuda") for k, v in cond.items()}

#         model_output = denoiser(
#             model, noised_input, sigmas, cond
#         )

#         outputs = decode_first_stage(model_output).detach()
#     return outputs

# batch = next(iter(loader))
# vae = AutoEncoder().eval().to("cuda")
# output_latents = run_one_scene(model, conditioner, sigma_sampler, denoiser, batch)
# rgb_outputs = vae.decode(output_latents.view(-1, *output_latents.shape[2:])).to("cpu")
# gt_images = batch["frames"].view(-1, *batch["frames"].shape[2:])

# masks = batch["mask"].reshape(-1)
# show_tensor_batch(rgb_outputs, masks=masks)
# show_tensor_batch(gt_images, frames=batch["frames"].view(-1, *batch["frames"].shape[2:]), masks=masks)
