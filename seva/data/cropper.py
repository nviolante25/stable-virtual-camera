import torch
from typing import Callable, Tuple
from einops import repeat

from seva.data.preprocessing import update_intrinsics, get_bbox_center_and_size

# NOTE: this should be applied to the OUTPUT 576x576 image shape AFTER initial cropping!
class RandomBBoxCropper(object):
    def __init__(self, random_crop=True, crop_size_bounds=None, padding=[0,0,0,0]):
        """
        Random (Gaussian) crop transform centered around a 2D bounding box.
        NOTE: images are NOT resized to (576, 576) here!
        - padding: [left, top, right, bottom] (in pixels) only for deterministic crop!
        """
        self.crop_size_bounds = crop_size_bounds # (min_crop_size, max_crop_size)
        self.random_crop = random_crop # if maximal_crop only, then should be False
        if isinstance(padding, int) or isinstance(padding, float):
            # ! for now, should always be an int for uniform padding!
            self.padding = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            self.padding = padding
        else:
            raise ValueError(f"Invalid padding type: {type(padding)}")
        if random_crop:
            self.padding = [0,0,0,0]

    def _get_crop_params(
        self, 
        bbox: torch.Tensor, 
        K: torch.Tensor,
        options: dict,
    ) -> Tuple[int, int, int, int, torch.Tensor]:
        """
        Calculate crop parameters based on bbox and intrinsics.
        BBox is the initial "crop" onto the image, before random cropping (done here).

        NOTE: `pre_scale` will affect bbox parameters here.
        
        Args:
            bbox: Tensor of shape (B,4) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (B, 3, 3)
            options: Dictionary containing the following keys:
                - "W": Width of the image (used to clamp samples)
                - "H": Height of the image (used to clamp samples)
                - "center_mean": 2D list of mean values (x,y)_mean
                - "center_std": 2D list of std values (x,y)_std
                - "crop_size_mean": 1D list of mean values (crop_size)_mean
                - "crop_size_std": 1D list of std values (crop_size)_std
            NOTE: mean & std can be in normalized (0-1) or absolute (pixel) values.
            
        Returns:
            (x1, y1, x2, y2): Crop coordinates (B, 4)
            K_new: Updated intrinsics matrix (B, 3, 3)
            rel_bbox: Relative bbox coordinates to inconsistent images
        """
        W = options["W"] # of IMAGE (not bbox)
        H = options["H"] # of IMAGE (not bbox)
        B = bbox.shape[0] # batch size

        # get initial bbox params
        center, size = get_bbox_center_and_size(bbox)
        centers = torch.stack(center, dim=1) # (B, 2)
        sizes = torch.stack(size, dim=1) # (B, 2)
        center_x, center_y = centers.T # (B, 2)
        bbox_W, bbox_H = sizes.T # (B, 2)

        # deterministic bbox crop:
        old_center_x, old_center_y = center_x, center_y
        bbox_max_dim = torch.maximum(bbox_W, bbox_H) # (B,)
        x1 = torch.floor(old_center_x - (bbox_max_dim // 2) - self.padding[0]).int()
        x2 = torch.floor(old_center_x + (bbox_max_dim // 2) + self.padding[2]).int()
        y1 = torch.ceil(old_center_y - (bbox_max_dim // 2) - self.padding[1]).int()
        y2 = torch.ceil(old_center_y + (bbox_max_dim // 2) + self.padding[3]).int()
        rel_bbox = torch.zeros(B, 4)

        if self.random_crop:
            # if random, then give relative bbox (to the initial maximal crop)
            # sampling distribution parameters
            center_mean    = options.get("center_mean", centers)
            center_std     = options.get("center_std", torch.stack([(W - bbox_W) / 6, (H - bbox_H) / 6], dim=1))
            crop_size_mean = options.get("crop_size_mean", (bbox_W + bbox_H) // 2)
            crop_size_std  = options.get("crop_size_std", (bbox_W + bbox_H) / 6)
            min_crop_size  = options.get("min_crop_size", bbox_max_dim // 2) # 1/2 of the larger dimension

            # transform all to absolute pixel values (for crop_size, based on min(H,W))
            # mean should be WITHIN the bbox
            center_mean    = percent_to_absolute(center_mean, torch.tensor([H, W]))
            center_std     = torch.as_tensor(center_std)
            crop_size_mean = percent_to_absolute(crop_size_mean, torch.tensor([min(H, W)]))
            crop_size_std  = torch.as_tensor(crop_size_std)

            # sample a crop_size
            size_sample = torch.clamp(torch.randn(B) * crop_size_std + crop_size_mean, min=min_crop_size, max=bbox_max_dim)

            # sample x and y offsets (that remain in the initial bbox) from center
            x_offset = torch.clamp(
                torch.randn(B) * center_std[0] + center_mean[0],
                min=x1 + size_sample // 2,
                max=x2 - size_sample // 2
            )
            y_offset = torch.clamp(
                torch.randn(B) * center_std[1] + center_mean[1], 
                min=y1 + size_sample // 2,
                max=y2 - size_sample // 2
            )

            # calculate new crop coordinates
            x1_new = torch.floor(x_offset - (size_sample // 2)).int()
            y1_new = torch.floor(y_offset - (size_sample // 2)).int()
            x2_new = torch.ceil(x_offset + (size_sample // 2)).int()
            y2_new = torch.ceil(y_offset + (size_sample // 2)).int()

            if self.crop_size_bounds is not None:
                size_sample = torch.clamp(
                    size_sample,
                    min=percent_to_absolute(self.crop_size_bounds[0], torch.tensor([min(H, W)])),
                    max=percent_to_absolute(self.crop_size_bounds[1], torch.tensor([min(H, W)]))
                )
            
            # # calculate crop coordinates of NEW post-sampled crop
            # # center_x, center_y = map(int, center_sample)
            # center_sample = center_sample.int()
            # crop_size = torch.maximum(size_sample.int(), min_crop_size)
            # center_x, center_y = center_sample.T
        
            # # c1, c2 = K[0, 2], K[1, 2] # ! assumes same intrinsics
            # x1_new = (center_x - (crop_size // 2) - self.padding[0]).int()
            # y1_new = (center_y - (crop_size // 2) - self.padding[1]).int()
            # x2_new = (center_x + (crop_size // 2) + self.padding[2]).int()
            # y2_new = (center_y + (crop_size // 2) + self.padding[3]).int()

            # calculate relative bbox
            rel_bbox[:, 0] = x1_new - x1 # d_x1
            rel_bbox[:, 1] = y1_new - y1 # d_y1
            rel_bbox[:, 2] = x2_new - x2 # d_x2
            rel_bbox[:, 3] = y2_new - y2 # d_y2
            x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new

        # * update intrinsics
        if len(K.shape) == 2: # repeat original K (3,3) to (B, 3, 3)
            K_ = repeat(K, 'd1 d2 -> n d1 d2', n=B).detach().clone()

        K_new = update_intrinsics(
            torch.as_tensor(K_), 
            crop_x=x1, # (B,)
            crop_y=y1, # (B,)
            scale=1, # for MVHumanNet images (downsampled)
            crop_first=False,
            padding_mode=True
        )

        # scale crop coordinates to to canonical 576^2
        scale = 576.0 / bbox_max_dim
        rel_bbox = (rel_bbox * scale.view(-1, 1)).int()

        # crop parameters, updated intrinsics
        return {
            "bbox": torch.stack([x1, y1, x2, y2], dim=1), # (B, 4)
            "K": K_new,
            "relative_bbox": rel_bbox # (B, 4)
        }

    def _possibly_pad_img(self, images, x1, y1, x2, y2):
        """
        Pad the image if the crop parameters extend beyond the image.
        """
        # handle padding if needed
        H, W = images.shape[-2:]
        pad_left = torch.maximum(torch.zeros_like(x1), -x1)
        pad_top = torch.maximum(torch.zeros_like(y1), -y1)
        pad_right = torch.maximum(torch.zeros_like(x2), x2 - W)
        pad_bottom = torch.maximum(torch.zeros_like(y2), y2 - H)
        
        # if the new crop parameters extend beyond the image, pad the image
        if torch.any(pad_left > 0) or torch.any(pad_top > 0) or torch.any(pad_right > 0) or torch.any(pad_bottom > 0):
            # print("WARNING: Crop parameters extend beyond the image!")
            image_list = []
            padding = torch.stack([pad_left.int(), pad_right.int(), pad_top.int(), pad_bottom.int()], dim=1)

            # images is a list of different sized images, so need to iterate separately
            for i, image in enumerate(images):
                image = torch.nn.functional.pad(image, padding[i].tolist(), mode="constant", value=0)
                image_list.append(image) # keep list since padding is different for each image

            # ! already done in get_crop_params
            # K_new = update_intrinsics(
            #     K,
            #     crop_x=-pad_left,
            #     crop_y=-pad_top,
            #     scale=1,
            #     crop_first=False,
            #     padding_mode=True
            # )
            new_bbox = torch.stack([x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top], dim=1)
            return image_list, new_bbox
        else:
            return images, torch.stack([x1, y1, x2, y2], dim=1)

    def __call__(
        self, 
        images: torch.Tensor, 
        bbox: torch.Tensor, 
        K: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, C, H, W)
            bbox: Tensor of shape (B, 4) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (B, 3, 3)

        Returns:
            Cropped image and updated intrinsics matrix
        """

        # kwargs will always include image size metadata
        # if other parameters (mean, std) are NOT provided, then, we use the center as the mean,
        # and the crop shape
        options = {
            "H": images.shape[-2],
            "W": images.shape[-1],
        }
        options.update(kwargs) # bias towards face based on "annots" face box!

        # get new crop parameters
        crop_params = self._get_crop_params(bbox, K, options) # * GOOD
        bbox = crop_params["bbox"]
        K_new = crop_params["K"]
        rel_bbox = crop_params["relative_bbox"] # for cropping ic images (scaled to 576^2)

        # get new crop coordinates
        x1, y1, x2, y2 = bbox.T
        # if negative coordinates, need to pad the image (K already previously updated)
        images, bbox = self._possibly_pad_img(images, x1, y1, x2, y2)
        x1, y1, x2, y2 = bbox.T

        # perform the actual crop
        cropped_images = []
        for i in range(len(images)):
            cropped_img = images[i][:, int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            cropped_images.append(cropped_img)

        return cropped_images, K_new, rel_bbox


def percent_to_absolute(arr, abs_arr):
    _arr = torch.as_tensor(arr)
    orig_shape = _arr.shape
    _arr = _arr.reshape(-1)
    decimal_mask = (torch.where((_arr <= 1) & (_arr >= 0))[0]).to(torch.int32)
    if len(decimal_mask) == 0:
        return _arr.reshape(orig_shape).to(torch.float32)
    _arr[decimal_mask] = _arr[decimal_mask] * abs_arr # convert to pixel coords
    return _arr.reshape(orig_shape).to(torch.float32)


# use for later; we'll need this to convert the bbox from crop_params.npz to centered square
# NOTE: bbox values can be negative, in which case, we'll need to pad the image to fit (during runtime)
# REMEMBER TO UPDATE INTRINSICS!
# - for our inconsistent dataset, we'll use this to explicitly crop images to square, then reshape to 576x576 to put into pipeline
# - for our MVHN dataloader, we only explicitly crop the image (and update intrinsics) when we need it (during runtime)
# ! - DATALOADER CURRENTLY HAS K NORMALIZED INTRINSICS! Be sure to do the cropping inside the dataloader!
def convert_to_square_crop(bbox):
    """
    Convert a bbox to a square crop.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    crop_size = max(w, h)
    center = (x1 + (w // 2), y1 + (h // 2))
    x1 = center[0] - (crop_size // 2)
    y1 = center[1] - (crop_size // 2)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return (x1, y1, x2, y2)