import torch
from typing import Callable, Tuple

from seva.data.preprocessing import update_intrinsics, get_bbox_center_and_size

# NOTE: this is applied to the OUTPUT 576x576 image shape AFTER initial cropping!
class RandomBBoxCropper(object):
    def __init__(self, crop_size_bounds=None):
        """
        Random crop transform centered around a bounding box with Gaussian mixture sampling.
        Expected OFFSETS (zero mean) instead of pixel-space means.
        NOTE: images are NOT resized to (576, 576) here!

        Args:
            mean: A 2D tensor/array/tuple of shape (2,)
            std: A 2D tensor/array/tuple of shape (2,)
        """
        self.crop_size_bounds = crop_size_bounds # (min_crop_size, max_crop_size)

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
            bbox: Tensor of shape (4,) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (3, 3)
            options: Dictionary containing the following keys:
                - "W": Width of the image (used to clamp samples)
                - "H": Height of the image (used to clamp samples)
                - "center_mean": 2D list of mean values (x,y)_mean
                - "center_std": 2D list of std values (x,y)_std
                - "crop_size_mean": 1D list of mean values (crop_size)_mean
                - "crop_size_std": 1D list of std values (crop_size)_std
            NOTE: mean & std can be in normalized (0-1) or absolute (pixel) values.
            
        Returns:
            (x1, y1, x2, y2): Crop coordinates
            K_new: Updated intrinsics matrix
        """
        W = options["W"] # of IMAGE (not bbox)
        H = options["H"] # of IMAGE (not bbox)

        # random params
        print("orig bbox:", bbox)

        # get initial bbox params
        center, size = get_bbox_center_and_size(bbox)
        center_x, center_y = center
        bbox_W, bbox_H = size

        center_mean    = options.get("center_mean", (center_x, center_y))
        center_std     = options.get("center_std", ((W - bbox_W) / 6, (H - bbox_H) / 6))
        crop_size_mean = options.get("crop_size_mean", (bbox_W + bbox_H) // 2)
        crop_size_std  = options.get("crop_size_std", (bbox_W + bbox_H) / 6)

        # transform all to absolute pixel values (for crop_size, based on min(H,W))
        # mean should be WITHIN the bbox
        center_mean    = percent_to_absolute(center_mean, torch.tensor([H, W]))
        center_std     = torch.as_tensor(center_std)
        crop_size_mean = percent_to_absolute(crop_size_mean, torch.tensor([min(H, W)]))
        crop_size_std  = torch.as_tensor(crop_size_std)

        print("samples:")
        print(center_mean, center_std, crop_size_mean, crop_size_std)

        # sample new center and length of crop
        center_sample = torch.randn(2) * center_std    + center_mean
        size_sample   = torch.randn(1) * crop_size_std + crop_size_mean

        if self.crop_size_bounds is not None:
            size_sample = torch.clamp(
                size_sample,
                min=percent_to_absolute(self.crop_size_bounds[0], torch.tensor([min(H, W)])),
                max=percent_to_absolute(self.crop_size_bounds[1], torch.tensor([min(H, W)]))
            )
        
        # calculate crop coordinates of NEW post-sampled crop
        center_x, center_y = map(int, center_sample)
        crop_size = int(size_sample[0])

        # "clamp" center within initial bbox
        x1 = max(0, center_x - (crop_size // 2))
        y1 = max(0, center_y - (crop_size // 2))
        x2 = min(W, center_x + (crop_size // 2))
        y2 = min(H, center_y + (crop_size // 2))
        
        # update intrinsics
        K_new = update_intrinsics(
            torch.as_tensor(K), 
            crop_x=x1, 
            crop_y=y1, 
            scale=1, # for MVHumanNet images (downsampled)
            crop_first=False,
            padding_mode=True
        )

        # crop parameters, updated intrinsics
        return {
            "bbox": torch.tensor([x1, y1, x2, y2]),
            "K": K_new
        }

    def __call__(
        self, 
        image: torch.Tensor, 
        bbox: torch.Tensor, 
        K: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: Tensor of shape (B, C, H, W)
            bbox: Tensor of shape (B, 4) with [x1, y1, x2, y2]
            K: Intrinsics matrix of shape (B, 3, 3)

        Returns:
            Cropped image and updated intrinsics matrix
        """

        # kwargs will always include image size metadata
        # if other parameters (mean, std) are NOT provided, then, we use the center as the mean,
        # and the crop shape
        options = {
            "H": image.shape[-2],
            "W": image.shape[-1],
        }
        options.update(kwargs) # bias towards face based on "annots" face box!

        # get new crop parameters
        crop_params = self._get_crop_params(bbox, K, options)
        bbox = crop_params["bbox"]
        K_new = crop_params["K"]

        # get new crop coordinates
        x1, y1, x2, y2 = bbox
        
        # Handle padding if needed
        H, W = image.shape[1:3]
        pad_left = int(max(0, -x1))
        pad_top = int(max(0, -y1))
        pad_right = int(max(0, x2 - W))
        pad_bottom = int(max(0, y2 - H))
        
        # if the new crop parameters extend beyond the image, pad the image
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            padding = [pad_left, pad_right, pad_top, pad_bottom]
            image = T.Pad(padding)(image)
            
            # Adjust crop coordinates
            x1, x2 = int(x1 + pad_left), int(x2 + pad_left)
            y1, y2 = int(y1 + pad_top), int(y2 + pad_top)
            
            # and then update the intrinsics from padding (left or top; 
            # (negative because negative cropping is positive padding)
            # if right/bottom, no need to update intrinsics
            K_new = update_intrinsics(
                K_new,
                crop_x=-pad_left,
                crop_y=-pad_top,
                scale=1,
                crop_first=False,
                padding_mode=True
            )

        image_ = torch.as_tensor(image)

        # Perform the actual crop
        # Instead of cropping, preserve original dimensions and add a bounding box
        image = image_[:, y1:y2, x1:x2]  # Clone to avoid modifying the original
        return image, K_new


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