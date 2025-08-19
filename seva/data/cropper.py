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
        - padding: [left, top, right, bottom] (in pixels)
        """
        self.crop_size_bounds = crop_size_bounds # (min_crop_size, max_crop_size)
        self.random_crop = random_crop # if maximal_crop only, then should be False
        if isinstance(padding, int) or isinstance(padding, float):
            self.padding = [padding, padding, padding, padding]
        elif isinstance(padding, list):
            self.padding = padding
        else:
            raise ValueError(f"Invalid padding type: {type(padding)}")

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

        if self.random_crop:

            # sampling distribution parameters
            center_mean    = options.get("center_mean", centers)
            center_std     = options.get("center_std", torch.stack([(W - bbox_W) / 6, (H - bbox_H) / 6], dim=1))
            crop_size_mean = options.get("crop_size_mean", (bbox_W + bbox_H) // 2)
            crop_size_std  = options.get("crop_size_std", (bbox_W + bbox_H) / 6)
            longest_dim    = torch.max(bbox_W, bbox_H)
            min_crop_size  = options.get("min_crop_size", longest_dim // 2) # half of the larger dimension

            # transform all to absolute pixel values (for crop_size, based on min(H,W))
            # mean should be WITHIN the bbox
            center_mean    = percent_to_absolute(center_mean, torch.tensor([H, W]))
            center_std     = torch.as_tensor(center_std)
            crop_size_mean = percent_to_absolute(crop_size_mean, torch.tensor([min(H, W)]))
            crop_size_std  = torch.as_tensor(crop_size_std)

            # * sample new center and length of crop
            max_bound = torch.cat((torch.ones(B,1) * W, torch.ones(B,1) * H), dim=1) - (min_crop_size.reshape(-1, 1) // 2)
            center_sample = torch.clamp(
                torch.randn(B, 2) * center_std + center_mean, 
                min=torch.zeros_like(longest_dim).unsqueeze(1), 
                max=max_bound
            )
            size_sample = torch.clamp(torch.randn(B) * crop_size_std + crop_size_mean, min=min_crop_size, max=longest_dim)

            if self.crop_size_bounds is not None:
                size_sample = torch.clamp(
                    size_sample,
                    min=percent_to_absolute(self.crop_size_bounds[0], torch.tensor([min(H, W)])),
                    max=percent_to_absolute(self.crop_size_bounds[1], torch.tensor([min(H, W)]))
                )
            
            # calculate crop coordinates of NEW post-sampled crop
            # center_x, center_y = map(int, center_sample)
            center_sample = center_sample.int()
            crop_size = torch.maximum(size_sample.int(), min_crop_size)

            center_x, center_y = center_sample.T
        else:
            crop_size = torch.maximum(bbox_W.int(), bbox_H.int())
        
        # if not random crop, then here, center_x and center_y are the "means"

        # "clamp" center within initial bbox, ensuring positive optical centers
        c1, c2 = K[0, 2], K[1, 2] # ! assumes same intrinsics
        x1 = torch.clamp(center_x - (crop_size // 2), min=0, max=c1).int()
        y1 = torch.clamp(center_y - (crop_size // 2), min=0, max=c2).int()
        x2 = torch.clamp(center_x + (crop_size // 2), min=0, max=W).int()
        y2 = torch.clamp(center_y + (crop_size // 2), min=0, max=H).int()

        # add padding here (but only within bounds)
        x1 = torch.clamp(x1 - self.padding[0], min=0, max=W)
        y1 = torch.clamp(y1 - self.padding[1], min=0, max=H)
        x2 = torch.clamp(x2 + self.padding[2], min=0, max=W)
        y2 = torch.clamp(y2 + self.padding[3], min=0, max=H)
        
        # if crop size has changed, add the rest to x2 and y2 (but clamp to W, H)
        x_crop_size = x2 - x1
        y_crop_size = y2 - y1

        # Update x2 where crop size has changed
        x_mask = x_crop_size != crop_size
        x2 = torch.where(x_mask, torch.clamp(x1 + crop_size, max=W), x2)

        # Update y2 where crop size has changed
        y_mask = y_crop_size != crop_size
        y2 = torch.where(y_mask, torch.clamp(y1 + crop_size, max=H), y2)

        # * update intrinsics
        if len(K.shape) == 2: # repeat original K (3,3) to (B, 3, 3)
            K_ = torch.tensor(repeat(K, 'd1 d2 -> n d1 d2', n=B))

        K_new = update_intrinsics(
            torch.as_tensor(K_), 
            crop_x=x1, # (B,)
            crop_y=y1, # (B,)
            scale=1, # for MVHumanNet images (downsampled)
            crop_first=False,
            padding_mode=True
        )

        # crop parameters, updated intrinsics
        return {
            "bbox": torch.stack([x1, y1, x2, y2], dim=1), # (B, 4)
            "K": K_new
        }

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

        # get new crop coordinates
        x1, y1, x2, y2 = bbox.T
        
        # Handle padding if needed
        H, W = images.shape[-2:]
        pad_left = torch.maximum(torch.zeros_like(x1), -x1)
        pad_top = torch.maximum(torch.zeros_like(y1), -y1)
        pad_right = torch.maximum(torch.zeros_like(x2), x2 - W)
        pad_bottom = torch.maximum(torch.zeros_like(y2), y2 - H)
        
        # if the new crop parameters extend beyond the image, pad the image
        # ! this should NEVER occur with the updated code
        if torch.any(pad_left > 0) or torch.any(pad_top > 0) or torch.any(pad_right > 0) or torch.any(pad_bottom > 0):
            print("RandomBBoxCropper::__call__: image padding occurred; should not happen!")
            # padding = [pad_left, pad_top, pad_right, pad_bottom]
            # images = torch.nn.functional.pad(images, padding, mode="constant", value=0)
            
            # # Adjust crop coordinates
            # x1, x2 = x1 + pad_left, x2 + pad_left
            # y1, y2 = y1 + pad_top, y2 + pad_top
            
            # # and then update the intrinsics from padding (left or top; 
            # # (negative because negative cropping is positive padding)
            # # if right/bottom, no need to update intrinsics
            # K_new = update_intrinsics(
            #     K_new,
            #     crop_x=-pad_left,
            #     crop_y=-pad_top,
            #     scale=1,
            #     crop_first=False,
            #     padding_mode=True
            # )

        images_ = torch.as_tensor(images)

        # Perform the actual crop
        # Instead of cropping, preserve original dimensions and add a bounding box

        batch_size = images_.shape[0]
        cropped_images = []
        for i in range(batch_size):
            cropped_img = images_[i, :, int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])]
            cropped_images.append(cropped_img)

        return cropped_images, K_new


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