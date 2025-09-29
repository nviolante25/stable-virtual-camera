import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from einops import rearrange, repeat

from ...util import append_dims, default

logpy = logging.getLogger(__name__)


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG(Guider):
    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "mask", "plucker", "replace", "dense_vector"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


# from seva.sampling
class ConstantScaleRule(object):
    def __call__(self, scale: float | torch.Tensor) -> float | torch.Tensor:
        return scale

class MultiviewScaleRule(object):
    def __init__(self, min_scale: float = 1.0):
        self.min_scale = min_scale

    def __call__(
        self,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        c2w_input = c2w[input_frame_mask]
        rotation_diff = get_camera_dist(c2w, c2w_input, mode="rotation").min(-1).values
        translation_diff = (
            get_camera_dist(c2w, c2w_input, mode="translation").min(-1).values
        )
        K_diff = (
            ((K[:, None] - K[input_frame_mask][None]).flatten(-2) == 0).all(-1).any(-1)
        )
        close_frame = (rotation_diff < 10.0) & (translation_diff < 1e-5) & K_diff
        if isinstance(scale, torch.Tensor):
            scale = scale.clone()
            scale[close_frame] = self.min_scale
        elif isinstance(scale, float):
            scale = torch.where(close_frame, self.min_scale, scale)
        else:
            raise ValueError(f"Invalid scale type {type(scale)}.")
        return scale


class ConstantScaleSchedule(object):
    def __call__(
        self, sigma: float | torch.Tensor, scale: float | torch.Tensor
    ) -> float | torch.Tensor:
        if isinstance(sigma, float):
            return scale
        elif isinstance(sigma, torch.Tensor):
            if len(sigma.shape) == 1 and isinstance(scale, torch.Tensor):
                sigma = append_dims(sigma, scale.ndim)
            return scale * torch.ones_like(sigma)
        else:
            raise ValueError(f"Invalid sigma type {type(sigma)}.")


class ConstantGuidance(object):
    def __call__(
        self,
        uncond: torch.Tensor,
        cond: torch.Tensor,
        scale: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(scale, torch.Tensor) and len(scale.shape) == 1:
            scale = append_dims(scale, cond.ndim)
        return uncond + scale * (cond - uncond)


class SevaVanillaCFG(Guider):
    def __init__(self):
        self.scale_rule = ConstantScaleRule()
        self.scale_schedule = ConstantScaleSchedule()
        self.guidance = ConstantGuidance()

    def __call__(
        self, x: torch.Tensor, sigma: float | torch.Tensor, scale: float | torch.Tensor
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        scale = self.scale_rule(scale)
        scale_value = self.scale_schedule(sigma, scale)
        x_pred = self.guidance(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "replace", "dense_vector"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class SevaMultiviewCFG(SevaVanillaCFG):
    def __init__(self, cfg_min: float = 1.0):
        self.scale_min = cfg_min
        self.scale_rule = MultiviewScaleRule(min_scale=cfg_min)
        self.scale_schedule = ConstantScaleSchedule()
        self.guidance = ConstantGuidance()

    def __call__(  # type: ignore
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor,
        scale: float | torch.Tensor,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        scale = self.scale_rule(scale, c2w, K, input_frame_mask)
        scale_value = self.scale_schedule(sigma, scale)
        x_pred = self.guidance(x_u, x_c, scale_value)
        return x_pred


class IdentityGuider(Guider):
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out


class LinearPredictionGuider(Guider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)

        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)

        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"] + self.additional_cond_keys:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                # assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class TrianglePredictionGuider(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        period: Union[float, List[float]] = 1.0,
        period_fusing: Literal["mean", "multiply", "max"] = "max",
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(max_scale, num_frames, min_scale, additional_cond_keys)
        values = torch.linspace(0, 1, num_frames)
        # Constructs a triangle wave
        if isinstance(period, float):
            period = [period]

        scales = []
        for p in period:
            scales.append(self.triangle_wave(values, p))

        if period_fusing == "mean":
            scale = sum(scales) / len(period)
        elif period_fusing == "multiply":
            scale = torch.prod(torch.stack(scales), dim=0)
        elif period_fusing == "max":
            scale = torch.max(torch.stack(scales), dim=0).values
        self.scale = (scale * (max_scale - min_scale) + min_scale).unsqueeze(0)

    def triangle_wave(self, values: torch.Tensor, period) -> torch.Tensor:
        return 2 * (values / period - torch.floor(values / period + 0.5)).abs()


class TrapezoidPredictionGuider(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        edge_perc: float = 0.1,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(max_scale, num_frames, min_scale, additional_cond_keys)

        rise_steps = torch.linspace(min_scale, max_scale, int(num_frames * edge_perc))
        fall_steps = torch.flip(rise_steps, [0])
        self.scale = torch.cat(
            [
                rise_steps,
                torch.ones(num_frames - 2 * int(num_frames * edge_perc)),
                fall_steps,
            ]
        ).unsqueeze(0)

        
class SpatiotemporalPredictionGuider(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        num_views: int = 1,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(max_scale, num_frames, min_scale, additional_cond_keys)
        V = num_views
        T = num_frames // V
        scale = torch.zeros(num_frames).view(T, V)
        scale += torch.linspace(0, 1, T)[:,None] * 0.5
        scale += self.triangle_wave(torch.linspace(0, 1, V))[None,:] * 0.5
        scale = scale.flatten()
        self.scale = (scale * (max_scale - min_scale) + min_scale).unsqueeze(0)

    def triangle_wave(self, values: torch.Tensor, period=1) -> torch.Tensor:
        return 2 * (values / period - torch.floor(values / period + 0.5)).abs()