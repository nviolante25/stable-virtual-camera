from abc import ABC, abstractmethod

import torch


class DiffusionLossWeighting(ABC):
    @abstractmethod
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        pass


class UnitWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DiffusionLossWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma**-2.0
    
class SevaWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor, mask, max_weight=5.0) -> torch.Tensor:
        # * for phase 2, mask is now ref_mask (originally input_frames_mask)
        # NOTE: weights  are based on "dists", but indices are not necessarily 
        # ordered by distance from a reference frame
        # if we want, we can sort them by distance from a reference frame within the mvhn_dataloader
        bools = mask.to(torch.bool)
        batch_size, N = bools.shape
        indices = torch.arange(N, device=bools.device).unsqueeze(0).expand(batch_size, N)
        weights = torch.full((batch_size, N), max_weight, dtype=torch.float, device=bools.device)
        
        for b in range(batch_size):
            true_idx = indices[b][bools[b]]
            if len(true_idx) > 0:
                dists = torch.stack([torch.abs(indices[b] - t) for t in true_idx]).min(dim=0).values
                dists[bools[b]] = 0
                weights[b] = dists / dists.max() * max_weight
            else:
                weights[b] = max_weight

        return weights

        
class SimVSWeighting(DiffusionLossWeighting):
    def __call__(self, sigma: torch.Tensor, mask, ref_mask, max_weight=5.0) -> torch.Tensor:
        # * for phase 2, mask is now ref_mask (originally input_frames_mask)
        # only ref_mask gets no weight (since it's the only clean latent)
        # every input needs a weight to learn ic->consistent
        # every output needs to learn NVS for new samples

        # weight non-ref frames by a constant (1.0)
        # additionally weight by a constant for ref frames
        bools = mask.to(torch.bool)
        ref_bools = ref_mask.to(torch.bool)
        batch_size, N = bools.shape
        indices = torch.arange(N, device=bools.device).unsqueeze(0).expand(batch_size, N)
        weights = torch.full((batch_size, N), max_weight, dtype=torch.float, device=bools.device)
        ref_weights = torch.ones_like(ref_bools, dtype=torch.float) - ref_bools.to(torch.float) # zero for ref frames, one for others
        
        for b in range(batch_size):
            true_idx = indices[b][bools[b]]
            if len(true_idx) > 0:
                dists = torch.stack([torch.abs(indices[b] - t) for t in true_idx]).min(dim=0).values
                dists[bools[b]] = 0
                weights[b] = dists / dists.max() * max_weight
            else:
                weights[b] = max_weight

        weights = torch.clamp(weights + ref_weights, min=0, max=max_weight)
        return weights

