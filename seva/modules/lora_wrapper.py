import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .lora import LoRALinear, LoRAAttentionWrapper
from ..model import Seva, SevaParams
from sgm.util import instantiate_from_config


def skip_module_if_excluded(module_path: str, excluded_modules: list[str]) -> bool:
    path_parts = module_path.split('.')
    for i in range(len(path_parts)):
        parent_path = '.'.join(path_parts[:i+1])
        if parent_path in excluded_modules:
            return True
    return False


class SevaLoRAWrapper(nn.Module):
    def __init__(
        self,
        seva_model_config: Dict[str, Any],
        self_attn_rank: int = 4,
        cross_attn_rank: int = 8,
        alpha: float = 1.0,  # Reduced from 4.0
        dropout: float = 0.0,
        target_modules: Optional[list[str]] = None,
        keys_to_lora: list[str] = ["q", "k", "v"],
        excluded_modules: list[str] = [],
    ):
        super().__init__()
        self.seva_model = instantiate_from_config(seva_model_config)
        self.seva_model.eval()
        self.self_attn_rank = self_attn_rank
        self.cross_attn_rank = cross_attn_rank
        self.alpha = alpha
        self.dropout = dropout
        self.excluded_modules = excluded_modules
        self.keys_to_lora = keys_to_lora
        
        # excluded_modules expects an list/set of module path strings
        # an example: output_blocks.9.TimestepEmbedSequential.1.MultiviewTransformer.time_mix_blocks.ModuleList.0.TransformerBlockTimeMix.attn1
        # (excludes the LoRA-transformed MultiviewTransformer's attn1 block in the 9th element of output_blocks)
        # can exclude an entire block by passing in the block's name (e.g. "output_blocks.9", "input_blocks", etc.)

        # Freeze base model parameters
        for param in self.seva_model.parameters():
            param.requires_grad = False

        if target_modules is None:
            # only attention-based layers by default (can extend to 'ff' if needed)
            target_modules = {"TransformerBlockTimeMix", "MultiviewTransformer", "TransformerBlock"}

        # Wrap attention layers with LoRA
        self._wrap_attention_with_lora(
            self.seva_model, 
            target_modules, 
            dropout, 
            self_attn_rank, 
            cross_attn_rank, 
            alpha, 
            keys_to_lora,
            excluded_modules
        )

    def _wrap_attention_with_lora(self, module, target_modules, dropout, self_attn_rank, cross_attn_rank, alpha, keys_to_lora, excluded_modules, current_path=""):
        """Recursively wrap attention modules with LoRA instead of replacing them."""
        for name, child in module.named_children():
            module_path = f"{current_path}.{name}.{child.__class__.__name__}" if current_path else name

            if skip_module_if_excluded(module_path, excluded_modules):
                continue

            if child.__class__.__name__ in target_modules and module_path not in excluded_modules:
                if hasattr(child, "attn1"):  # self-attention
                    child.attn1 = LoRAAttentionWrapper(
                        original_attn=child.attn1,
                        rank=self_attn_rank,
                        alpha=alpha,
                        dropout=dropout,
                        keys_to_lora=keys_to_lora,
                    )
                if hasattr(child, "attn2"):  # cross-attention
                    child.attn2 = LoRAAttentionWrapper(
                        original_attn=child.attn2,
                        rank=cross_attn_rank,
                        alpha=alpha,
                        dropout=dropout,
                        keys_to_lora=keys_to_lora,
                    )
                    
            # Recursively process child modules
            self._wrap_attention_with_lora(child, target_modules, dropout, self_attn_rank, cross_attn_rank, alpha, keys_to_lora, excluded_modules, module_path)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, dense_y: torch.Tensor, num_frames: Optional[int] = None) -> torch.Tensor:
        return self.seva_model(x, t, y, dense_y, num_frames)

    def save_lora_weights(self, path: str):
        """Save only the LoRA weights."""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRAAttentionWrapper):
                for param_name, param in module.named_parameters():
                    if "lora_" in param_name:
                        lora_state_dict[f"{name}.{param_name}"] = param
        torch.save(lora_state_dict, path)

    def load_lora_weights(self, path: str):
        """Load only the LoRA weights."""
        lora_state_dict = torch.load(path)
        self.load_state_dict(lora_state_dict, strict=False) 
