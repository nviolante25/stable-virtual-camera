import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA layers
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class LoRAAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rank: int = 4,
        alpha: float = 4.0,
        keys_to_lora: list[str] = ["q", "k", "v"],
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.keys_to_lora = keys_to_lora
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        # Original attention layers
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        
        # Freeze original attention layers
        for param in self.to_q.parameters():
            param.requires_grad = False
        for param in self.to_k.parameters():
            param.requires_grad = False
        for param in self.to_v.parameters():
            param.requires_grad = False
        for param in self.to_out.parameters():
            param.requires_grad = False

        # LoRA layers
        if "q" in self.keys_to_lora:
            self.lora_q = LoRALinear(query_dim, inner_dim, rank=rank, alpha=alpha, dropout=dropout)
        if "k" in self.keys_to_lora:
            self.lora_k = LoRALinear(context_dim, inner_dim, rank=rank, alpha=alpha, dropout=dropout)
        if "v" in self.keys_to_lora:
            self.lora_v = LoRALinear(context_dim, inner_dim, rank=rank, alpha=alpha, dropout=dropout)
        if "o" in self.keys_to_lora:
            self.lora_out = LoRALinear(inner_dim, query_dim, rank=rank, alpha=alpha, dropout=dropout)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Original projections
        q = self.to_q(x) + self.lora_q(x) if "q" in self.keys_to_lora else self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context) + self.lora_k(context) if "k" in self.keys_to_lora else self.to_k(context)
        v = self.to_v(context) + self.lora_v(context) if "v" in self.keys_to_lora else self.to_v(context)

        # Convert to float16 for attention
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
        
        # Convert back to original dtype
        out = out.to(x.dtype)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.to_out(out) + self.lora_out(out) if "o" in self.keys_to_lora else self.to_out(out)
        return out

# unused for now
class LoRAFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        
        # Original layers
        self.proj = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Linear(inner_dim, dim_out)
        
        # LoRA layers
        self.lora_proj = LoRALinear(dim, inner_dim * 2, rank=rank, alpha=alpha, dropout=dropout)
        self.lora_proj_out = LoRALinear(inner_dim, dim_out, rank=rank, alpha=alpha, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj(x) + self.lora_proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.proj_out(x) + self.lora_proj_out(x)
        return self.dropout(x) 