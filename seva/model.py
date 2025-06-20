from dataclasses import dataclass, field

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from seva.modules.layers import (
    Downsample,
    GroupNorm32,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
    timestep_embedding,
)
from seva.modules.transformer import MultiviewTransformer
from typing import Union

@dataclass
class SevaParams(object):
    in_channels: int = 11
    model_channels: int = 320
    out_channels: int = 4
    num_frames: int = 21
    num_res_blocks: int = 2
    attention_resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    channel_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_head_channels: int = 64
    transformer_depth: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    context_dim: int = 1024
    dense_in_channels: int = 6
    dropout: float = 0.0
    unflatten_names: list[str] = field(
        default_factory=lambda: ["middle_ds8", "output_ds4", "output_ds2"]
    )
    ckpt_path: str | None = None

    def __post_init__(self):
        assert len(self.channel_mult) == len(self.transformer_depth)


class Seva(nn.Module):
    def __init__(self, params: SevaParams, freeze_layers:bool=False) -> None:
        super().__init__()
        self.params = params
        self.model_channels = params.model_channels
        self.out_channels = params.out_channels
        self.num_head_channels = params.num_head_channels

        time_embed_dim = params.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(params.model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(params.in_channels, params.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = params.model_channels
        input_block_chans = [params.model_channels]
        ch = params.model_channels
        ds = 1
        for level, mult in enumerate(params.channel_mult):
            for _ in range(params.num_res_blocks):
                input_layers: list[ResBlock | MultiviewTransformer | Downsample] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        out_channels=mult * params.model_channels,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = mult * params.model_channels
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels
                    input_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"input_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*input_layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(params.channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        num_heads = ch // params.num_head_channels
        dim_head = params.num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
            MultiviewTransformer(
                ch,
                num_heads,
                dim_head,
                name=f"middle_ds{ds}",
                depth=params.transformer_depth[-1],
                context_dim=params.context_dim,
                unflatten_names=params.unflatten_names,
            ),
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                out_channels=None,
                dense_in_channels=params.dense_in_channels,
                dropout=params.dropout,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(params.channel_mult))[::-1]:
            for i in range(params.num_res_blocks + 1):
                ich = input_block_chans.pop()
                output_layers: list[ResBlock | MultiviewTransformer | Upsample] = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        out_channels=params.model_channels * mult,
                        dense_in_channels=params.dense_in_channels,
                        dropout=params.dropout,
                    )
                ]
                ch = params.model_channels * mult
                if ds in params.attention_resolutions:
                    num_heads = ch // params.num_head_channels
                    dim_head = params.num_head_channels

                    output_layers.append(
                        MultiviewTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            name=f"output_ds{ds}",
                            depth=params.transformer_depth[level],
                            context_dim=params.context_dim,
                            unflatten_names=params.unflatten_names,
                        )
                    )
                if level and i == params.num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    output_layers.append(Upsample(ch, out_ch))
                self.output_blocks.append(TimestepEmbedSequential(*output_layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, params.out_channels, 3, padding=1),
        )
        if params.ckpt_path is not None:
            from safetensors.torch import load_file
            from seva.utils import print_load_warning
            state_dict = load_file(params.ckpt_path)
            missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)
            print_load_warning(missing, unexpected)

        # if freeze_layers:
        #     print("layers of seva are FROZEN")
        #     for param in self.input_blocks.parameters():
        #         param.requires_grad = False
        #     for param in self.middle_block.parameters():
        #         param.requires_grad = False
        #     for param in self.output_blocks.parameters():
        #         param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        dense_y: torch.Tensor,
        num_frames: int | None = None,
    ) -> torch.Tensor:
        num_frames = num_frames or self.params.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
        h = h.type(x.dtype)
        print("SEVA OUTPUT SHAPE: ", self.out(h).shape)
        return self.out(h) # [B*num_images, C=4, H=72, W=72]

# for compatibility with SGM
class SGMWrapper(nn.Module):
    def __init__(self, module: Seva): # or SevaLoRAWrappers
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.module(
            x,
            t=t,
            y=c["crossattn"],
            dense_y=c["dense_vector"],
            **kwargs,
        )


# class SevaLightningModule(pl.LightningModule):
#     def __init__(self, seva_model: Seva):
#         super().__init__()
#         self.model = seva_model
        
#     def forward(self, x, t, y, dense_y, num_frames=None):
#         return self.model(x, t, y, dense_y, num_frames)

    # this is never used or reached
    # @rank_zero_only
    # def log_images(self, input_tensor, output_tensor, target_tensor):
    #     print("within SevaLightningModule::log_images!")
    #     # Assume [B,C,H,W] and normalize to [0,1] for wandb.Image
    #     images = []

    #     for i in range(min(4, input_tensor.size(0))):  # limit to first 4 images
    #         img_grid = torchvision.utils.make_grid([
    #             input_tensor[i].detach().cpu(),
    #             output_tensor[i].detach().cpu(),
    #             target_tensor[i].detach().cpu(),
    #         ], nrow=3, normalize=True, scale_each=True)

    #         images.append(wandb.Image(img_grid, caption=f"Sample {i}"))

    #     self.logger.experiment.log({"val/reconstructions": images, "global_step": self.global_step})