import os
import torch
import torch.nn as nn
import numpy as np

from archs.unet import UNet2DModel

class SvbrdfEstimator(nn.Module):
    def __init__(self, proj_dim, use_rgb_input, num_norm_groups, nogeom=False):
        super().__init__()

        # Greyscale only
        if nogeom:
            in_channels = 1
        # Greyscale+Depth+Normals
        elif use_rgb_input:
            in_channels = 5
        # Depth+Normals
        else:
            in_channels = 4

        self.model = UNet2DModel(
            in_channels = in_channels,
            out_channels = 5,
            down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            # block_in_channels = (16, 32, 64, 128),
            # block_out_channels = (128, 64, 32, 16),
            layers_per_block = 2,
            mid_block_scale_factor = 1,
            downsample_padding = 1,
            downsample_type = "conv",
            upsample_type = "conv",
            dropout = 0.0,
            act_fn = "silu",
            attention_head_dim = 0,
            norm_num_groups = num_norm_groups,
            attn_norm_num_groups = None,
            # norm_eps = 1e-5,
            # resnet_time_scale_shift = "default",
            add_attention = False,
            proj_dim=proj_dim
        )

    def forward(self, hf_tensor, inputs_list):

        input = torch.cat(inputs_list, axis=1)

        result = self.model(input, hf_tensor)

        result = 0.5 * result.clamp(-1, 1) + 0.5

        return result
        
    
