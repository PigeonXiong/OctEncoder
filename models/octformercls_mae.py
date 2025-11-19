# abandoned

import torch
import torch.nn as nn
from models.octformer_mae import OctFormerMAE
from models.octformercls import ClsHeader
from ocnn.octree import Octree
from typing import List


class OctFormerClsMAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 channels: List[int] = [96, 192, 384, 384],
                 num_blocks: List[int] = [2, 2, 18, 2],
                 num_heads: List[int] = [6, 12, 24, 24],
                 patch_size: int = 32, dilation: int = 4,
                 drop_path: float = 0.5, nempty: bool = True,
                 stem_down: int = 2, head_drop: float = 0.5):
        super().__init__()

        # Encoders (MAE, only using encoder part)
        self.encoder1 = OctFormerMAE(
            in_channels, channels, num_blocks, num_heads,
            patch_size, dilation, drop_path, nempty, stem_down
        ).encoder
        self.encoder2 = OctFormerMAE(
            in_channels, channels, num_blocks, num_heads,
            patch_size, dilation, drop_path, nempty, stem_down
        ).encoder

        # Store feature stage info
        self.final_depth_offset = stem_down + (len(channels) - 1)
        self.out_channels = channels[-1]

        # Classifier head (gating + fc)
        self.head = ClsHeader(
            out_channels=out_channels,
            in_channels=self.out_channels,
            nempty=nempty,
            dropout=head_drop
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data:torch.Tensor, data2:torch.Tensor, octree:Octree, octree2:Octree, depth:int, depth2:int, ptau:torch.Tensor):
        # Encode both views
        feat1 = self.encoder1(data, octree, depth)
        feat2 = self.encoder2(data2, octree2, depth2)
        print(f"------feat1shape:{feat1.shape}, feat2shape: {feat2.shape}------")

        # Extract features at final stage
        level1 = depth - self.final_depth_offset
        level2 = depth2 - self.final_depth_offset
        x1 = feat1[level1]
        x2 = feat2[level2]
        print(f"------Depth1: {level1}, Depth2: {level2}, x1shape: {x1.shape}, x2shape: {x2.shape}------")
        # Classification
        return self.head(x1, x2, octree, octree2, level1, level2, ptau)