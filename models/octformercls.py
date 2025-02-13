# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# ------------------------------------------------------

import ocnn
import torch
import torch.nn as nn
from ocnn.octree import Octree
from typing import List

from .octformer import OctFormer


class ClsHeader(nn.Module):
  def __init__(self, out_channels: int, in_channels: int,
                nempty: bool = False, dropout: float = 0.5):
      super().__init__()
      self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
      # Increase input dimension by 1 to account for the ptau scalar
      self.cls_header = nn.Sequential(
          ocnn.modules.FcBnRelu(in_channels + 1, 128),
          nn.Dropout(p=dropout),
          nn.Linear(128, out_channels)
      )

  def forward(self, data: torch.Tensor, octree: Octree, depth: int, ptau: torch.Tensor):
      # Global pooling: produces a feature vector of shape [batch_size, in_channels]
      pooled = self.global_pool(data, octree, depth)
      # Ensure ptau is shape [batch_size, 1]
      if ptau.dim() == 1:
          ptau = ptau.unsqueeze(1)
      # Concatenate pooled features with ptau along the feature dimension.
      combined = torch.cat([pooled, ptau.float()], dim=1)
      logit = self.cls_header(combined)
      return logit


class OctFormerCls(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int,
               channels: List[int] = [3200, 3200, 6400, 64],
               num_blocks: List[int] = [2, 2, 2, 2],
               num_heads: List[int] = [6, 6, 6, 6],
               patch_size: int = 0, dilation: int = 0,
               drop_path: float = 0.5, nempty: bool = True,
               stem_down: int = 2, head_drop: float = 0.5, **kwargs):
    super().__init__()
    self.biomarker_fc = nn.Linear(1, 1) 
    self.fc = nn.Linear(out_channels + 1, out_channels)
    self.backbone = OctFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
    self.head = ClsHeader(
        out_channels, channels[-1], nempty, head_drop)
    self.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.trunc_normal_(m.weight, std=0.02)
      if isinstance(m, torch.nn.Linear) and m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int, ptau:torch.Tensor):
    features = self.backbone(data, octree, depth)
    curr_depth = min(features.keys())
    output = self.head(features[curr_depth], octree, curr_depth, ptau)
    # print(output.shape)
    # if ptau.dim() == 1:
    #     ptau = ptau.unsqueeze(1)
    # Process the 1D ptau using the biomarker fully connected layer.
    # biomarker_features = self.biomarker_fc(ptau.float())  # Shape: [batch_size, 1]

    # # Concatenate the head output with the ptau features.
    # combined_features = torch.cat([output, biomarker_features], dim=1)  # Shape: [batch_size, out_channels + 1]
    # # Pass the combined features through the final fully connected layer.
    # final_logits = self.fc(combined_features)
    # print(final_logits.shape)
    return output
