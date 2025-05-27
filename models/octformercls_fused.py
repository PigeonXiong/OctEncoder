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
    def __init__(self, out_channels: int, in_channels: int, nempty: bool = True, dropout: float = 0.5):
        super().__init__()
        self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
        
        # Project both feature sets to the same embedding space with normalization
        self.proj1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        self.proj2 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        
        # Gating mechanism: learn weights for each feature vector
        self.gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classifier after fusion
        self.cls_header = nn.Sequential(
            ocnn.modules.FcBnRelu(in_channels, 128),
            nn.Dropout(p=dropout),
            nn.Linear(128, out_channels)
        )

    def forward(self, data: torch.Tensor, data2: torch.Tensor, octree: ocnn.octree.Octree, 
                octree2: ocnn.octree.Octree, depth: int, depth2: int, ptau: torch.Tensor):
        # Global pooling to get features from each branch
        pooled1 = self.global_pool(data, octree, depth)   # shape: [B, in_channels]
        pooled2 = self.global_pool(data2, octree2, depth2)  # shape: [B, in_channels]
        
        # Project and normalize features
        f1 = self.proj1(pooled1)
        f2 = self.proj2(pooled2)
        
        # Learn gating weights
        gate_input = torch.cat([f1, f2], dim=1)  # shape: [B, 2*in_channels]
        weights = self.gate(gate_input)           # shape: [B, 2]
        w1, w2 = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        
        # Fuse features using the learned weights (weighted sum)
        fused = w1 * f1 + w2 * f2
        
        # Optionally, you can concatenate ptau if needed:
        # fused = torch.cat([fused, ptau.float().unsqueeze(1)], dim=1)
        
        logit = self.cls_header(pooled1)
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
    # print('dilation::::::::',dilation)
    self.backbone = OctFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
    self.backbone2= OctFormer(
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

  def forward(self, data: torch.Tensor,  data2: torch.Tensor, octree: Octree,octree2: Octree, depth: int, depth2: int,  ptau: torch.Tensor):
    features = self.backbone(data, octree, depth)
    features2 = self.backbone(data2, octree2, depth2)
    curr_depth = min(features.keys())
    curr_depth2 = min(features2.keys())
    output = self.head(features[curr_depth], features2[curr_depth2],octree,octree2, curr_depth,curr_depth2, ptau)
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
