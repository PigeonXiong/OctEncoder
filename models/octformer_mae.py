import torch
import torch.nn as nn
import ocnn
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

from .octformer import OctFormer
from ocnn.octree import Octree

class OctFormerMAE(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: List[int] = [96, 192, 384, 384],
            num_blocks: List[int] = [2, 2, 18, 2],
            num_heads: List[int] = [6, 12, 24, 24],
            patch_size: int = 32,
            dilation: int = 4,
            drop_path: float = 0.5,
            nempty: bool = True,
            stem_down: int = 2,
            mask_ratio: float = 0.75,
            decoder_depth: int = 4,
            decoder_num_heads: int = 8,
            decoder_dim: int = 256):
        super().__init__()
        
        # Encoder
        self.encoder = OctFormer(
            in_channels, channels, num_blocks, num_heads, patch_size,
            dilation, drop_path, nempty, stem_down)
        print(f"-----Encoder initialized with channels: {channels}, num_blocks: {num_blocks}, num_heads: {num_heads}, patch_size: {patch_size}, dilation: {dilation}, drop_path: {drop_path}, nempty: {nempty}, stem_down: {stem_down}-----")
        # Parameters
        self.mask_ratio = mask_ratio
        self.decoder_dim = decoder_dim
        
        # Decoder
        encoder_last_dim = channels[-1]
        self.decoder_embed = nn.Linear(encoder_last_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Decoder transformer blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_dim * 4,
            activation='gelu',
            batch_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, decoder_depth)
        
        # Final prediction head
        self.decoder_pred = nn.Linear(decoder_dim, in_channels)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking following PointMAE approach
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # Number of tokens to keep
        
        # For each sample in the batch
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Generate mask directly using scatter
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_(1, ids_keep, 0)
        
        # Gather the kept tokens
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, data: torch.Tensor, octree: Octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get features from backbone
        features = self.encoder(data, octree, depth)
        
        # Get the last layer features for masking
        final_depth = depth - self.encoder.stem_down - (self.encoder.num_stages - 1)
        x = features[final_depth]  # Get features from the final depth level
        
        # Get the feature dimensions
        B = octree.batch_size
        F = x.shape[0]  # total features
        C = self.encoder.channels[-1]  # channel dimension from last stage
        
        # Calculate points per batch, ensuring even distribution
        points_per_batch = (F + B - 1) // B  # Ceiling division
        
        # Pad features if needed
        pad_size = points_per_batch * B - F
        if pad_size > 0:
            padding = torch.zeros(pad_size, C, device=x.device)
            x = torch.cat([x, padding], dim=0)
            
        # Reshape features to [B, N, C] format
        x = x.view(B, points_per_batch, C)
        
        # Only apply masking if we have enough tokens
        if points_per_batch < 4:
            return x, torch.zeros(B, points_per_batch, device=x.device), torch.arange(points_per_batch, device=x.device).expand(B, points_per_batch)
        
        # Apply random masking with original mask ratio
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        return x_masked, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        # Embed tokens - input shape should be [N, L, C]
        x = self.decoder_embed(x)  # [N, L, decoder_dim]
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # [N, L+M, decoder_dim]
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Apply Transformer decoder
        x = self.decoder(x, x)
        
        # Predictor projection
        x = self.decoder_pred(x)  # [N, L, in_channels]
        
        return x

    def forward(self, data: torch.Tensor, octree: Octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Store original input for loss computation
        orig_data = data.clone()
        B = octree.batch_size
        
        # Forward through encoder with original data
        latent, mask, ids_restore = self.forward_encoder(data, octree, depth)
        
        # Forward through decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Get number of points per batch from prediction shape
        points_per_batch = pred.shape[1]
        total_points = orig_data.shape[0]
        
        # Reshape and pad original data to match prediction shape
        pad_size = points_per_batch * B - total_points
        if pad_size > 0:
            padding = torch.zeros(pad_size, orig_data.shape[1], device=orig_data.device)
            orig_data_padded = torch.cat([orig_data, padding], dim=0)
        else:
            orig_data_padded = orig_data[:points_per_batch * B]
            
        # Reshape to match prediction shape
        data_reshaped = orig_data_padded.view(B, points_per_batch, -1)
        
        # Calculate reconstruction loss only on masked patches
        loss = self.compute_loss(data_reshaped, pred, mask)
        
        return loss, pred

    def compute_chamfer_distance(self, xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional Chamfer Distance between two point clouds
        xyz1, xyz2: [B, N, 3]
        """
        dist1, dist2 = self.get_chamfer_distance(xyz1, xyz2)
        # Compute mean distances for each point
        loss_cd = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        return loss_cd

    def get_chamfer_distance(self, xyz1: torch.Tensor, xyz2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate nearest neighbor distances between two point clouds
        """
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        
        dist1 = torch.cdist(xyz1, xyz2)  # [B, N, M]
        dist2 = torch.cdist(xyz2, xyz1)  # [B, M, N]
        
        min_dist1, _ = torch.min(dist1, dim=2)  # [B, N]
        min_dist2, _ = torch.min(dist2, dim=2)  # [B, M]
        
        return min_dist1, min_dist2

    def compute_loss(self, target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss using Chamfer Distance for masked tokens,
        similar to PointMAE approach
        """
        # Get batch size and number of points
        B, N, C = target.shape
        
        # Split into masked and unmasked points based on mask
        masked_indices = torch.where(mask == 1)
        unmasked_indices = torch.where(mask == 0)
        
        # Reshape predictions and targets for masked points
        masked_pred = pred[masked_indices[0], masked_indices[1]]
        masked_target = target[masked_indices[0], masked_indices[1]]
        
        # Reshape to [B, N, 3] format for Chamfer Distance
        num_masked_per_batch = mask.sum(dim=1)
        max_masked = int(num_masked_per_batch.max().item())  # Convert to integer
        
        if max_masked == 0:
            # If no points are masked, return zero loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Pad masked points to same length
        masked_pred_padded = torch.zeros(B, max_masked, C, device=pred.device)
        masked_target_padded = torch.zeros(B, max_masked, C, device=target.device)
        
        # Fill in the actual points
        curr_idx = 0
        for b in range(B):
            # Get indices for current batch
            batch_indices = (masked_indices[0] == b)
            curr_masked = int(num_masked_per_batch[b].item())
            
            if curr_masked > 0:
                # Get points for current batch
                batch_pred = masked_pred[batch_indices]
                batch_target = masked_target[batch_indices]
                
                # Fill padded tensors
                masked_pred_padded[b, :curr_masked] = batch_pred
                masked_target_padded[b, :curr_masked] = batch_target
        
        # Compute Chamfer Distance loss for masked points (only on xyz coordinates)
        cd_loss = self.compute_chamfer_distance(
            masked_pred_padded[..., :3], 
            masked_target_padded[..., :3]
        )
        
        # If we have additional features beyond XYZ, add feature reconstruction loss
        if C > 3:
            # Compute feature reconstruction loss only on valid (non-padded) points
            feat_pred = masked_pred[..., 3:]
            feat_target = masked_target[..., 3:]
            feat_diff = torch.clamp(feat_pred - feat_target, min=-100, max=100)
            feat_loss = (feat_diff * feat_diff).mean()
            total_loss = cd_loss.mean() + 0.1 * feat_loss  # Weight feature loss less than CD loss
        else:
            total_loss = cd_loss.mean()
        
        return total_loss 