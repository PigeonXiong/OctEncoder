import ocnn
import torch
from ocnn.octree import Octree
from typing import Optional, List, Dict
from .octformer import OctFormer
import torch.nn as nn

from typing import Dict, List, Tuple


class OctFormerMAE(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            channels: List[int] = [96, 192, 384, 384],
            num_blocks: List[int] = [2, 2, 18, 2],
            num_heads: List[int] = [6, 12, 24, 24],
            patch_size: int = 32,
            dilation: int = 4,
            mask_ratio: float = 0.75,
            decoder_dim: int = 256,
            decoder_depth: int = 4,
            decoder_heads: int = 8,
            nempty: bool = True,
            stem_down: int = 2,
            **kwargs):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = OctFormer(
            in_channels, channels, num_blocks, num_heads,
            patch_size, dilation, 0.0, nempty, stem_down)
            
        # Decoder
        self.decoder_embed = nn.Linear(channels[-1], decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_blocks = nn.ModuleList([
            OctreeTransformerBlock(
                decoder_dim, decoder_heads, mlp_ratio=4.0,
                qkv_bias=True, drop=0.0, attn_drop=0.0,
                drop_path=0.0, norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, in_channels, bias=True)
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def random_masking(self, x: torch.Tensor, octree: Octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        N = x.shape[0]  # number of leaf nodes
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)
        
        ids_keep = ids_shuffle[:len_keep]
        x_masked = torch.gather(x, dim=0, index=ids_keep.unsqueeze(-1).repeat(1, x.shape[1]))
        
        mask = torch.ones([N], device=x.device)
        mask[:len_keep] = 0
        mask = torch.gather(mask, dim=0, index=ids_restore)
        
        return x_masked, mask, ids_restore
        
    def forward_encoder(self, x: torch.Tensor, octree: Octree, depth: int) -> Dict[str, torch.Tensor]:
        x_masked, mask, ids_restore = self.random_masking(x, octree, depth)
        
        masked_octree = octree.clone()
        masked_octree.features[depth] = x_masked
        
        latent = self.encoder(x_masked, masked_octree, depth)
        
        return latent, mask, ids_restore
        
    def forward_decoder(self, latent: Dict[int, torch.Tensor], mask: torch.Tensor, 
                       ids_restore: torch.Tensor) -> torch.Tensor:
        x = latent[max(latent.keys())]
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], 1, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        
        x_ = self.decoder_pred(x_)
        x_ = x_[:, :x.shape[1], :]
        
        x_ = torch.gather(x_, dim=1, 
                         index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[-1]))
        
        return x_
        
    def forward(self, x: torch.Tensor, octree: Octree, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(x, octree, depth)
        pred = self.forward_decoder(latent, mask, ids_restore)
        return pred, mask

class OctreeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ocnn.modules.OctreeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = ocnn.modules.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ocnn.modules.Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x