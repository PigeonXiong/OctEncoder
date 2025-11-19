# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import datasets
import models
from models.octformer_mae import OctFormerMAE
from models.octformerseg import OctFormerSeg
from models.octformer import OctFormer
#from models.octformercls_mae import OctFormerClsMAE
import os


def octsegformer_large(in_channels, out_channels, **kwargs):
  return models.OctFormerSeg(
      in_channels, out_channels,
      channels=[192, 384, 768, 768],
      num_blocks=[2, 2, 18, 2],
      num_heads=[12, 24, 48, 48],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer(in_channels, out_channels, **kwargs):
  return models.OctFormerSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 18, 2],
      num_heads=[6, 12, 24, 24],
      patch_size=64, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer_small(in_channels, out_channels, **kwargs):
  return models.OctFormerSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 6, 2],
      num_heads=[6, 12, 24, 24],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octformer_cls(in_channels, out_channels, nemtpy, **kwargs):

  return models.OctFormerCls(
      in_channels, out_channels,
      channels=kwargs.get('channels', [64, 64]),
      num_blocks=kwargs.get('num_blocks', [1, 1]),
      num_heads=kwargs.get('num_heads', [16, 16]),
      patch_size=kwargs.get('patch_size', 64),
      dilation=kwargs.get('dilation', 4),
      drop_path=kwargs.get('drop_path', 0.0),
      nempty=nemtpy,
      stem_down=kwargs.get('stem_down', 2),
      head_drop=kwargs.get('head_drop', 0.0))


def get_segmentation_model(flags):
  if flags.name.lower() == 'octformerseg_pretrain_mae':
      return get_octformerseg_pretrain_mae(flags)
  params = {
      'in_channels': flags.channel, 'out_channels': flags.nout,
      'interp': flags.interp, 'nempty': flags.nempty,
  }
  networks = {
      'octsegformer': octsegformer,
      'octsegformer_large': octsegformer_large,
      'octsegformer_small': octsegformer_small,
  }

  return networks[flags.name.lower()](**params)


def get_classification_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.models.LeNet(
        flags.channel, flags.nout, flags.stages, flags.nempty)
  elif flags.name.lower() == 'hrnet':
    model = ocnn.models.HRNet(
        flags.channel, flags.nout, flags.stages, nempty=flags.nempty)
  elif flags.name.lower() == 'octformercls':
    # print(flags)
    model = octformer_cls(flags.channel, flags.nout, flags.nempty)
  elif flags.name.lower() == 'octformer':
    model = OctFormer(**flags.MODEL)
  elif flags.name.lower() == 'octformercls_pretrain_mae':
    model = get_octformercls_pretrain_mae(flags)
  else:
    raise ValueError
  return model


def get_segmentation_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_seg_dataset(flags)
  elif flags.name.lower() == 'scannet':
    return datasets.get_scannet_dataset(flags)
  elif flags.name.lower() == 'kitti':
    return datasets.get_kitti_dataset(flags)
  else:
    raise ValueError


# def octformer_mae(in_channels, **kwargs):
#     return OctFormerMAE(
#         in_channels=in_channels,
#         channels=[96, 192, 384, 384],
#         num_blocks=[2, 2, 18, 2],
#         num_heads=[6, 12, 24, 24],
#         patch_size=32,
#         dilation=4,
#         drop_path=0.5,
#         nempty=True,
#         stem_down=2,
#         mask_ratio=0.75,
#         decoder_depth=4,
#         decoder_num_heads=8,
#         decoder_dim=256
#     )


def get_mae_model(flags):
    if flags.name.lower() == 'octformermae':
        # Get all parameters from config, with defaults if not specified
        params = {
            'in_channels': getattr(flags, 'in_channels', flags.channel),
            'channels': getattr(flags, 'channels', [96, 192, 384, 384]),
            'num_blocks': getattr(flags, 'num_blocks', [2, 2, 18, 2]),
            'num_heads': getattr(flags, 'num_heads', [6, 12, 24, 24]),
            'mask_ratio': getattr(flags, 'mask_ratio', 0.25),
            'decoder_depth': getattr(flags, 'decoder_depth', 4),
            'decoder_num_heads': getattr(flags, 'decoder_num_heads', 8),
            'decoder_dim': getattr(flags, 'decoder_dim', 256),
            'patch_size': getattr(flags, 'patch_size', 64),
            'dilation': getattr(flags, 'dilation', 4),
            'drop_path': getattr(flags, 'drop_path', 0.5),
            'stem_down': getattr(flags, 'stem_down', 2),
            'nempty': getattr(flags, 'nempty', True)
        }
        model = OctFormerMAE(**params)
    else:
        raise ValueError(f'Model {flags.name} not supported')
    return model


def get_octformercls_pretrain_mae(flags):
    # 1. Build the standard classifier

    model = octformer_cls(flags.channel, flags.nout, flags.nempty, **flags)

    # 2. Load MAE encoder checkpoint
    mae_ckpt_path = getattr(flags, 'mae_ckpt', None)
    if mae_ckpt_path and os.path.exists(mae_ckpt_path):
        print(f"-----Loading MAE pretrained encoder from: {mae_ckpt_path}")
        ckpt = torch.load(mae_ckpt_path, map_location='cpu')

        if 'model' in ckpt:  # PyTorch Lightning / standard save
            ckpt = ckpt['model']

        # Filter to encoder weights only (assuming saved as encoder.<...>)
        encoder_state_dict = {
            k.replace("encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }

        missing, unexpected = model.backbone.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded MAE encoder into backbone. Missing: {missing}, Unexpected: {unexpected}")

        # Optionally load into backbone2 (shared or not)
        missing2, unexpected2 = model.backbone2.load_state_dict(encoder_state_dict, strict=False)
        print(f"Also loaded into backbone2. Missing: {missing2}, Unexpected: {unexpected2}")
    else:
        print(f"MAE checkpoint not provided or not found: {mae_ckpt_path}")
    
    # 3. Freeze the encoder layers
    if getattr(flags, 'freeze_encoder', False):
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone2.parameters():
            param.requires_grad = False
        print("Encoder layers frozen. Only classifier will be trained.")

    return model

def get_octformerseg_pretrain_mae(flags):
    # 1. Build the standard segmentation model
    model = octsegformer(flags.channel, flags.nout, **flags)

    # 2. Load MAE encoder checkpoint
    mae_ckpt_path = getattr(flags, 'mae_ckpt', None)
    if mae_ckpt_path and os.path.exists(mae_ckpt_path):
        print(f"-----Loading MAE pretrained encoder from: {mae_ckpt_path}")
        ckpt = torch.load(mae_ckpt_path, map_location='cpu')

        if 'model' in ckpt:  # PyTorch Lightning / standard save
            ckpt = ckpt['model']

        # Filter to encoder weights only (assuming saved as encoder.<...>)
        encoder_state_dict = {
            k.replace("encoder.", ""): v
            for k, v in ckpt.items()
            if k.startswith("encoder.")
        }

        missing, unexpected = model.backbone.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded MAE encoder into backbone. Missing: {missing}, Unexpected: {unexpected}")
    else:
        print(f"MAE checkpoint not provided or not found: {mae_ckpt_path}")
    
    # 3. Freeze the encoder layers
    if getattr(flags, 'freeze_encoder', True):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Encoder layers frozen. Only classifier will be trained.")

    return model