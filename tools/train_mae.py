# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds - MAE Pre-training
# --------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import logging
import ocnn
from ocnn.octree import Octree
import yaml
from easydict import EasyDict
from types import SimpleNamespace

sys.path.append('.')  # Add current directory to path
from datasets.tetmesh_mae import get_tetmesh_dataset_mae
from models.octformer_mae import OctFormerMAE


def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='OctFormer MAE Pre-training')
    parser.add_argument('--cfg',
                      help='config file',
                      default='configs/mae_tet.yaml',
                      type=str)
    args = parser.parse_args()
    return args


def get_model(flags):
    model = OctFormerMAE(
        in_channels=flags.MODEL.in_channels,
        mask_ratio=flags.MODEL.mask_ratio,
        channels=flags.MODEL.channels,
        num_blocks=flags.MODEL.num_blocks,
        num_heads=flags.MODEL.num_heads,
        patch_size=flags.MODEL.patch_size,
        dilation=flags.MODEL.dilation,
        drop_path=flags.MODEL.drop_path,
        nempty=flags.MODEL.nempty,
        stem_down=flags.MODEL.stem_down,
        decoder_depth=flags.MODEL.decoder_depth,
        decoder_num_heads=flags.MODEL.decoder_num_heads,
        decoder_dim=flags.MODEL.decoder_dim
    )
    return model


def get_solver(flags, model):
    if flags.SOLVER.type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=flags.SOLVER.lr,
            momentum=0.9,
            weight_decay=flags.SOLVER.weight_decay)
    elif flags.SOLVER.type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=flags.SOLVER.lr,
            weight_decay=flags.SOLVER.weight_decay)
    elif flags.SOLVER.type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=flags.SOLVER.lr,
            weight_decay=flags.SOLVER.weight_decay)
    else:
        raise ValueError
    return optimizer


def cosine_lr_scheduler(optimizer, epoch, flags):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < flags.SOLVER.warmup_epochs:
        lr = flags.SOLVER.lr * epoch / flags.SOLVER.warmup_epochs 
    else:
        lr = flags.SOLVER.lr * 0.5 * (1. + torch.cos(
            torch.pi * (epoch - flags.SOLVER.warmup_epochs) / 
            (flags.SOLVER.epochs - flags.SOLVER.warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, train_loader, optimizer, epoch, flags, device):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        # Debug print for first batch
        if batch_idx == 0:
            logging.info(f"Batch keys: {batch.keys()}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logging.info(f"Key: {k}, Shape: {v.shape}, Type: {v.dtype}")
                else:
                    logging.info(f"Key: {k}, Type: {type(v)}")
        
        # Get data from batch
        octree = batch['octree'].to(device)
        octree2 = batch['octree2'].to(device)
        depth = octree.depth
        
        # Get input features
        data = ocnn.modules.InputFeature(flags.MODEL.feature, flags.MODEL.nempty)(octree)
        data2 = ocnn.modules.InputFeature(flags.MODEL.feature, flags.MODEL.nempty)(octree2)
        
        # Forward pass
        loss, _ = model(data, octree, depth)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % flags.SOLVER.log_per_iter == 0:
            logging.info(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, '
                        f'Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    args = parse_args()
    
    # Load config
    with open(args.cfg) as f:
        flags = EasyDict(yaml.safe_load(f))
    
    # Setup logging and output directory
    os.makedirs(flags.SOLVER.logdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(flags.SOLVER.logdir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log the config
    logging.info('Config:\n' + str(flags))
    
    # Setup device
    device = torch.device(f'cuda:{flags.SOLVER.gpu}' 
                         if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model(flags)
    model = model.to(device)
    
    # Setup optimizer
    optimizer = get_solver(flags, model)
    
    # Setup data loader
    train_dataset, collate_fn = get_tetmesh_dataset_mae(flags.DATA.train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=flags.DATA.train.batch_size,
        shuffle=flags.DATA.train.shuffle,
        num_workers=flags.DATA.train.num_workers if hasattr(flags.DATA.train, 'num_workers') else 0,
        collate_fn=collate_fn,
        pin_memory=flags.DATA.train.pin_memory if hasattr(flags.DATA.train, 'pin_memory') else True
    )
    
    # Training loop
    for epoch in range(flags.SOLVER.epochs):
        # Adjust learning rate
        lr = cosine_lr_scheduler(optimizer, epoch, flags)
        logging.info(f'Epoch: {epoch}, Learning Rate: {lr:.6f}')
        
        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, flags, device)
        
        logging.info(f'Epoch: {epoch}, Average Loss: {train_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % flags.SOLVER.test_every_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'flags': flags,
            }
            torch.save(
                checkpoint,
                os.path.join(flags.SOLVER.logdir, f'checkpoint_{epoch:04d}.pth')
            )

if __name__ == '__main__':
    main() 