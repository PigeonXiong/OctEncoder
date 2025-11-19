# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds - MAE Pre-training
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import ocnn
import sys
import os

from thsolver import Solver
from builder import get_mae_model
import builder


class MAESolver(Solver):
    def get_model(self, flags):
        return get_mae_model(flags)

    def get_dataset(self, flags):
        return builder.get_segmentation_dataset(flags)

    def get_input_feature(self, octree):
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data
    
    def process_batch(self, batch, flags):
        def points2octree(points):
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['points'] = ocnn.octree.merge_points(points)
            batch['octree'] = octree
        return batch

    def forward(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        octree = batch['octree']
        
        # Get input features for both octrees
        data = self.get_input_feature(octree)
        
        # Forward pass with first octree (we can alternate or combine them in future)
        loss, pred = self.model(data, octree, octree.depth)
        return loss, pred

    def train_step(self, batch):
        """Performs a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        loss, _ = self.forward(batch)
        
        # Return the loss without calling backward
        # The training loop will handle the backward pass
        return {'train/loss': loss}

    def test_step(self, batch):
        with torch.no_grad():
            loss, _ = self.forward(batch)
        return {'test/loss': loss}


if __name__ == "__main__":
    # Add the root directory to Python path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    MAESolver.main() 