import torch
import numpy as np

from thsolver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch
from .utils import ReadPly, Transform


class TetTransformMAE(Transform):
    def preprocess(self, sample: dict, idx: int):
        points = super().preprocess(sample, idx)
        # Normalize points
        bbmin, bbmax = points.bbox()
        points.normalize(bbmin, bbmax, scale=0.8)
        points.scale(torch.Tensor([0.8, 0.8, 0.8]))
        return points


def read_file(filename: str):
    filename = filename.replace('\\', '/')
    if filename.endswith('.ply'):
        read_ply = ReadPly(has_normal=True)
        return read_ply(filename)
    elif filename.endswith('.npz'):
        raw = np.load(filename)
        output = {'points': raw['points'], 'normals': raw['normals']}
        return output
    else:
        raise ValueError


def get_tetmesh_dataset_mae(flags):
    transform = TetTransformMAE(flags)
    transform2 = TetTransformMAE(flags)
    collate_batch = CollateBatch()
    
    dataset = Dataset(
        flags.location, flags.location2,
        flags.filelist, flags.filelist,
        transform, transform2,
        read_file=read_file,
        take=flags.take if hasattr(flags, 'take') else -1)
    
    return dataset, collate_batch 