import torch
import numpy as np

from thsolver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch
import pdb
from .utils import ReadPly, Transform



class TetTransform(Transform):
        # def __init__(self, flags):
        # # Convert SimpleNamespace to a dictionary before passing to super
        # super().__init__(**vars(flags))
        # self.flags = flags

    def preprocess(self, sample: dict, idx: int):
        points = super().preprocess(sample, idx)
        # Uncomment and adjust these lines if normalization is needed.
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


def get_tetmesh_dataset(flags):
  # print(type(flags))
  transform = TetTransform(flags)
  transform2 = TetTransform(flags)
  collate_batch = CollateBatch()
  # print(flags.adaptive)
  dataset = Dataset(flags.location,flags.location2, flags.filelist,flags.filelist, transform,transform2,
                    read_file=read_file, take=flags.take)
  # print(dataset[1])
  # pdb.set_trace()
  return dataset, collate_batch