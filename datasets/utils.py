# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import numpy as np
from plyfile import PlyData


class ReadPly:
    def __init__(self, has_normal: bool = True, has_color: bool = False,
                 has_label: bool = False):
        self.has_normal = has_normal
        self.has_color = has_color
        self.has_label = has_label

    def __call__(self, filename: str):
        plydata = PlyData.read(filename)
        vtx = plydata['vertex']
        # print(vtx.data.dtype.names)
        output = dict()
        points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
        output['points'] = points.astype(np.float32)
        
        if self.has_normal:
            normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
            output['normals'] = normal.astype(np.float32)
        
        if self.has_color:
          if 'red' in vtx.data.dtype.names and 'green' in vtx.data.dtype.names and 'blue' in vtx.data.dtype.names:
            # If RGB color channels are present
            color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
            output['colors'] = color.astype(np.float32)
          else:
            # If color channels are not present, set to empty array
            output['colors'] = np.zeros((points.shape[0], 3), dtype=np.float32)
        
        if self.has_label:
            # Now reading the ptau field, which is a float.
            if 'ptau' in vtx.data.dtype.names:
                ptau = vtx['ptau']
                output['labels'] = ptau.astype(np.float32)
            elif 'label' in vtx.data.dtype.names:
                # If 'ptau' is not available, fall back to 'label'
                label = vtx['label'] - 1
                label = np.clip(label, 0, 7)  # Assuming labels are in range [0, 7]
                output['labels'] = label.astype(np.int32)
        # output['ptau'] = vtx['ptau']
        
        return output

class ReadPly2:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.has_normal = has_normal
    self.has_color = has_color
    self.has_label = has_label

  def __call__(self, filename: str):
    try:
      raw = np.load(filename)

      output = dict()
      output['points'] = raw['points'].astype(np.float32)
      if self.has_normal:
          output['normals'] = raw['normals'].astype(np.float32)
      if self.has_color:
          output['colors'] = raw['colors'].astype(np.float32)
      if self.has_label:
          output['labels'] = raw['labels'].astype(np.int32)
      return output

    except Exception as e:
      print(f"[ERROR] Failed to read PLY file: {filename} | Error: {e}", file=sys.stderr, flush=True)
      raise  # Re-raise so Dataset can handle it


class ReadNpz:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.has_normal = has_normal
    self.has_color = has_color
    self.has_label = has_label

  def __call__(self, filename: str):
    raw = np.load(filename)

    output = dict()
    output['points'] = raw['points'].astype(np.float32)
    if self.has_normal:
      output['normals'] = raw['normals'].astype(np.float32)
    if self.has_color:
      output['colors'] = raw['colors'].astype(np.float32)
    if self.has_label:
      output['labels'] = raw['labels'].astype(np.int32)
    return output


class ReadFile:

  def __init__(self, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    self.read_npz = ReadNpz(has_normal, has_color, has_label)
    self.read_ply = ReadPly(has_normal, has_color, has_label)

  def __call__(self, filename: str):
    func = {'npz': self.read_npz, 'ply': self.read_ply}
    suffix = filename.split('.')[-1]
    return func[suffix](filename)


class Transform(ocnn.dataset.Transform):
  r''' Wraps :class:`ocnn.data.Transform` for convenience.
  '''

  def __init__(self, flags):
    super().__init__(**flags)
    self.flags = flags
