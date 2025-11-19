import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from thsolver.dataset import Dataset as BaseDataset  # import original Dataset

class TetDataset(BaseDataset):
    def __init__(self, root, root2, filelist, filelist2, transform, transform2,
                 read_file=BaseDataset.__init__.__defaults__[2], in_memory=False, take: int = -1,
                 ptau_csv_path=None):
        # Init base Dataset (just once for filenames/labels)
        super().__init__(root, filelist, transform, read_file, in_memory, take)

        # Save extra info
        self.root2 = root2
        self.filelist2 = filelist2
        self.transform2 = transform2

        # Load second file list
        self.filenames, self.labels = self.load_filenames_from_file(filelist)
        self.filenames2, _ = self.load_filenames_from_file(filelist2)

        # Load ptau metadata if provided
        if ptau_csv_path:
            print('-----Loading ptau metadata from: ', ptau_csv_path)
            self.metadata_df = pd.read_csv(ptau_csv_path)
            meta = self.metadata_df.set_index('mri_id')
            filenames_base = pd.Series(self.filenames).apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            #filenames_base2 = pd.Series(self.filenames2).apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            self.ptau_values = meta.reindex(filenames_base)['ptau_c2n'].fillna(0.0).values.astype(np.float32)
            self.ptau_values = torch.as_tensor(self.ptau_values, dtype=torch.float32)
        else:
            print('-----No ptau metadata path (treat as 0): ', ptau_csv_path)
            self.ptau_values = np.zeros(len(self.filenames), dtype=np.float32)
            self.ptau_values = torch.as_tensor(self.ptau_values, dtype=torch.float32)

        # Optional in-memory loading
        if self.in_memory:
            print('Loading second view into memory from', filelist2)
            self.samples2 = [
                self.read_file(os.path.join(self.root2, f))
                for f in tqdm(self.filenames2, ncols=80, leave=False)
            ]

    def __getitem__(self, idx):
        # Load first sample
        sample1 = self.samples[idx] if self.in_memory else \
                  self.read_file(os.path.join(self.root, self.filenames[idx]))
        output = self.transform(sample1, idx)

        # Load second sample
        sample2 = self.samples2[idx] if self.in_memory else \
                  self.read_file(os.path.join(self.root2, self.filenames2[idx]))
        output2 = self.transform2(sample2, idx)

        # Merge second view
        output['points2'] = output2['points']
        output['inbox_mask2'] = output2['inbox_mask']
        output['octree2'] = output2['octree']

        # Add label and metadata
        output['label'] = self.labels[idx]
        output['ptau'] = self.ptau_values[idx]
        output['filename'] = self.filenames[idx]
        return output

    def load_filenames_from_file(self, filelist_path):
        filenames, labels = [], []
        with open(filelist_path) as fid:
            for line in fid:
                tokens = line.strip().split()
                filename = tokens[0].replace('\\', '/')
                label = tokens[1] if len(tokens) == 2 else 0
                filenames.append(filename)
                labels.append(int(label))

        num = len(filenames)
        if self.take > num or self.take < 1:
            self.take = num

        return filenames[:self.take], labels[:self.take]