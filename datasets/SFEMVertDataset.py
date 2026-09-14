import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, -np.sin(a), 0],
                         [0, 1, 0, 0],
                         [np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, False)
    return rot

@DATASETS.register_module()
class SFEMVert(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        self.rot = config.get('rot', False)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'SFEMVert')
        
        self.file_list = []

        if self.subset == 'train' or self.whole:
            self._make_file_list(os.path.join(self.data_root, 'train'))
        elif self.subset == 'test' or self.whole:
            self._make_file_list(os.path.join(self.data_root, 'val'))

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'SFEMVert')

    def _make_file_list(self, path):
        for root, dirs, files in os.walk(path):
            for f in files:
                self.file_list.append({
                    'dataset': 'SFEM',
                    'file_name': f,
                    'file_path': os.path.join(root, f)
                })

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        with IO.get(sample['file_path']) as f:
            vertices = f['/Vertices'][()].astype(np.float64) # shape: (N, 3)
            fixed_facet = f['/Fixed_Facet'][()].astype(np.float64) # shape: (N, 2)
            u = f['/u'][()].astype(np.float64) # shape: (N, 3)
            stress = f['/VonMises'][()].astype(np.float64) # shape: (N, 1)

            data = np.concatenate([vertices, fixed_facet, u, stress], axis=1) # shape: (N, 9)

        # Remove rows where fixed_facet is (0, 0), which indicates that the vertex is not on the surface
        data = data[~((data[:, 3] == 0) & (data[:, 4] == 0))]

        sampled_ids = np.random.choice(data.shape[0], self.sample_points_num, replace=True)
        data = data[sampled_ids]
        data = self.pc_norm(data)
        if self.rot:
            data = data @ rnd_rot()
        data = torch.from_numpy(data).float()
        return sample['dataset'], sample['file_name'], (data, 0)

    def __len__(self):
        return len(self.file_list)