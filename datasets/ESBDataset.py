import os
import torch
import trimesh
import numpy as np
import torch.utils.data as data
from pathlib import Path
from .build import DATASETS
from utils.logger import *

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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
class EngineeringShapeBenchmark(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        self.rot = config.get('rot', False)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'EngineeringShapeBenchmark')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'EngineeringShapeBenchmark')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'EngineeringShapeBenchmark')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            super_class = line.split('/')[0]
            category = line.split('/')[1]
            object_name = line.split('/')[2]
            self.file_list.append({
                'super_class': super_class,
                'category': category,
                'object_name': object_name,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'EngineeringShapeBenchmark')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def stl_to_point_cloud(self, stl_path, n_points):
        stl_file = os.path.join(self.data_root, stl_path)
        mesh = trimesh.load_mesh(stl_file, force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        points = np.asarray(points, dtype=np.float32)
        return points

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = self.stl_to_point_cloud(sample['file_path'], self.sample_points_num)
        data = self.pc_norm(data)
        if self.rot:
            data = data @ rnd_rot()
        data = torch.from_numpy(data).float()
        return sample['category'], sample['object_name'], data

    def __len__(self):
        return len(self.file_list)