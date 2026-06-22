import os
import torch
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
class MechanicalComponentsBenchmark(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        
        train_data_folder = os.path.join(self.data_root, 'train')
        test_data_folder = os.path.join(self.data_root, 'test')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        self.rot = config.get('rot', False)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'MechanicalComponentsBenchmark')
        
        self.file_list = []

        if self.whole or self.subset == 'train':
            self.fill_file_list(self.file_list, train_data_folder)

        if self.whole or self.subset == 'test':
            self.fill_file_list(self.file_list, test_data_folder)

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'MechanicalComponentsBenchmark')

    def fill_file_list(self, file_list, data_folder):
        for root, dirs, files in os.walk(data_folder):
            for f in files:
                if f.endswith('.obj'):
                    category = root.split('/')[-1]
                    file_list.append({
                        'category': category,
                        'object_name': f,
                        'file_path': os.path.join(root, f)
                    })

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def obj_to_point_cloud(self, obj_path, n_points):
        try:
            vertices = []

            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if not vertices:
                print_log(f'[WARNING] No vertices found: {obj_path}', logger='MechanicalComponentsBenchmark')
                return np.zeros((n_points, 3), dtype=np.float32)

            vertices = np.asarray(vertices, dtype=np.float32)

            replace = len(vertices) < n_points
            indices = np.random.choice(len(vertices), size=n_points, replace=replace)
            points = vertices[indices]

            return points.astype(np.float32)

        except Exception as e:
            print_log(f'[WARNING] Failed to load {obj_path}: {e}', logger='MechanicalComponentsBenchmark')
            return np.zeros((n_points, 3), dtype=np.float32)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = self.obj_to_point_cloud(sample['file_path'], self.sample_points_num)
        data = self.pc_norm(data)
        if self.rot:
            data = data @ rnd_rot()
        data = torch.from_numpy(data).float()
        return sample['category'], sample['object_name'], data

    def __len__(self):
        return len(self.file_list)