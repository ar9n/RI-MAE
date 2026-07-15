import os
import torch
import trimesh
import json
import numpy as np
import torch.utils.data as data
from pathlib import Path
from .build import DATASETS, build_dataset_from_cfg
from utils.logger import *
import random


@DATASETS.register_module()
class Replay(data.Dataset):
    def __init__(self, config):
        # Extract target_dataset and replay_dataset from config.others
        target_cfg = config.get('target_dataset')
        replay_cfg = config.get('replay_dataset')
        
        if not target_cfg or not replay_cfg:
            raise ValueError("Mix dataset requires 'target_dataset' and 'replay_dataset' in config")
        
        # Build datasets by passing _base_ and others separately
        self.target_dataset = build_dataset_from_cfg(
            target_cfg['_base_'], 
            target_cfg['others']
        )
        self.replay_dataset = build_dataset_from_cfg(
            replay_cfg['_base_'], 
            replay_cfg['others']
        )
        
        self.sample_points_num = config.get('npoints', 1024)
        self.rot = config.get('rot', False)
        self.replay_ratio = config.get('replay_ratio', 0.1)  # Default to 10% replay

    def __getitem__(self, idx):
        # default: 10% chance to use replay dataset, 90% use target dataset
        if random.random() < self.replay_ratio:
            sample = self.replay_dataset[random.randrange(len(self.replay_dataset))]
        else:
            sample = self.target_dataset[idx % len(self.target_dataset)]
        
        # Ensure sample is not None
        if sample is None:
            return self.target_dataset[idx % len(self.target_dataset)]
        
        return sample

    def __len__(self):
        return len(self.target_dataset)