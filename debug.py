from data_loader_kitti import kitti_loader_exp2
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy.linalg import expm, norm
import math
import open3d as o3d

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def sample_random_trans(pcd, randg, rotation_range=0):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

dataset = kitti_loader_exp2.KITTINMPairDataset(phase='train',
                                                   random_rotation=False,
                                                   random_scale=False,)
pcd = dataset[0][0]
o3d.visualization.draw_geometries(pcd.cpu().detach().numpy(),
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])