
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Randomly sample m points from a set of N points
def random_perturbation_selection(features, m_samples):
    indices = np.random.choice(features.shape[0], m_samples, replace=False)
    return features[indices]

# Code from PatchCore paper's official implementation
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape (m, d)
      y: pytorch Variable, with shape (n, d)
    Returns:
      dist: pytorch Variable, with shape (m, n)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True)
    yy = torch.pow(y, 2).sum(1, keepdim=True)
    xy = torch.matmul(x, y.transpose(0, 1))
    dist = (xx.tile(1, n) + yy.tile(m, 1) - 2 * xy).clamp(min=1e-5)
    return dist.sqrt()

