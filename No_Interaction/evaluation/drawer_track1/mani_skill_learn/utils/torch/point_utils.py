import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# points: B, N, C
# idx:    B, N, nsamples
# out:    B, N, nsamples, C 
def index_points(points, idx):
    B, N, C = points.shape
    points = points.unsqueeze(2).expand(B, N, N, C)
    idx = idx.unsqueeze(-1).expand(B, N, -1, C)
    out = torch.gather(points, 2, idx)
    return out

# src: B, N, C
# dst: B, M, C
def square_distance(src, dst):
    with torch.no_grad():
        diff = src.unsqueeze(-2) - dst.unsqueeze(1)
        dist = (diff * diff).sum(-1)
        return diff, dist

# xyz: B, N, 3
# npoint: number of samples
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros((B, ), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# xyz: B, N, 3
# npoint: number of samples
# mask: B, N
def farthest_point_sample_mask(xyz, npoint, key_mask):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros((B, ), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        key_mask[:, farthest] = False
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist + 100 * key_mask
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def samples_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



#if __name__ == '__main__':
#    B, N, nsamples, C = 3, 30, 5, 10
#    points = torch.randn(B, N, C)
#    idx = torch.randint(0, N, (B, N, nsamples), dtype=torch.int64)
#    out = index_points(points, idx)