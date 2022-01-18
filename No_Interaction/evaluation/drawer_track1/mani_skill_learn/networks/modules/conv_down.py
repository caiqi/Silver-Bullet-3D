import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_points3d.core.spatial_ops import DenseRadiusNeighbourFinder, DenseFPSSampler
    import torch_points_kernels as tp
except:
    pass
class ConvDown(nn.Module):
    def __init__(
        self, 
        in_dim,
        out_dim,
        npoint=None,
        radii=None,
        nsample=None,
    ):
        super(ConvDown, self).__init__()
        self.sampler = DenseFPSSampler(num_to_sample=npoint)
        self.neighbour_finder = DenseRadiusNeighbourFinder(radii, nsample)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim+3, out_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )

    # feat:     B, N, C
    # xyz:      B, N, 3
    # mask:     B, N
    # key_mask: B, N
    def forward(self, feat, xyz, masks, key_mask=None):
        B, N, C = feat.shape

        if key_mask is not None:
            scale = (torch.arange(N, dtype=torch.long).to(feat.device) + 10) * 10
            scale_xyz = xyz + key_mask.unsqueeze(-1) * scale.unsqueeze(0).unsqueeze(-1)
        else:
            scale_xyz = xyz
        idx = self.sampler(scale_xyz).long()
        M = idx.shape[-1]

        new_masks = [ mask.gather(1, idx) for mask in masks ]
        key_mask = key_mask.gather(1, idx) if key_mask is not None else None
        idx = idx.unsqueeze(-1).expand(B, M, 3)
        new_xyz = xyz.gather(1, idx)
        radius_idx = self.neighbour_finder(xyz, new_xyz, 0)

        new_xyz_trans = xyz.transpose(1, 2).contiguous()   
        grouped_xyz = tp.grouping_operation(new_xyz_trans, radius_idx)  # B, 3, npoint, nsample
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        
        feat_trans = feat.transpose(1, 2).contiguous()
        feat = tp.grouping_operation(feat_trans, radius_idx)
        feat = torch.cat([feat, grouped_xyz], dim=1)

        feat = self.conv(feat)
        sep = feat.shape[1] // 2
        feat_avg = feat[:, :sep]
        feat_max = feat[:, sep:]
        feat_avg = F.avg_pool2d(feat_avg, kernel_size=[1, feat_avg.size(3)]).squeeze(-1)
        feat_max = F.max_pool2d(feat_max, kernel_size=[1, feat_max.size(3)]).squeeze(-1)
        feat = torch.cat([feat_avg, feat_max], dim=1)
        feat = feat.transpose(1, 2).contiguous()
        return feat, new_xyz, new_masks, key_mask