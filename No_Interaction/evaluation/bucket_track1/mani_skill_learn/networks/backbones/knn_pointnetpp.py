import torch
import torch.nn as nn
from mani_skill_learn.networks.modules.conv_down import ConvDown

from ..builder import BACKBONES

@BACKBONES.register_module()
class KnnPointNetPP(nn.Module):
    def __init__(
        self, 
        mlp_spec,
        npoint,
        radii,
        nsamples,
        num_blocks,
    ):
        super(KnnPointNetPP, self).__init__()
        blocks = []
        for i in range(num_blocks):
            block = ConvDown(
                in_dim = mlp_spec[i],
                out_dim = mlp_spec[i+1],
                npoint = npoint[i],
                radii = radii[i],
                nsample = nsamples[i]
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(mlp_spec[-1], mlp_spec[-1])
        self.num_blocks = num_blocks
        
    def forward(self, feat, xyz, obj_masks, downsample_mask):
        for i in range(self.num_blocks):
            feat, xyz, obj_masks, downsample_mask = self.blocks[i](feat, xyz, obj_masks, downsample_mask)
        feat = self.fc(feat)
        return feat, obj_masks
        

