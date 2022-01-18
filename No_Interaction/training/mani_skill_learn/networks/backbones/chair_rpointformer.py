import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from mani_skill_learn.utils.data import dict_to_seq
from mani_skill_learn.utils.torch import masked_average, masked_max
from ..builder import BACKBONES, build_backbone
from .pointnet import PointBackbone
from mani_skill_learn.utils.torch.point_utils import square_distance

@BACKBONES.register_module()
class ChairRPointFormer(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None):
        super(ChairRPointFormer, self).__init__()
        # OpenCabinetDoor and OpenCabinetDrawer: 2,  PushChair and MoveBucket: 4
        finger_num = 2 if num_objs == 3 else 4
        state_other_input_dim = 32 if num_objs == 3 else 56
        xyz_dim = 3
        rgb_dim = 3
        xyz_mean_dim = 3
        xy_dim = 2
        
        self.stack_frame = stack_frame
        self.num_objs = num_objs + 2
        assert self.num_objs > 0

        if 'color_jitter' in pcd_pn_cfg:
            self.color_jitter = pcd_pn_cfg.pop('color_jitter')
        else:
            self.color_jitter = [0., 0., 0.]

        state_mlp_cfg.mlp_spec[0] = xyz_dim * finger_num + state_other_input_dim + xyz_mean_dim
        pcd_pn_cfg.conv_cfg.mlp_spec[0] = state_mlp_cfg.mlp_spec[0] + xyz_dim + rgb_dim + finger_num*(xyz_dim+1)+(xy_dim+1)

        self.pcd_pns = nn.ModuleList([build_backbone(pcd_pn_cfg) for i in range(self.num_objs + 2)])
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None

        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.pcd_type = pcd_pn_cfg.conv_cfg.type

    def normalize_coordinate(self, xyz, state):
        # OpenCabinetDoor and OpenCabinetDrawer
        if state.shape[1] == 38:
            finger_pos = state[:, 0:6].reshape(-1, 2, 3)
            base_pos_xy = state[:, 12:14]
            state_others = torch.cat([state[:, 6:12], state[:, 14:38]], dim=1)
        else: # PushChair and MoveBucket
            finger_pos = state[:, 0:12].reshape(-1, 4, 3)
            base_pos_xy = state[:, 24:26]
            state_others = torch.cat([state[:, 12:24], state[:, 26:68]], dim=1)

        xy_cat = torch.cat([xyz[:, :, 0:2], finger_pos[:, :, 0:2], base_pos_xy.unsqueeze(1)], dim=1)
        z_cat = torch.cat([xyz[:, :, 2], finger_pos[:, :, 2]], dim=1).unsqueeze(-1)
        xy_mean = xy_cat.mean(1)
        z_mean = z_cat.mean(1)
        xyz_mean = torch.cat([xy_mean, z_mean], dim=-1)

        ################################################
        xyz = xyz - xyz_mean.unsqueeze(1)
        finger_pos = finger_pos - xyz_mean.unsqueeze(1)
        base_pos_xy = base_pos_xy - xy_mean
        ################################################

        state_others = torch.cat([base_pos_xy, state_others], dim=-1)
        return xyz, finger_pos, base_pos_xy, state_others, xyz_mean

    def forward_raw(self, pcd, state):
        assert isinstance(pcd, dict) and 'xyz' in pcd and 'seg' in pcd
        pcd = pcd.copy()
        seg = pcd.pop('seg')  # [B, N, NO]
        xyz = pcd['xyz']  # [B, N, 3]
        rgb = pcd['rgb']  # [B, N, 3]

        ################################################################
        with torch.no_grad():
            z  = xyz[:, :, 2] + seg[:, :, 0] * 1000
            trgpt_mask = (z <= 0.15) & (rgb[:, :, 0] > 0.6) & (rgb[:, :, 1] < 0.05) & (rgb[:, :, 2] < 0.05)
            seg_chair_pt = torch.logical_not(seg[:, :, 0])
            seg_trgpt = torch.logical_and(seg_chair_pt, trgpt_mask)
            seg_chair = torch.logical_and(seg_chair_pt, torch.logical_not(trgpt_mask))
            seg = torch.cat([seg, seg_trgpt.unsqueeze(-1), seg_chair.unsqueeze(-1)], dim=-1)

        #import open3d as o3d
        #for i in range(16):
        #    batch_idx = i
        #    v_xyz = xyz[batch_idx].data.cpu().numpy()
        #    v_rgb = rgb[batch_idx].data.cpu().numpy()
        #    ptr_mask = seg_trgpt[batch_idx].data.cpu().numpy()
        #    v_rgb[ptr_mask, 0] = 0
        #    v_rgb[ptr_mask, 1] = 1
        #    v_rgb[ptr_mask, 2] = 0
        #    pcd = o3d.geometry.PointCloud()
        #    pcd.points = o3d.utility.Vector3dVector(v_xyz)
        #    pcd.colors = o3d.utility.Vector3dVector(v_rgb)
        #    o3d.visualization.draw_geometries([pcd])
        ################################################################

        if np.sum(self.color_jitter) > 0 and self.training:
            jitter = transforms.ColorJitter(*self.color_jitter)
            _rgb = (rgb*255).permute(0, 2, 1).reshape(rgb.shape[0], 3, 40, 30)
            _rgb = _rgb.type(torch.int64)
            _rgb = jitter(_rgb).view(rgb.shape[0], 3, 1200).type(xyz.dtype)
            rgb = _rgb.permute(0, 2, 1) / 255.0

        obj_masks = [1. - (torch.sum(seg[..., 0].unsqueeze(-1), dim=-1) > 0.5).type(xyz.dtype)]
        for i in range(self.num_objs):
            obj_masks.append(seg[..., i])
        obj_masks.append(torch.ones_like(seg[..., 0])) # the entire point cloud

        xyz, finger_pos, base_pos_xy, state_others, xyz_mean = self.normalize_coordinate(xyz, state)
        xy = xyz[:, :, 0:2]

        # xyz: B, N, 3
        # finger_pos: B, M, 3
        batch_size, num_points = xyz.shape[0:2]

        ################################# diff #################################
        xyz_finger_diff = xyz.unsqueeze(2) - finger_pos.unsqueeze(1)
        xyz_finger_dist = (xyz_finger_diff * xyz_finger_diff).sum(-1)
        xyz_finger_diff = xyz_finger_diff.view(batch_size, num_points, -1)
        xy_base_diff = xy - base_pos_xy.unsqueeze(1)
        xy_base_dist = (xy_base_diff * xy_base_diff).sum(-1).unsqueeze(-1)
        state_diff = torch.cat([xyz_finger_diff, xyz_finger_dist, xy_base_diff, xy_base_dist], dim=-1)
        ################################# diff #################################

        # state = finger_pos + state_others
        finger_pos = finger_pos.view(batch_size, -1)
        state = torch.cat([finger_pos, state_others, xyz_mean], dim=-1)

        # point cloud
        # xyz = xyz + rgb + state_diff
        if 'Knn' in self.pcd_type:
            ori_xyz = xyz
            xyz_diff, dist = square_distance(xyz, xyz)
        else:
            ori_xyz = None
            xyz_diff = None
            dist = None
        point = torch.cat([xyz, rgb, state_diff], dim=-1)

        # networks
        obj_features = [] 
        obj_features.append(self.state_mlp(state))
        for i in range(len(obj_masks)):
            obj_mask = obj_masks[i]
            downsample_mask = obj_mask if i != len(obj_masks) - 1 else seg[..., 0]
            obj_features.append(self.pcd_pns[i].forward_raw(point, state, obj_mask, ori_xyz, xyz_diff, dist, downsample_mask))  # [B, F]

        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > 0.5).any(1).float()  # [B, NO + 2]
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]           
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]

        x = self.global_mlp(global_feature)
        return x
