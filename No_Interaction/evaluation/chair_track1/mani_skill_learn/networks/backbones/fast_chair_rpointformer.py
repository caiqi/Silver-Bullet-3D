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
class FastRPointFormerCoreV0(PointBackbone):
    def __init__(
        self, 
        num_objs,
        conv_cfg, 
        mlp_cfg, 
        stack_frame
    ):
        super(FastRPointFormerCoreV0, self).__init__()
        conv_cfg = conv_cfg.deepcopy()
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.global_mlp = nn.ModuleList([build_backbone(mlp_cfg) for i in range(num_objs)])
        self.is_conv = conv_cfg.type == 'ConvMLP'

    def forward_raw(self, point, state, obj_masks, xyz=None, ds_mask=None):
        B, N = point.shape[:2]
        state = torch.cat([point, state[:, None].repeat(1, N, 1)], dim=-1)  # [B, N, CS]

        if self.is_conv:
            point_feature = self.conv_mlp(state.transpose(2, 1)).transpose(2, 1)
        else:
            point_feature, obj_masks = self.conv_mlp(state, xyz, obj_masks, ds_mask)
            N = point_feature.shape[1]
            
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])

        # [B, K, N / K, CF]
        global_features = []
        for i in range(len(obj_masks)):
            mask = obj_masks[i].view(B, self.stack_frame, N // self.stack_frame, 1)
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]
            global_feature = global_feature.reshape(B, -1)
            global_feature = self.global_mlp[i](global_feature)
            global_features.append(global_feature)
        return global_features

@BACKBONES.register_module()
class FastChairRPointFormer(PointBackbone):
    def __init__(self, pcd_pn_cfg, state_mlp_cfg, final_mlp_cfg, stack_frame, num_objs, transformer_cfg=None, disable_rgb = False, disable_agent = False):
        super(FastChairRPointFormer, self).__init__()
        # OpenCabinetDoor and OpenCabinetDrawer: 2,  PushChair and MoveBucket: 4
        finger_num = 2 if num_objs == 3 else 4
        state_other_input_dim = 32 if num_objs == 3 else 56
        xyz_dim = 3
        rgb_dim = 3
        xyz_mean_dim = 3
        xy_dim = 2
        self.disable_rgb = disable_rgb
        self.disable_agent = disable_agent
        
        self.stack_frame = stack_frame
        self.num_objs = num_objs + 2
        assert self.num_objs > 0

        state_mlp_cfg.mlp_spec[0] = xyz_dim * finger_num + state_other_input_dim + xyz_mean_dim
        pcd_pn_cfg.conv_cfg.mlp_spec[0] = state_mlp_cfg.mlp_spec[0] + xyz_dim + rgb_dim + finger_num*(xyz_dim+1)+(xy_dim+1)

        pcd_pns_cfg = pcd_pn_cfg
        pcd_pns_cfg.update({'num_objs': self.num_objs+2})
        self.pcd_pns = build_backbone(pcd_pns_cfg)
        self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None

        self.state_mlp = build_backbone(state_mlp_cfg)
        self.global_mlp = build_backbone(final_mlp_cfg)

        self.pcd_type = pcd_pn_cfg.conv_cfg.type
        self.out_dim = self.global_mlp.mlp[-1].out_features

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
        if self.disable_agent:
            state = state * 0.0
        if self.disable_rgb:
            rgb = rgb * 0.0
        
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
        point = torch.cat([xyz, rgb, state_diff], dim=-1)

        # networks
        obj_features = [] 
        obj_features.append(self.state_mlp(state))
        obj_features += self.pcd_pns.forward_raw(point, state, obj_masks, xyz, obj_masks[2])

        if self.attn is not None:
            obj_features = torch.stack(obj_features, dim=-2)  # [B, NO + 3, F]
            new_seg = torch.stack(obj_masks, dim=-1)  # [B, N, NO + 2]
            non_empty = (new_seg > -0.5).any(1).float()  # [B, NO + 2] ###################################################
            non_empty = torch.cat([torch.ones_like(non_empty[:,:1]), non_empty], dim=-1) # [B, NO + 3]
            obj_attn_mask = non_empty[..., None] * non_empty[:, None]  # [B, NO + 3, NO + 3]           
            global_feature = self.attn(obj_features, obj_attn_mask)  # [B, F]
        else:
            global_feature = torch.cat(obj_features, dim=-1)  # [B, (NO + 3) * F]

        x = self.global_mlp(global_feature)
        return x
