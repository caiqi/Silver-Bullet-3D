import math
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mani_skill_learn.utils.meta import get_root_logger
from mani_skill_learn.utils.torch import load_checkpoint
from ..builder import BACKBONES
from ..modules import ConvModule, build_init
from ..modules import build_activation_layer, build_norm_layer
from mani_skill_learn.networks.modules.weight_init import trunc_normal_

class FeedForwardBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(out_features)

        if in_features != out_features:
            self.res = nn.Linear(in_features, out_features)
        else:
            self.res = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        residual = self.res(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = residual + x
        x = self.ln(x)
        return x  

@BACKBONES.register_module()
class FeedForward(nn.Module):
    def __init__(self, mlp_spec, dropout, num_blocks):
        super(FeedForward, self).__init__()
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(FeedForwardBlock(mlp_spec[0], mlp_spec[1], mlp_spec[2], dropout))
            else:
                blocks.append(FeedForwardBlock(mlp_spec[2], mlp_spec[1], mlp_spec[2], dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)



@BACKBONES.register_module()
class FeedForwardInteract(nn.Module):
    def __init__(self, mlp_spec, dropout, num_blocks):
        super(FeedForwardInteract, self).__init__()
        blocks = []
        embeds = []
        layernorms = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(FeedForwardBlock(mlp_spec[0], mlp_spec[1], mlp_spec[2], dropout))
            else:
                blocks.append(FeedForwardBlock(mlp_spec[2], mlp_spec[1], mlp_spec[2], dropout))

            if i != num_blocks - 1:
                embeds.append(nn.Linear(mlp_spec[2]*2, mlp_spec[2]))
                layernorms.append(nn.LayerNorm(mlp_spec[2]))
        self.blocks = nn.ModuleList(blocks)
        self.embeds = nn.ModuleList(embeds)
        self.layernorms = nn.ModuleList(layernorms)
        self.num_blocks = num_blocks
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        B, N = x.shape[0:2]
        mask = mask.unsqueeze(-1)
        for i in range(self.num_blocks):
            x = self.blocks[i](x)

            if i != self.num_blocks - 1:
                gx = (x * mask).sum(1) / (mask.sum(1) + 1e-18)
                gx = gx.unsqueeze(1).expand(B, N, -1)
                x = torch.cat([gx, x], dim=-1)
                x = self.embeds[i](x)
                x = self.layernorms[i](x)
        return x

@BACKBONES.register_module()
class LinearMLP(nn.Module):
    def __init__(self, mlp_spec, norm_cfg=dict(type='BN1d'), bias='auto', inactivated_output=True,
                 pretrained=None, linear_init_cfg=None, norm_init_cfg=None):
        super(LinearMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            else:
                act_cfg = dict(type='ReLU')
            bias_i = norm_cfg is None if bias == 'auto' else bias
            # print(mlp_spec[i], mlp_spec[i + 1], bias_i)
            self.mlp.add_module(f'linear{i}', nn.Linear(mlp_spec[i], mlp_spec[i + 1], bias=bias_i))
            if norm_cfg:
                self.mlp.add_module(f'norm{i}', build_norm_layer(norm_cfg, mlp_spec[i + 1])[1])
            if act_cfg:
                self.mlp.add_module(f'act{i}', build_activation_layer(act_cfg))
        self.init_weights(pretrained, linear_init_cfg, norm_init_cfg)

    def forward(self, input):
        input = input
        return self.mlp(input)

    def init_weights(self, pretrained=None, linear_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            linear_init = build_init(linear_init_cfg) if linear_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Linear) and linear_init:
                    linear_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError('pretrained must be a str or None')


@BACKBONES.register_module()
class ConvMLP(nn.Module):
    def __init__(self, mlp_spec, norm_cfg=dict(type='BN1d'), bias='auto', inactivated_output=True,
                 pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        super(ConvMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
            else:
                act_cfg = dict(type='ReLU')
            self.mlp.add_module(
                f'layer{i}',
                ConvModule(
                    mlp_spec[i],
                    mlp_spec[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=bias,
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=True,
                    with_spectral_norm=False,
                    padding_mode='zeros',
                    order=('conv', 'norm', 'act'))
            )
        self.init_weights(pretrained, conv_init_cfg, norm_init_cfg)

    def forward(self, input):
        return self.mlp(input)

    def init_weights(self, pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            conv_init = build_init(conv_init_cfg) if conv_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Conv1d) and conv_init:
                    conv_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError('pretrained must be a str or None')


