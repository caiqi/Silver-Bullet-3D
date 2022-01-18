import torch

from mani_skill_learn.utils.torch import ExtendedModule
from ..builder import POLICYNETWORKS, build_backbone, build_dense_head
from ..utils import replace_placeholder_with_args, get_kwargs_from_shape


@POLICYNETWORKS.register_module()
class ContinuousPolicy(ExtendedModule):
    def __init__(self, nn_cfg, policy_head_cfg, action_space, obs_shape=None, action_shape=None, num_bins=-1):
        super(ContinuousPolicy, self).__init__()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.num_bins = num_bins
        self.backbone = build_backbone(nn_cfg)
        if action_space.is_bounded():
            low = action_space.low
            high = action_space.high
            scale_prior = (high - low) / 2
            bias_prior = (low + high) / 2
            policy_head_cfg['scale_prior'] = scale_prior
            policy_head_cfg['bias_prior'] = bias_prior
        self.policy_head = build_dense_head(policy_head_cfg)
        self.action_shape = action_shape

    def init_weights(self, pretrained=None, init_cfg=None):
        self.backbone.init_weights(pretrained, **init_cfg)

    def forward(self, state, num_actions=1, mode='sample'):
        if self.num_bins > 0:
            if self.training:
                return self.backbone(state)
            else:
                pred = self.backbone(state)
                pred = torch.reshape(pred, shape=(pred.shape[0], -1, self.num_bins))
                pred = torch.argmax(pred, dim=-1) * 1.0 / self.num_bins * 2.0 - 1
                return pred
        else:

            state_features = self.backbone(state)
            all_info_base = self.policy_head(state_features[:, :self.action_shape], num_actions=num_actions)
            if self.training and state_features.shape[-1] > self.action_shape:
                all_info_base_v2 = self.policy_head(state_features[:, self.action_shape:], num_actions=num_actions)
                all_info = [torch.cat((k,m), dim=-1) for k,m in zip( all_info_base, all_info_base_v2 )]
            else:
                all_info = all_info_base
            # all_info = self.policy_head(self.backbone(state)[:, :self.action_shape], num_actions=num_actions)
            if mode != "eval":
                raise NotImplementedError
            sample, log_prob, mean = all_info[:3]
            if mode == 'all':
                return all_info
            elif mode == 'eval':
                return mean
            elif mode == 'sample':
                return sample
            else:
                raise ValueError(f"Unsupported mode {mode}!!")


@POLICYNETWORKS.register_module()
class EnsembleContinuousPolicy(ExtendedModule):
    def __init__(self, nn_cfg, policy_head_cfg, action_space, obs_shape=None, action_shape=None, num_bins=-1):
        super(EnsembleContinuousPolicy, self).__init__()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.num_bins = num_bins
        self.ensemble_num = policy_head_cfg.pop('ensemble_num')
        if action_space.is_bounded():
            low = action_space.low
            high = action_space.high
            scale_prior = (high - low) / 2
            bias_prior = (low + high) / 2
            policy_head_cfg['scale_prior'] = scale_prior
            policy_head_cfg['bias_prior'] = bias_prior
        self.action_shape = action_shape
        self.backbone = torch.nn.ModuleList([build_backbone(nn_cfg) for i in range(self.ensemble_num)])
        self.policy_head = torch.nn.ModuleList([build_dense_head(policy_head_cfg) for i in range(self.ensemble_num)])

    def init_weights(self, pretrained=None, init_cfg=None):
        for i in range(self.ensemble_num):
            self.backbone[i].init_weights(pretrained, **init_cfg)

    def forward(self, state, num_actions=1, mode='sample', model_idx=-1):
        if self.training:
            if self.num_bins > 0:
                return self.backbone[model_idx](state)
            else:
                state_features = self.backbone[model_idx](state)

                all_info_base = self.policy_head[model_idx](state_features[:, :self.action_shape], num_actions=num_actions)
                if state_features.shape[-1] > self.action_shape:
                    all_info_base_v2 = self.policy_head[model_idx](state_features[:, self.action_shape:], num_actions=num_actions)
                    all_info = [torch.cat((k,m), dim=-1) for k,m in zip( all_info_base, all_info_base_v2 )]
                else:
                    all_info = all_info_base
                sample, log_prob, mean = all_info[:3]
                if mode == 'all':
                    return all_info
                elif mode == 'eval':
                    return mean
                elif mode == 'sample':
                    return sample
                else:
                    raise ValueError(f"Unsupported mode {mode}!!")
        else:
            if self.num_bins > 0:
                means = 0
                for i in range(self.ensemble_num):
                    pred = self.backbone[i](state)
                    pred = torch.reshape(pred, shape=(pred.shape[0], -1, self.num_bins))
                    pred = torch.argmax(pred, dim=-1) * 1.0 / self.num_bins * 2.0 - 1
                    means += pred
                means /= self.ensemble_num
            else:
                means = 0
                for i in range(self.ensemble_num):
                    all_info = self.policy_head[i](self.backbone[i](state)[:, :self.action_shape],
                                                   num_actions=num_actions)
                    sample, log_prob, mean = all_info[:3]
                    means += mean
                means /= self.ensemble_num
            return means
