import torch
from mani_skill_learn.utils.torch import ExtendedModule

from ..builder import POLICYNETWORKS, build_backbone, build_dense_head
from ..utils import replace_placeholder_with_args, get_kwargs_from_shape
from .dt_model.base_transformer import TransformerModel


@POLICYNETWORKS.register_module()
class ContinuousDTPolicy(ExtendedModule):
    def __init__(self, nn_cfg, policy_head_cfg, action_space, obs_shape=None, action_shape=None,
                 dt_state_dim=128,
                 dt_K=20,
                 dt_max_ep_len=80,
                 dt_embed_dim=128,
                 dt_n_layer=4,
                 dt_n_head=8,
                 dt_attn_pdrop=0.1,
                 disable_reward=False,
                 pass_through=False,
                 original_weight=0.0,
                 shuffle_noise=0.0
                 ):
        super(ContinuousDTPolicy, self).__init__()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.backbone = build_backbone(nn_cfg)
        self.dt_state_dim = dt_state_dim
        self.action_shape = action_shape
        self.original_weight = original_weight
        self.dt_max_ep_len = dt_max_ep_len
        self.dt_model = TransformerModel(
            state_dim=dt_state_dim,
            act_dim=action_shape,
            max_length=dt_K,
            disable_action=disable_reward,
            nhead=dt_n_head, d_hid=dt_embed_dim,
            nlayers=dt_n_layer, dropout=dt_attn_pdrop,
            pass_through=pass_through,
            original_weight=original_weight,
            shuffle_noise=shuffle_noise
        )
        if action_space.is_bounded():
            low = action_space.low
            high = action_space.high
            scale_prior = (high - low) / 2
            bias_prior = (low + high) / 2
            policy_head_cfg['scale_prior'] = scale_prior
            policy_head_cfg['bias_prior'] = bias_prior
        self.policy_head = build_dense_head(policy_head_cfg)

    def init_weights(self, pretrained=None, init_cfg=None):
        self.backbone.init_weights(pretrained, **init_cfg)

    def forward(self, state, num_actions=1, mode="sample"):
        if self.training:
            return self.forward_train(state, num_actions, mode, self.backbone, self.dt_model, self.policy_head)
        else:
            return self.forward_eval(state, num_actions, mode, self.backbone, self.dt_model, self.policy_head)

    def forward_eval(self, state, num_actions=1, mode="eval", backbone=None, dt_model=None, policy_head=None):
        states = state['states']
        dt_states = state['dt_states']
        actions = state['actions']
        input_features = backbone(states)
        dt_states = torch.cat((dt_states, input_features), dim=0)
        actions_pred = dt_model.get_action(dt_states, actions)
        all_info = policy_head(actions_pred, num_actions=num_actions)
        sample, log_prob, mean = all_info[:3]
        if mode == 'all':
            return all_info, dt_states
        elif mode == 'eval':
            return mean, dt_states
        elif mode == 'sample':
            return sample, dt_states
        else:
            raise ValueError(f"Unsupported mode {mode}!!")

    def forward_train(self, state, num_actions=1, mode='eval', backbone=None, dt_model=None, policy_head=None):
        point_cloud_state = state["s"]
        num_traj, max_traj_len, point_num, point_dim = point_cloud_state.shape
        feature_state = {
            "pointcloud": {
                "xyz": point_cloud_state[:, :, :, :3].view(num_traj * max_traj_len, point_num, 3),
                "rgb": point_cloud_state[:, :, :, 3:6].view(num_traj * max_traj_len, point_num, 3),
                "seg": point_cloud_state[:, :, :, 6:].view(num_traj * max_traj_len, point_num, -1),
            },
            "state": state["agent"].view(num_traj * max_traj_len, -1)
        }
        per_frame_features = backbone(feature_state)
        per_frame_features = per_frame_features.view(num_traj, max_traj_len, -1)
        states, actions, rewards, dones, rtg, timesteps, attention_mask = per_frame_features, state["a"], state['r'], \
                                                                          state['d'], state['rtg'], state['timesteps'], \
                                                                          state['mask']
        action_preds = dt_model.forward(
            states, actions, attention_mask=attention_mask)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        all_info = policy_head(action_preds, num_actions=num_actions)
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
class EmsembleContinuousDTPolicy(ContinuousDTPolicy):
    def __init__(self, nn_cfg, policy_head_cfg, action_space, obs_shape=None, action_shape=None,
                 dt_state_dim=128,
                 dt_K=20,
                 dt_max_ep_len=80,
                 dt_embed_dim=128,
                 dt_n_layer=4,
                 dt_n_head=8,
                 dt_attn_pdrop=0.1,
                 disable_reward=False,
                 pass_through=False,
                 original_weight=0.0,
                 shuffle_noise=0.0,
                 ensemble_dt=False
                 ):
        super(ExtendedModule, self).__init__()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.dt_state_dim = dt_state_dim
        self.action_shape = action_shape
        self.original_weight = original_weight
        self.dt_max_ep_len = dt_max_ep_len
        self.ensemble_num = policy_head_cfg.pop('ensemble_num')
        self.ensemble_dt = ensemble_dt
        if action_space.is_bounded():
            low = action_space.low
            high = action_space.high
            scale_prior = (high - low) / 2
            bias_prior = (low + high) / 2
            policy_head_cfg['scale_prior'] = scale_prior
            policy_head_cfg['bias_prior'] = bias_prior

        self.backbone = torch.nn.ModuleList([build_backbone(nn_cfg) for i in range(self.ensemble_num)])
        self.policy_head = torch.nn.ModuleList([build_dense_head(policy_head_cfg) for i in range(self.ensemble_num)])
        if self.ensemble_dt:
            self.dt_model = torch.nn.ModuleList([TransformerModel(
                state_dim=dt_state_dim,
                act_dim=action_shape,
                max_length=dt_K,
                disable_action=disable_reward,
                nhead=dt_n_head, d_hid=dt_embed_dim,
                nlayers=dt_n_layer, dropout=dt_attn_pdrop,
                pass_through=pass_through,
                original_weight=original_weight,
                shuffle_noise=shuffle_noise
            ) for i in range(self.ensemble_num)])
        else:
            self.dt_model = TransformerModel(
                state_dim=dt_state_dim,
                act_dim=action_shape,
                max_length=dt_K,
                disable_action=disable_reward,
                nhead=dt_n_head, d_hid=dt_embed_dim,
                nlayers=dt_n_layer, dropout=dt_attn_pdrop,
                pass_through=pass_through,
                original_weight=original_weight,
                shuffle_noise=shuffle_noise
            )
        self.backbone.out_dim = self.backbone[0].out_dim

    def init_weights(self, pretrained=None, init_cfg=None):
        for i in range(self.ensemble_num):
            self.backbone[i].init_weights(pretrained, **init_cfg)

    def forward(self, state, num_actions=1, mode="sample", model_idx=-1):
        if self.training:
            if self.ensemble_dt:
                dt_model = self.dt_model[model_idx]
            else:
                dt_model = self.dt_model
            return self.forward_train(state, num_actions, mode, self.backbone[model_idx], dt_model,
                                      self.policy_head[model_idx])
        else:
            mean = 0
            state_sum = 0
            for i in range(self.ensemble_num):
                if self.ensemble_dt:
                    dt_model = self.dt_model[i]
                else:
                    dt_model = self.dt_model
                pred_mean, pred_state = self.forward_eval(state, num_actions, mode, self.backbone[i], dt_model,
                                                          self.policy_head[i])
                mean = mean + pred_mean
                state_sum = state_sum + pred_state
            state_mean = state_sum / self.ensemble_num
            mean = mean / self.ensemble_num
            return mean, state_mean
