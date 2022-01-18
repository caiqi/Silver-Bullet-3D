"""
Behavior cloning(BC)
"""
import torch
import torch.nn.functional as F

from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.lrscheduler import build_lr_scheduler
from ..builder import BRL


@BRL.register_module()
class DT(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128, l1_loss=0.0, l2_loss=1.0):
        super(DT, self).__init__()
        self.batch_size = batch_size
        self.l1_loss = l1_loss
        self.l2_loss = l2_loss

        policy_optim_cfg = policy_cfg.pop("optim_cfg")

        lr_scheduler_cfg = policy_cfg.pop("lr_scheduler_cfg")

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.policy = build_model(policy_cfg)
        self.policy_optim = build_optimizer(self.policy, policy_optim_cfg)
        lr_scheduler_cfg.update({'optimizer': self.policy_optim})
        self.policy_lr_scheduler = build_lr_scheduler(lr_scheduler_cfg)

    def get_lr(self):
        lr = []
        for param_group in self.policy_optim.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        lr = lr[0]
        return lr

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size)
        sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
        pred_action = self.policy(sampled_batch, mode='eval')
        action_target = torch.clone(sampled_batch['a'])
        attention_mask = sampled_batch['mask']
        act_dim = pred_action.shape[-1]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        l2_loss = F.mse_loss(pred_action, action_target) * self.l2_loss
        l1_loss = F.l1_loss(pred_action, action_target) * self.l1_loss
        total_loss = l1_loss + l2_loss
        self.policy_optim.zero_grad()
        total_loss.backward()
        self.policy_optim.step()
        self.policy_lr_scheduler.step()
        return {
            'policy_abs_error': l1_loss.item(),
            'policy_loss': total_loss.item()
        }


@BRL.register_module()
class EnsembleDT(DT):
    def update_parameters(self, memory, updates):
        l1_loss_sum = 0
        l2_loss_sum = 0
        if isinstance(self.policy, torch.nn.DataParallel):
            ensemble_num = self.policy.module.ensemble_num
        else:
            ensemble_num = self.policy.ensemble_num
        for i in range(ensemble_num):
            sampled_batch = memory.sample(self.batch_size)
            sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
            pred_action = self.policy(sampled_batch, mode='eval', model_idx=i)
            action_target = torch.clone(sampled_batch['a'])
            attention_mask = sampled_batch['mask']
            act_dim = pred_action.shape[-1]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            l2_loss = F.mse_loss(pred_action, action_target) * self.l2_loss
            l1_loss = F.l1_loss(pred_action, action_target) * self.l1_loss
            l1_loss_sum = l1_loss_sum + l1_loss
            l2_loss_sum = l2_loss_sum + l2_loss

        total_loss = l1_loss_sum + l2_loss_sum
        self.policy_optim.zero_grad()
        total_loss.backward()
        self.policy_optim.step()
        self.policy_lr_scheduler.step()
        return {
            'policy_abs_error': l1_loss_sum.item() / ensemble_num,
            'policy_loss': total_loss.item() / ensemble_num
        }
