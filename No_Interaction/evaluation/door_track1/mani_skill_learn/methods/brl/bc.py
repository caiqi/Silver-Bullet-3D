"""
Behavior cloning(BC)
"""
import torch
import torch.nn.functional as F
from torch import nn

from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.lrscheduler import build_lr_scheduler
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.torch import BaseAgent
from ..builder import BRL


@BRL.register_module()
class BC(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128):
        super(BC, self).__init__()
        self.batch_size = batch_size

        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        lr_scheduler_cfg = policy_cfg.pop("lr_scheduler_cfg")

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.loss_type = policy_cfg.pop('loss_type')
        if "loss_weight" in policy_cfg:
            self.loss_weight_type = policy_cfg.pop('loss_weight')
        else:
            self.loss_weight_type = None
        if "add_c_loss" in policy_cfg:
            self.add_c_loss = policy_cfg.pop('add_c_loss')
        else:
            self.add_c_loss = False
        if self.add_c_loss:
            self.discriminator = nn.Sequential(
                nn.Linear(action_shape * 2, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 1),
            )
        else:
            self.discriminator = None

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
        sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"])
        sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        pred_action = self.policy(sampled_batch['obs'], mode='eval')
        if self.loss_type == "mse_loss" or self.loss_type == "l1_loss":
            if self.loss_weight_type == None:
                loss_weight = torch.ones_like(pred_action)[:, 0:1]
            elif self.loss_weight_type == "hard":
                with torch.no_grad():
                    pre_loss = torch.square(pred_action - sampled_batch['actions']).sum(dim=-1)
                    loss_weight = pre_loss < torch.median(pre_loss)
                    loss_weight = loss_weight / (loss_weight.sum() + 1e-5) * loss_weight.numel()
                    loss_weight = loss_weight.unsqueeze(dim=-1)
            elif self.loss_weight_type == "soft":
                with torch.no_grad():
                    loss_weight = torch.abs(pred_action - sampled_batch['actions']).sum(dim=-1)
                    loss_weight = loss_weight.max() * 5 - loss_weight
                    loss_weight = loss_weight / (loss_weight.sum() + 1e-5) * loss_weight.numel()
                    loss_weight = loss_weight.unsqueeze(dim=-1)
            else:
                raise NotImplementedError

        if self.add_c_loss:
            assert self.loss_type in ['mse_loss', 'l1_loss']
            B, D = pred_action.shape
            input_features = pred_action.view(B, 1, D)
            target_features = sampled_batch['actions'].view(1, B, D)
            input_features = input_features.expand(B, B, D).reshape(B * B, D)
            target_features = target_features.expand(B, B, D).reshape(B * B, D)
            features = torch.cat((input_features, target_features), dim=-1)
            features_embed = self.discriminator(features).flatten()
            features_embed = features_embed.reshape(B, B)
            dis_loss = F.cross_entropy(features_embed,
                                       torch.arange(0, B, dtype=torch.long, device=features_embed.device))
        else:
            dis_loss = 0

        if self.loss_type == 'mse_loss':
            policy_loss = torch.square((pred_action - sampled_batch['actions']) * loss_weight).mean()
            # policy_loss = F.mse_loss(pred_action, sampled_batch['actions'])
        elif self.loss_type == 'l1_loss':
            policy_loss = torch.abs((pred_action - sampled_batch['actions']) * loss_weight).mean()
            # policy_loss = F.l1_loss(pred_action, sampled_batch['actions'])
        elif self.loss_type == "cls":
            target_shape = sampled_batch['actions'].shape
            B, K = target_shape
            pred_action = torch.reshape(pred_action, shape=(B * K, -1))
            target_to_label = (sampled_batch['actions'] + 1) / 2 * K
            num_cls = pred_action.shape[-1]
            target_to_label = torch.clamp(target_to_label, min=0, max=num_cls - 1)
            policy_loss = F.cross_entropy(pred_action, target_to_label.type(torch.long).flatten())
            pred_action = (torch.argmax(pred_action, dim=-1) * 1.0 / K) * 2 - 1
            pred_action = torch.reshape(pred_action, shape=(B, K))
        else:
            assert False
        self.policy_optim.zero_grad()
        (policy_loss + dis_loss).backward()
        self.policy_optim.step()
        self.policy_lr_scheduler.step()
        if self.add_c_loss:
            return {
                'policy_abs_error': torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean().item(),
                'policy_loss': policy_loss.item(),
                'dis_loss': dis_loss.item()
            }
        else:
            return {
                'policy_abs_error': torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean().item(),
                'policy_loss': policy_loss.item(),
            }


@BRL.register_module()
class EnsembleBC(BaseAgent):
    def __init__(self, policy_cfg, obs_shape, action_shape, action_space, batch_size=128):
        super(EnsembleBC, self).__init__()
        self.batch_size = batch_size

        policy_optim_cfg = policy_cfg.pop("optim_cfg")
        lr_scheduler_cfg = policy_cfg.pop("lr_scheduler_cfg")

        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space

        self.loss_type = policy_cfg.pop('loss_type')
        if "loss_weight" in policy_cfg:
            self.loss_weight_type = policy_cfg.pop('loss_weight')
        else:
            self.loss_weight_type = None
        if "add_c_loss" in policy_cfg:
            self.add_c_loss = policy_cfg.pop('add_c_loss')
        else:
            self.add_c_loss = False
        if self.add_c_loss:
            self.discriminator = nn.Sequential(
                nn.Linear(action_shape * 2, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 1),
            )
        else:
            self.discriminator = None

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
        policy_losses = 0
        dis_losses = 0

        policy_abs_error = 0
        for i in range(self.policy.ensemble_num):
            sampled_batch = memory.sample(self.batch_size)
            sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"])
            sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
            for key in sampled_batch:
                if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                    sampled_batch[key] = sampled_batch[key][..., None]
            pred_action = self.policy(sampled_batch['obs'], mode='eval', model_idx=i)
            if self.loss_type == "mse_loss" or self.loss_type == "l1_loss":
                if self.loss_weight_type == None:
                    loss_weight = torch.ones_like(pred_action)[:, 0:1] * 1.0
                elif self.loss_weight_type == "hard":
                    with torch.no_grad():
                        pre_loss = torch.square(pred_action - sampled_batch['actions']).sum(dim=-1)
                        loss_weight = pre_loss < torch.median(pre_loss)
                        loss_weight = loss_weight / (loss_weight.sum() + 1e-5) * loss_weight.numel()
                        loss_weight = loss_weight.unsqueeze(dim=-1)
                elif self.loss_weight_type == "soft":
                    with torch.no_grad():
                        loss_weight = torch.abs(pred_action - sampled_batch['actions']).sum(dim=-1)
                        loss_weight = loss_weight.max() * 5 - loss_weight
                        loss_weight = loss_weight / (loss_weight.sum() + 1e-5) * loss_weight.numel()
                        loss_weight = loss_weight.unsqueeze(dim=-1)
                else:
                    raise NotImplementedError
            if self.add_c_loss:
                assert self.loss_type in ['mse_loss', 'l1_loss']
                B, D = pred_action.shape
                input_features = pred_action.reshape(B, 1, D)
                target_features = sampled_batch['actions'].reshape(1, B, D)
                input_features = input_features.expand(B, B, D).reshape(B * B, D)
                target_features = target_features.expand(B, B, D).reshape(B * B, D)
                features = torch.cat((input_features, target_features), dim=-1)
                features_embed = self.discriminator(features).flatten()
                features_embed = features_embed.reshape(B, B)
                dis_loss = F.cross_entropy(features_embed,
                                           torch.arange(0, B, dtype=torch.long, device=features_embed.device))
            else:
                dis_loss = 0

            if self.loss_type == 'mse_loss':
                policy_loss = torch.square((pred_action - sampled_batch['actions']) * loss_weight).mean()
                # policy_loss = F.mse_loss(pred_action, sampled_batch['actions'])
            elif self.loss_type == 'l1_loss':
                policy_loss = torch.abs((pred_action - sampled_batch['actions']) * loss_weight).mean()
                # policy_loss = F.l1_loss(pred_action, sampled_batch['actions'])
            elif self.loss_type == "cls":
                target_shape = sampled_batch['actions'].shape
                B, K = target_shape
                pred_action = torch.reshape(pred_action, shape=(B * K, -1))
                target_to_label = (sampled_batch['actions'] + 1) / 2 * K
                num_cls = pred_action.shape[-1]
                target_to_label = torch.clamp(target_to_label, min=0, max=num_cls - 1)
                policy_loss = F.cross_entropy(pred_action, target_to_label.type(torch.long).flatten())
                pred_action = (torch.argmax(pred_action, dim=-1) * 1.0 / K) * 2 - 1
                pred_action = torch.reshape(pred_action, shape=(B, K))
            else:
                assert False
            policy_losses = policy_losses + policy_loss
            dis_losses = dis_losses + dis_loss
            policy_abs_error += torch.abs(pred_action - sampled_batch['actions']).sum(-1).mean().item()

        self.policy_optim.zero_grad()
        (policy_losses + dis_losses).backward()
        self.policy_optim.step()
        self.policy_lr_scheduler.step()
        if self.add_c_loss:
            return {
                'policy_abs_error': policy_abs_error / self.policy.ensemble_num,
                'policy_loss': policy_losses.item() / self.policy.ensemble_num,
                'dis_loss': dis_losses.item() / self.policy.ensemble_num
            }
        else:
            return {
                'policy_abs_error': policy_abs_error / self.policy.ensemble_num,
                'policy_loss': policy_losses.item() / self.policy.ensemble_num
            }
