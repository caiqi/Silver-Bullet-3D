import os
from copy import deepcopy
import pathlib
from collections import deque

import gym
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from mani_skill_learn.networks import build_model

from mani_skill_learn.env import get_env_info
from mani_skill_learn.methods.builder import build_brl
from mani_skill_learn.utils.data import to_np, unsqueeze
from mani_skill_learn.utils.meta import Config
from mani_skill_learn.utils.torch import load_checkpoint


class ObsProcess:
    # modified from SapienRLWrapper
    def __init__(self, env, obs_mode, stack_frame=1):
        """
        Stack k last frames for point clouds or rgbd
        """
        self.env = env
        self.obs_mode = obs_mode
        self.stack_frame = stack_frame
        self.buffered_data = {}

    def _update_buffer(self, obs):
        for key in obs:
            if key not in self.buffered_data:
                self.buffered_data[key] = deque([obs[key]] * self.stack_frame, maxlen=self.stack_frame)
            else:
                self.buffered_data[key].append(obs[key])

    def _get_buffer_content(self):
        axis = 0 if self.obs_mode == 'pointcloud' else -1
        return {key: np.concatenate(self.buffered_data[key], axis=axis) for key in self.buffered_data}

    def process_observation(self, observation):
        if self.obs_mode == "state":
            return observation
        observation = process_mani_skill_base(observation, "pointcloud")
        visual_data = observation[self.obs_mode]
        self._update_buffer(visual_data)
        visual_data = self._get_buffer_content()
        state = observation['agent']
        ret = {}
        ret[self.obs_mode] = visual_data
        ret['state'] = state
        return ret


class BasePolicy(object):
    def __init__(self, opts=None):
        self.obs_mode = 'pointcloud'

    def act(self, observation):
        raise NotImplementedError()

    def reset(self):  # if you use an RNN-based policy, you need to implement this function
        pass


import numpy as np


def process_mani_skill_base(obs, obs_mode=None):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}

    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    if obs_mode in ['state', 'rgbd']:
        return obs
    elif obs_mode == 'pointcloud':
        rgb = obs[obs_mode]['rgb']
        xyz = obs[obs_mode]['xyz']
        seg = obs[obs_mode]['seg']

        # Given that xyz are already in world-frame, then filter the point clouds that belong to ground.
        mask = (xyz[:, 2] > 1e-3)
        rgb = rgb[mask]
        xyz = xyz[mask]
        seg = seg[mask]

        tot_pts = 1200
        target_mask_pts = 800
        min_pts = 50
        num_pts = np.sum(seg, axis=0)
        tgt_pts = np.array(num_pts)
        # if there are fewer than min_pts points, keep all points
        surplus = np.sum(np.maximum(num_pts - min_pts, 0)) + 1e-6

        # randomly sample from the rest
        sample_pts = target_mask_pts - np.sum(np.minimum(num_pts, min_pts))
        for i in range(seg.shape[1]):
            if num_pts[i] <= min_pts:
                tgt_pts[i] = num_pts[i]
            else:
                tgt_pts[i] = min_pts + int((num_pts[i] - min_pts) / surplus * sample_pts)

        chosen_seg = []
        chosen_rgb = []
        chosen_xyz = []
        chosen_mask_pts = 0
        for i in range(seg.shape[1]):
            if num_pts[i] == 0:
                continue
            cur_seg = np.where(seg[:, i])[0]
            shuffle_indices = np.random.permutation(cur_seg)[:tgt_pts[i]]
            chosen_mask_pts += shuffle_indices.shape[0]
            chosen_seg.append(seg[shuffle_indices])
            chosen_rgb.append(rgb[shuffle_indices])
            chosen_xyz.append(xyz[shuffle_indices])
        sample_background_pts = tot_pts - chosen_mask_pts

        if seg.shape[1] == 1:
            bk_seg = np.logical_not(seg[:, 0])
        else:
            bk_seg = np.logical_not(np.logical_or(*([seg[:, i] for i in range(seg.shape[1])])))
        bk_seg = np.where(bk_seg)[0]
        shuffle_indices = np.random.permutation(bk_seg)[:sample_background_pts]

        chosen_seg.append(seg[shuffle_indices])
        chosen_rgb.append(rgb[shuffle_indices])
        chosen_xyz.append(xyz[shuffle_indices])

        chosen_seg = np.concatenate(chosen_seg, axis=0)
        chosen_rgb = np.concatenate(chosen_rgb, axis=0)
        chosen_xyz = np.concatenate(chosen_xyz, axis=0)
        if chosen_seg.shape[0] < tot_pts:
            pad_pts = tot_pts - chosen_seg.shape[0]
            chosen_seg = np.concatenate([chosen_seg, np.zeros([pad_pts, chosen_seg.shape[1]]).astype(chosen_seg.dtype)],
                                        axis=0)
            chosen_rgb = np.concatenate([chosen_rgb, np.zeros([pad_pts, chosen_rgb.shape[1]]).astype(chosen_rgb.dtype)],
                                        axis=0)
            chosen_xyz = np.concatenate([chosen_xyz, np.zeros([pad_pts, chosen_xyz.shape[1]]).astype(chosen_xyz.dtype)],
                                        axis=0)
        obs[obs_mode]['seg'] = chosen_seg
        obs[obs_mode]['xyz'] = chosen_xyz
        obs[obs_mode]['rgb'] = chosen_rgb
        return obs
    else:
        print(f'Unknown observation mode {obs_mode}')
        exit(0)


class UserPolicy1(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        env = gym.make(env_name)
        config_path = './models/1201_door_dt/1201_door_dt_resume_pformer.py'
        resume_from = './models/1201_door_dt/model_105000.ckpt'

        cfg = Config.fromfile(config_path)
        cfg.resume_from = resume_from
        eval_cfg = cfg.eval_cfg
        eval_cfg['env_cfg'] = deepcopy(cfg.env_cfg)
        obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        policy_cfg = cfg.agent['policy_cfg']
        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space
        policy_cfg.pop("optim_cfg")
        policy = build_model(policy_cfg)
        checkpoint = torch.load(cfg.resume_from, map_location="cpu")
        checkpoint['state_dict'] = {k.replace("policy.", ""): v for k, v in checkpoint['state_dict'].items()}
        torch.save(checkpoint, cfg.resume_from)
        load_checkpoint(policy, cfg.resume_from, map_location='cpu')

        policy = policy.cuda()
        policy.eval()
        self.policy = policy
        env.close()
        del env
        self.obs_mode = 'pointcloud'  # remember to set this!

    def reset(self):
        self.states = torch.zeros((0, self.policy.backbone.out_dim), device="cuda", dtype=torch.float32)

    def act(self, observation):
        observation = process_mani_skill_base(observation, obs_mode="pointcloud")
        observation['state'] = torch.from_numpy(observation['agent']).type(torch.float32).cuda().unsqueeze(dim=0)
        observation.pop("agent")
        observation['pointcloud']['rgb'] = torch.from_numpy(observation['pointcloud']['rgb']).type(
            torch.float32).cuda().unsqueeze(dim=0)
        observation['pointcloud']['xyz'] = torch.from_numpy(observation['pointcloud']['xyz']).type(
            torch.float32).cuda().unsqueeze(dim=0)
        observation['pointcloud']['seg'] = torch.from_numpy(observation['pointcloud']['seg']).type(
            torch.float32).cuda().unsqueeze(dim=0)
        input_data = {
            "states": observation,
            "dt_states": self.states,
            "actions": self.states.detach().clone(),
            "rewards": None,
            "target_return": None,
            "timesteps": None
        }
        with torch.no_grad():
            action, self.states = self.policy(input_data, mode="eval")
            output = action.cpu().numpy()
        return output


class UserPolicy2(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        ##### Replace with your code
        env = gym.make(env_name)

        config_path = './models/submission_1125/door_v1.py'
        resume_from = './models/submission_1125/model_170000.ckpt'

        cfg = Config.fromfile(config_path)
        cfg.resume_from = resume_from
        eval_cfg = cfg.eval_cfg
        eval_cfg['env_cfg'] = deepcopy(cfg.env_cfg)
        obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        policy_cfg = cfg.agent['policy_cfg']
        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space
        policy_cfg.pop("optim_cfg")
        policy = build_model(policy_cfg)
        checkpoint = torch.load(cfg.resume_from, map_location="cpu")
        checkpoint['state_dict'] = {k.replace("policy.", ""): v for k, v in checkpoint['state_dict'].items()}
        torch.save(checkpoint, cfg.resume_from)
        load_checkpoint(policy, cfg.resume_from, map_location='cpu')
        policy = policy.cuda()
        policy.eval()
        self.policy = policy
        env.close()
        del env
        self.obs_mode = 'pointcloud'  # remember to set this!

    def reset(self):
        self.states = torch.zeros((0, self.policy.backbone.out_dim), device="cuda", dtype=torch.float32)

    def act(self, observation):
        observation = process_mani_skill_base(observation, obs_mode="pointcloud")
        observation['state'] = torch.from_numpy(observation['agent']).type(torch.float32).cuda().unsqueeze(dim=0)
        observation.pop("agent")
        observation['pointcloud']['rgb'] = torch.from_numpy(observation['pointcloud']['rgb']).type(
            torch.float32).cuda().unsqueeze(dim=0)
        observation['pointcloud']['xyz'] = torch.from_numpy(observation['pointcloud']['xyz']).type(
            torch.float32).cuda().unsqueeze(dim=0)
        observation['pointcloud']['seg'] = torch.from_numpy(observation['pointcloud']['seg']).type(
            torch.float32).cuda().unsqueeze(dim=0)
        input_data = {
            "states": observation,
            "dt_states": self.states,
            "actions": self.states.detach().clone(),
            "rewards": None,
            "target_return": None,
            "timesteps": None
        }
        with torch.no_grad():
            action, self.states = self.policy(input_data, mode="eval")
            output = action.cpu().numpy()
        return output


class UserPolicy3(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.env = gym.make(env_name)
        self.obs_mode = 'pointcloud'  # remember to set this!
        self.env.set_env_mode(obs_mode=self.obs_mode)
        self.stack_frame = 1

        cfg_path = str(pathlib.Path('./models/yehao_v3/drawer_door_pointformer_embed_conv.py').resolve())
        cfg = Config.fromfile(cfg_path)
        cfg.env_cfg['env_name'] = env_name
        obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        cfg.agent['obs_shape'] = obs_shape
        cfg.agent['action_shape'] = action_shape
        cfg.agent['action_space'] = action_space

        self.agent = build_brl(cfg.agent)
        load_checkpoint(self.agent,
                        str(pathlib.Path('./models/yehao_v3/model_600000.ckpt').resolve()),
                        map_location='cpu'
                        )
        self.agent.to('cuda')  # dataparallel not done here
        self.agent.eval()

        self.obsprocess = ObsProcess(self.env, self.obs_mode, self.stack_frame)

    def act(self, observation):
        observation = self.obsprocess.process_observation(observation)
        return to_np(self.agent(unsqueeze(observation, axis=0), mode='eval'))[0]


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.policy1 = UserPolicy1(env_name)
        self.policy2 = UserPolicy2(env_name)
        self.policy3 = UserPolicy3(env_name)

    def reset(self):
        self.policy1.reset()
        self.policy2.reset()
        self.policy3.reset()

    def act(self, observation):
        ##### Replace with your code
        action1 = self.policy1.act(deepcopy(observation))
        action2 = self.policy2.act(deepcopy(observation))
        action3 = self.policy3.act(deepcopy(observation))
        action = (action1 + action2 + action3) / 3.
        return action
