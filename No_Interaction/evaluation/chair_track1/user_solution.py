import numpy as np
import random
from mani_skill_learn.utils.torch import load_checkpoint
from mani_skill_learn.utils.meta import Config
from mani_skill_learn.utils.data import to_np, unsqueeze
from mani_skill_learn.methods.builder import build_brl
from mani_skill_learn.env import get_env_info
from mani_skill_learn.networks import build_model
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

GLOBAL_ENV_NAME = None


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
                self.buffered_data[key] = deque(
                    [obs[key]] * self.stack_frame, maxlen=self.stack_frame)
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


def process_mani_skill_base(obs, obs_mode=None):
    assert GLOBAL_ENV_NAME is not None
    if GLOBAL_ENV_NAME == "door" or GLOBAL_ENV_NAME == "drawer":
        return process_mani_skill_door_door(obs, obs_mode=obs_mode)
    elif GLOBAL_ENV_NAME == "bucket":
        return process_mani_skill_bucket(obs, obs_mode=obs_mode)
    elif GLOBAL_ENV_NAME == "chair":
        return process_mani_skill_chair(obs, obs_mode=obs_mode)
    else:
        raise NotImplementedError


def process_mani_skill_door_door(obs, obs_mode=None):
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
                tgt_pts[i] = min_pts + \
                    int((num_pts[i] - min_pts) / surplus * sample_pts)

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
            bk_seg = np.logical_not(np.logical_or(
                *([seg[:, i] for i in range(seg.shape[1])])))
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


def process_mani_skill_chair(obs, obs_mode=None):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}
    random.seed(1024)
    np.random.seed(1024)
    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    if obs_mode in ['state', 'rgbd']:
        return obs
    elif obs_mode == 'pointcloud':
        rgb = obs[obs_mode]['rgb']
        xyz = obs[obs_mode]['xyz']
        seg = obs[obs_mode]['seg']

        mask = (xyz[:, 2] > 1e-3)
        rgb = rgb[mask]
        xyz = xyz[mask]
        seg = seg[mask]

        #mask = (rgb[:, 0] - rgb[:, 1] > 0.7) & (rgb[:, 0] - rgb[:, 2] > 0.7) & (xyz[:, 2] < 0.2)
        mask = (xyz[:, 2] <= 0.15) & (rgb[:, 0] > 0.6) & (
            rgb[:, 1] < 0.05) & (rgb[:, 2] < 0.05)
        extra_rgb = rgb[mask]
        extra_xyz = xyz[mask]
        extra_seg = seg[mask]
        xyz = xyz[~mask]
        rgb = rgb[~mask]
        seg = seg[~mask]
        red_point_num = 50
        if extra_xyz.shape[0] > red_point_num:
            random_index = np.arange(0, extra_xyz.shape[0])
            np.random.shuffle(random_index)
            part_1 = random_index[:red_point_num]
            #part_2 = random_index[red_point_num:]
            #xyz = np.concatenate((xyz, extra_xyz[part_2]),axis=0)
            #rgb = np.concatenate((rgb, extra_rgb[part_2]),axis=0)
            #seg = np.concatenate((seg, extra_seg[part_2]),axis=0)
            extra_xyz = extra_xyz[part_1]
            extra_rgb = extra_rgb[part_1]
            extra_seg = extra_seg[part_1]

        tot_pts = 1200 - extra_rgb.shape[0]
        target_mask_pts = 700
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
                tgt_pts[i] = min_pts + \
                    int((num_pts[i] - min_pts) / surplus * sample_pts)

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
            bk_seg = np.logical_not(np.logical_or(
                *([seg[:, i] for i in range(seg.shape[1])])))
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

        chosen_rgb = np.concatenate((chosen_rgb, extra_rgb), axis=0)
        chosen_xyz = np.concatenate((chosen_xyz, extra_xyz), axis=0)
        chosen_seg = np.concatenate((chosen_seg, extra_seg), axis=0)

        obs[obs_mode]['seg'] = chosen_seg
        obs[obs_mode]['xyz'] = chosen_xyz
        obs[obs_mode]['rgb'] = chosen_rgb
        return obs
    else:
        print(f'Unknown observation mode {obs_mode}')
        exit(0)


def process_mani_skill_bucket(obs, obs_mode=None):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}
    random.seed(1024)
    np.random.seed(1024)
    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    if obs_mode in ['state', 'rgbd']:
        return obs
    elif obs_mode == 'pointcloud':
        rgb = obs[obs_mode]['rgb']
        xyz = obs[obs_mode]['xyz']
        seg = obs[obs_mode]['seg']

        mask = (xyz[:, 2] > 1e-3)
        rgb = rgb[mask]
        xyz = xyz[mask]
        seg = seg[mask]

        #mask = (rgb[:, 0] - rgb[:, 1] > 0.7) & (rgb[:, 0] - rgb[:, 2] > 0.7) & (xyz[:, 2] < 0.2)
        #mask = (xyz[:, 2] <= 0.15) & (rgb[:, 0] > 0.6) & (rgb[:, 1] < 0.05) & (rgb[:, 2] < 0.05)
        noise = np.arange(xyz.shape[0])
        z = np.round(xyz[:, 2]*100) + seg[:, 0] * noise
        #z_mode = statistics.mode(z)

        values, counts = np.unique(z, return_counts=True)
        ind = np.argmax(counts)
        z_mode = z[ind]

        mask = abs(z - z_mode) < 0.1
        if abs(z_mode - 10) > 0.1:
            mask2 = abs(z - 10) < 0.1
            if mask2.sum() / mask.sum() > 0.4:
                mask = mask2

        extra_rgb = rgb[mask]
        extra_xyz = xyz[mask]
        extra_seg = seg[mask]
        xyz = xyz[~mask]
        rgb = rgb[~mask]
        seg = seg[~mask]
        red_point_num = 70
        if extra_xyz.shape[0] > red_point_num:
            random_index = np.arange(0, extra_xyz.shape[0])
            np.random.shuffle(random_index)
            part_1 = random_index[:red_point_num]
            part_2 = random_index[red_point_num:]
            xyz = np.concatenate((xyz, extra_xyz[part_2]), axis=0)
            rgb = np.concatenate((rgb, extra_rgb[part_2]), axis=0)
            seg = np.concatenate((seg, extra_seg[part_2]), axis=0)
            extra_xyz = extra_xyz[part_1]
            extra_rgb = extra_rgb[part_1]
            extra_seg = extra_seg[part_1]

        tot_pts = 1200 - extra_rgb.shape[0]
        target_mask_pts = 700
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
                tgt_pts[i] = min_pts + \
                    int((num_pts[i] - min_pts) / surplus * sample_pts)

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
            bk_seg = np.logical_not(np.logical_or(
                *([seg[:, i] for i in range(seg.shape[1])])))
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

        chosen_rgb = np.concatenate((chosen_rgb, extra_rgb), axis=0)
        chosen_xyz = np.concatenate((chosen_xyz, extra_xyz), axis=0)
        chosen_seg = np.concatenate((chosen_seg, extra_seg), axis=0)

        obs[obs_mode]['seg'] = chosen_seg
        obs[obs_mode]['xyz'] = chosen_xyz
        obs[obs_mode]['rgb'] = chosen_rgb

        return obs
    else:
        print(f'Unknown observation mode {obs_mode}')
        exit(0)


class TemplatePolicy_BC(BasePolicy):
    def __init__(self, env_name, config_file_path, model_path):
        super().__init__()
        # config_file_path = './models/ManiSkillModels3/bucket/ensemble_bucket_pointformer_embed_conv.py'
        # model_path = "./models/ManiSkillModels3/bucket/model_300000.ckpt"
        global GLOBAL_ENV_NAME
        if "Chair" in env_name:
            GLOBAL_ENV_NAME = "chair"
        elif "Bucket" in env_name:
            GLOBAL_ENV_NAME = "bucket"
        elif "Drawer" in env_name:
            GLOBAL_ENV_NAME = "drawer"
        elif "Door" in env_name:
            GLOBAL_ENV_NAME = "door"
        else:
            raise NotImplementedError

        self.env = gym.make(env_name)
        self.obs_mode = 'pointcloud'  # remember to set this!
        self.env.set_env_mode(obs_mode=self.obs_mode)
        self.stack_frame = 1

        cfg_path = str(pathlib.Path(config_file_path).resolve())
        cfg = Config.fromfile(cfg_path)
        cfg.env_cfg['env_name'] = env_name
        obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        cfg.agent['obs_shape'] = obs_shape
        cfg.agent['action_shape'] = action_shape
        cfg.agent['action_space'] = action_space

        self.agent = build_brl(cfg.agent)
        load_checkpoint(self.agent,
                        str(pathlib.Path(model_path).resolve()),
                        map_location='cpu'
                        )
        self.agent.to('cuda')
        self.agent.eval()

        self.obsprocess = ObsProcess(self.env, self.obs_mode, self.stack_frame)

    def act(self, observation):
        with torch.no_grad():
            observation = self.obsprocess.process_observation(observation)
            return to_np(self.agent(unsqueeze(observation, axis=0), mode='eval'))[0]


class TemplatePolicy_DT(BasePolicy):
    def __init__(self, env_name, config_file_path, model_path):
        super().__init__()
        # config_file_path = './models/ManiSkillModels3/bucket/ensemble_bucket_pointformer_embed_conv.py'
        # model_path = "./models/ManiSkillModels3/bucket/model_300000.ckpt"
        global GLOBAL_ENV_NAME
        if "Chair" in env_name:
            GLOBAL_ENV_NAME = "chair"
        elif "Bucket" in env_name:
            GLOBAL_ENV_NAME = "bucket"
        elif "Drawer" in env_name:
            GLOBAL_ENV_NAME = "drawer"
        elif "Door" in env_name:
            GLOBAL_ENV_NAME = "door"
        else:
            raise NotImplementedError
        # config_path = './models/ManiSkillModels3/bucket_dt/1209_bucket_ensemble_v1.py'
        # resume_from = './models/ManiSkillModels3/bucket_dt/model_30000.ckpt'
        cfg = Config.fromfile(config_file_path)
        cfg.resume_from = model_path
        eval_cfg = cfg.eval_cfg
        eval_cfg['env_cfg'] = deepcopy(cfg.env_cfg)
        obs_shape, action_shape, action_space = get_env_info(cfg.env_cfg)
        policy_cfg = cfg.agent['policy_cfg']
        policy_cfg['obs_shape'] = obs_shape
        policy_cfg['action_shape'] = action_shape
        policy_cfg['action_space'] = action_space
        policy_cfg.pop("optim_cfg")
        policy_cfg.pop("lr_scheduler_cfg")
        policy = build_model(policy_cfg)
        checkpoint = torch.load(cfg.resume_from, map_location="cpu")
        checkpoint['state_dict'] = {
            k.replace("policy.", ""): v for k, v in checkpoint['state_dict'].items()}
        torch.save(checkpoint, cfg.resume_from)
        load_checkpoint(policy, cfg.resume_from, map_location='cpu')
        policy = policy.cuda()
        policy.eval()
        self.policy = policy
        self.obs_mode = 'pointcloud'  # remember to set this!

    def reset(self):
        self.states = torch.zeros(
            (0, self.policy.backbone.out_dim), device="cuda", dtype=torch.float32)

    def act(self, observation):
        observation = process_mani_skill_base(
            observation, obs_mode="pointcloud")
        observation['state'] = torch.from_numpy(observation['agent']).type(
            torch.float32).cuda().unsqueeze(dim=0)
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


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        # bc_models = [
        #     ["models_1230/1225_chair_ensemble_v1/EnsembleBC/1225_chair_ensemble_v1.py", "models_1230/1225_chair_ensemble_v1/EnsembleBC/models/model_900000.ckpt"]]
        # dt_models = [
        #     ["models_1230/1209_chair_ensemble_v1/EnsembleDT/1209_chair_ensemble_v1.py", "models_1230/1209_chair_ensemble_v1/EnsembleDT/models/model_750000.ckpt"]]
        bc_models = [
            ["models_1230/1225_chair_ensemble_v1/EnsembleBC/1225_chair_ensemble_v1.py",
                "models_1230/1225_chair_ensemble_v1/EnsembleBC/models/model_600000.ckpt"],

            ["models_1230/1225_chair_ensemble_v1/EnsembleBC/1225_chair_ensemble_v1.py",
                "models_1230/1225_chair_ensemble_v2/EnsembleBC/models/model_600000.ckpt"],

            ["models_1230/1225_chair_ensemble_v1/EnsembleBC/1225_chair_ensemble_v1.py",
                "models_1230/1225_chair_ensemble_v3/EnsembleBC/models/model_600000.ckpt"],

            ["models_1230/1225_chair_ensemble_v1/EnsembleBC/1225_chair_ensemble_v1.py",
                "models_1230/1225_chair_ensemble_v4/EnsembleBC/models/model_600000.ckpt"],

            ["models_1230/ManiSkillModels3/chair/ensemble_chair_pointformer_raw_conv.py",
                "models_1230/ManiSkillModels3/chair/model_600000.ckpt"],
        ]
        # bc_models = []
        dt_models = [
            ["models_1230/1209_chair_ensemble_v1/EnsembleDT/1209_chair_ensemble_v1.py",
                "models_1230/1209_chair_ensemble_v1/EnsembleDT/models/model_600000.ckpt"],
        ]

        # dt_models = []
        bc_agents = []
        for m in bc_models:
            bc_ = TemplatePolicy_BC(env_name, m[0], m[1])
            bc_agents.append(bc_)
        dt_agents = []
        for m in dt_models:
            dt_ = TemplatePolicy_DT(env_name, m[0], m[1])
            dt_agents.append(dt_)
        print("BC agents: {} DT agents: {}".format(
            len(bc_agents), len(dt_agents)))
        self.bc_agents = bc_agents
        self.dt_agents = dt_agents

    def reset(self):
        for m in self.bc_agents:
            m.reset()
        for m in self.dt_agents:
            m.reset()

    def act(self, observation):
        action_list = []
        for m in self.bc_agents:
            action = m.act(deepcopy(observation))
            action_list.append(action)
        for m in self.dt_agents:
            action = m.act(deepcopy(observation))
            action_list.append(action)
        assert len(action_list) >= 1
        return sum(action_list) / len(action_list)


