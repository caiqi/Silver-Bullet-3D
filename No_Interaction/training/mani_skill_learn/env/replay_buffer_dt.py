import random

import numpy as np
import torch
from tqdm import tqdm

from mani_skill_learn.utils.data import (store_dict_array_to_h5,
                                         is_seq_of)
from mani_skill_learn.utils.fileio import load_h5s_as_list_dict_array
from .builder import REPLAYS


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


@REPLAYS.register_module()
class ReplayMemoryDT:
    """
    Replay buffer uses dict-array as basic data structure, which can be easily saved as hdf5 file.
    
    See mani_skill_learn/utils/data/dict_array.py for more details.
    """

    def __init__(self, capacity, buffer_keys=None, max_ep_len=50, scale=100., K=20, mode="delayed", compressed=False):
        if buffer_keys is None:
            self.buffer_keys = ['obs', 'actions', 'next_obs', 'rewards', 'dones']
        else:
            self.buffer_keys = buffer_keys
        self.max_ep_len = max_ep_len
        self.capacity = capacity
        self.compressed = compressed

        self.scale = scale
        self.K = K
        self.memory = []
        self.position = 0
        self.running_count = 0
        self.p_sample = None
        self.mode = mode

    def __len__(self):
        return min(self.running_count, self.capacity)

    def reset(self):
        self.memory = []
        self.position = 0
        self.running_count = 0

    def push(self, **kwargs):
        # assert not self.fixed, "Fix replay buffer does not support adding items!"
        self.running_count += 1
        self.position = (self.position + 1) % self.capacity
        kwargs = dict(kwargs)
        self.memory.append(kwargs)

    def get_p_sample(self):
        if self.p_sample is None:
            states, traj_lens, returns = [], [], []
            for path in self.memory:
                if self.mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                    path['rewards'][-1] = path['rewards'].sum()
                    path['rewards'][:-1] = 0.
                states.append(path['obs'])
                traj_lens.append(len(path['actions']))
                returns.append(path['rewards'].sum())
            traj_lens, returns = np.array(traj_lens), np.array(returns)
            self.sorted_inds = np.argsort(returns)  # lowest to highest
            self.p_sample = traj_lens[self.sorted_inds] / sum(traj_lens[self.sorted_inds])
        return self.p_sample

    def sample(self, batch_size=256, max_len=None):
        if max_len is None:
            max_len = self.K

        batch_inds = np.random.choice(
            np.arange(len(self)),
            size=batch_size,
            replace=True,
            p=self.get_p_sample(),  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, agent = [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = self.memory[int(self.sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # get sequences from dataset
            s_points = traj['obs']['pointcloud']['xyz'][si:si + max_len]
            s_rgb = traj['obs']['pointcloud']['rgb'][si:si + max_len]
            if self.compressed:
                s_rgb = s_rgb.astype(np.float32) / 255.0
            s_seg = traj['obs']['pointcloud']['seg'][si:si + max_len]
            s_feature = np.concatenate((s_points, s_rgb, s_seg), axis=-1)
            s.append(s_feature)
            agent.append(traj['obs']['state'][si:si + max_len])
            a.append(traj['actions'][si:si + max_len])
            r.append(traj['rewards'][si:si + max_len])
            d.append(traj['dones'][si:si + max_len])
            timesteps.append(np.arange(si, si + s[-1].shape[0]))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[0] + 1].reshape(-1, 1))
            if rtg[-1].shape[0] <= s[-1].shape[0]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1))], axis=0)
            # padding and state + reward normalization
            tlen = s[-1].shape[0]
            s[-1] = np.concatenate([s[-1], np.zeros((max_len - tlen, *(s[-1].shape[1:])))], axis=0)
            agent[-1] = np.concatenate([agent[-1], np.zeros((max_len - tlen, *(agent[-1].shape[1:])))], axis=0)
            a[-1] = np.concatenate([a[-1], np.ones((max_len - tlen, *(a[-1].shape[1:]))) * -10.], axis=0)
            r[-1] = np.concatenate([r[-1], np.zeros(max_len - tlen)], axis=0)
            d[-1] = np.concatenate([d[-1], np.ones(max_len - tlen) * 2], axis=0)
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((max_len - tlen, *(rtg[-1].shape[1:])))], axis=0) / self.scale
            timesteps[-1] = np.concatenate([timesteps[-1], np.zeros((max_len - tlen))], axis=0)
            mask.append(np.concatenate([np.ones((tlen)), np.zeros((max_len - tlen))], axis=0))

        s = torch.from_numpy(np.stack(s, axis=0)).to(dtype=torch.float32, device="cuda")
        agent = torch.from_numpy(np.stack(agent, axis=0)).to(dtype=torch.float32, device="cuda")
        a = torch.from_numpy(np.stack(a, axis=0)).to(dtype=torch.float32, device="cuda")
        r = torch.from_numpy(np.stack(r, axis=0)).to(dtype=torch.float32, device="cuda")
        d = torch.from_numpy(np.stack(d, axis=0)).to(dtype=torch.long, device="cuda")
        rtg = torch.from_numpy(np.stack(rtg, axis=0)).to(dtype=torch.float32, device="cuda")
        timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).to(dtype=torch.long, device="cuda")
        mask = torch.from_numpy(np.stack(mask, axis=0)).to(device="cuda")
        return dict(
            s=s,
            a=a,
            r=r,
            d=d,
            rtg=rtg,
            timesteps=timesteps,
            mask=mask,
            agent=agent
        )

    def get_all(self):
        return self.memory

    def to_h5(self, file, with_traj_index=False):
        from h5py import File
        data = self.get_all()
        if with_traj_index:
            data = {'traj_0': data}
        if isinstance(file, str):
            with File(file, 'w') as f:
                store_dict_array_to_h5(data, f)
        else:
            store_dict_array_to_h5(data, file)

    def restore(self, init_buffers, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
        buffer_keys = self.buffer_keys
        # init_buffers = init_buffers[:2]
        if isinstance(init_buffers, list):
            init_buffers = [k for k in init_buffers if "state.h5" not in k]
        if isinstance(init_buffers, str):
            init_buffers = [init_buffers]
        if is_seq_of(init_buffers, str):
            init_buffers_new = []
            for _ in tqdm(init_buffers, total=len(init_buffers), desc="loading h5py files"):
                init_buffers_new.append(load_h5s_as_list_dict_array(_))
            init_buffers = init_buffers_new
        if isinstance(init_buffers, dict):
            init_buffers = [init_buffers]
        print('Num of datasets', len(init_buffers))
        for _ in range(replicate_init_buffer):
            cnt = 0
            for init_buffer in tqdm(init_buffers, total=len(init_buffers), desc="loading dataset"):
                for item in init_buffer:
                    if cnt >= num_trajs_per_demo_file and num_trajs_per_demo_file != -1:
                        break
                    item = {key: item[key] for key in buffer_keys}
                    self.push(**item)
                    cnt += 1
        print(f'Num of buffers {len(init_buffers)}, Total steps {self.running_count}')
