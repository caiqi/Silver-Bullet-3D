import warnings

import torch

warnings.simplefilter(action='ignore')

import copy, cv2, numpy as np, os, os.path as osp, shutil
from h5py import File
from .env_utils import build_env, true_done
from .replay_buffer import ReplayMemory
from .env_utils import get_env_state
from mani_skill_learn.utils.fileio import dump
from mani_skill_learn.utils.data import (flatten_dict, to_np, compress_size, dict_to_str, number_to_str)
from mani_skill_learn.utils.meta import get_logger, get_total_memory, flush_print
from .builder import EVALUATIONS


def save_eval_statistics(folder, lengths, rewards, finishes, logger=None):
    if logger is not None:
        logger.info(f'Num of trails: {len(lengths):.2f}, '
                    f'Length: {np.mean(lengths):.2f}+/-{np.std(lengths):.2f}, '
                    f'Reward: {np.mean(rewards):.2f}+/-{np.std(rewards):.2f}, '
                    f'Success or Early Stop Rate: {np.mean(finishes):.2f}')
    if folder is not None:
        table = [['length', 'reward', 'finish']]
        table += [[number_to_str(__, 2) for __ in _] for _ in zip(lengths, rewards, finishes)]
        dump(table, osp.join(folder, 'statistics.csv'))


@EVALUATIONS.register_module()
class EvaluationDT:
    def __init__(self, env_cfg, worker_id=None, save_traj=True, save_video=True, use_hidden_state=False, horizon=None,
                 use_log=True, log_every_step=False, sample_mode='eval', **kwargs):
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg['unwrapped'] = False
        self.env = build_env(env_cfg)
        if 'PushChair' in env_cfg.env_name:
            self.env.observation_type = 'chair'
        elif 'MoveBucket' in env_cfg.env_name:
            self.env.observation_type = 'bucket'
        self.env.reset()
        try:
            self.horizon = self.env.env.env._max_episode_steps if horizon is None else horizon
        except:
            self.horizon = self.env._max_episode_steps if horizon is None else horizon

        self.env_name = env_cfg.env_name
        self.worker_id = worker_id
        self.save_traj = save_traj
        self.save_video = save_video
        self.should_print = self.worker_id is None or self.worker_id == 0
        self.use_log = use_log and self.should_print
        self.log_every_step = self.use_log and log_every_step
        self.use_hidden_state = use_hidden_state
        self.sample_mode = sample_mode

        self.work_dir, self.video_dir, self.trajectory_path = None, None, None
        self.h5_file = None

        self.logger = flush_print
        self.episode_id = 0
        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        self.recent_obs = None
        self.data_episode = None
        self.video_writer = None
        self.video_file = None

        assert not (self.use_hidden_state and worker_id is not None), "Use hidden state is only for CEM evaluation!!"
        assert self.horizon is not None and self.horizon, f"{self.horizon}"
        assert self.worker_id is None or not use_hidden_state, "Parallel evaluation does not support hidden states!"

        if save_video:
            # Use rendering with use additional 1Gi memory in sapien
            image = self.env.render('rgb_array')
            if self.should_print:
                self.logger(f'Size of image in the rendered video {image.shape}')

        if hasattr(self.env, 'seed'):
            # Make sure that envs in different processes have different behaviors
            # self.env.seed(np.random.randint(0, 10000) + os.getpid())
            self.env.seed(1024)

    def start(self, work_dir=None):
        if work_dir is not None:
            self.work_dir = work_dir if self.worker_id is None else os.path.join(work_dir, f'thread_{self.worker_id}')
            shutil.rmtree(self.work_dir, ignore_errors=True)
            if self.save_video:
                self.video_dir = osp.join(self.work_dir, 'videos')
                os.makedirs(self.video_dir, exist_ok=True)
            if self.save_traj:
                os.makedirs(self.work_dir, exist_ok=True)
                self.trajectory_path = osp.join(self.work_dir, 'trajectory.h5')
                self.h5_file = File(self.trajectory_path, 'w')
                if self.should_print:
                    self.logger(f"Save trajectory at {self.trajectory_path}.")

        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.recent_obs = None
        self.data_episode = None
        self.video_writer = None
        if self.should_print:
            self.logger("Begin to evaluate")
        self.episode_id, self.episode_len, self.episode_reward, self.episode_finish = 0, 0, 0, False
        self.recent_obs = self.env.reset()

    def done(self):
        self.episode_lens.append(self.episode_len)
        self.episode_rewards.append(self.episode_reward)
        self.episode_finishes.append(self.episode_finish)

        if self.save_traj and self.data_episode is not None:
            group = self.h5_file.create_group(f'traj_{self.episode_id}')
            self.data_episode.to_h5(group, with_traj_index=False)
            self.data_episode = None
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def reset(self):
        self.episode_id += 1
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        self.recent_obs = self.env.reset()

    def step(self, action):
        data_to_store = {'obs': self.recent_obs}

        if self.save_traj:
            env_state = get_env_state(self.env)
            for key in env_state:
                data_to_store[key] = env_state[key]
            data_to_store.update(env_state)
        if self.save_video:
            image = self.env.render(mode='rgb_array')
            image = image[..., ::-1]
            if self.video_writer is None:
                self.video_file = osp.join(self.video_dir, f'{self.episode_id}.mp4')
                self.video_writer = cv2.VideoWriter(self.video_file, cv2.VideoWriter_fourcc(*'mp4v'), 20,
                                                    (image.shape[1], image.shape[0]))
            self.video_writer.write(image)

        next_obs, reward, done, info = self.env.step(action)
        self.episode_len += 1
        self.episode_reward += reward

        if self.log_every_step:
            self.logger(f'Episode {self.episode_id}: Step {self.episode_len} reward: {reward}, info: {info}')

        episode_done = done
        done = true_done(done, info)
        info = {'info': info}

        if self.save_traj:
            data_to_store['actions'] = compress_size(action)
            data_to_store['next_obs'] = compress_size(next_obs)
            data_to_store['rewards'] = compress_size(reward)
            data_to_store['dones'] = done
            data_to_store['episode_dones'] = episode_done
            data_to_store.update(compress_size(to_np(flatten_dict(info))))
            env_state = get_env_state(self.env)
            for key in env_state:
                data_to_store[f'next_{key}'] = env_state[key]
            """
            from mani_skill_learn.utils.data import get_shape_and_type
            print(list(data_to_store.keys()))
            for key in data_to_store.keys():
                print(key, get_shape_and_type(data_to_store[key]))
            exit(0)
            print(get_shape_and_type(data_to_store))
            exit(0)
            """
            if self.data_episode is None:
                self.data_episode = ReplayMemory(self.horizon)
            self.data_episode.push(**data_to_store)
        if episode_done:
            if self.use_log:
                self.logger(f'Episode {self.episode_id}: Length {self.episode_len} Reward: {self.episode_reward}')
            self.episode_finish = done
            self.done()
            self.reset()
        else:
            self.recent_obs = next_obs
        return episode_done

    def finish(self):
        if self.save_traj:
            self.h5_file.close()

    def single_epoch_evaluation(self, pi):
        pi.eval()
        pi.to(device="cuda")
        from mani_skill_learn.utils.torch import get_cuda_info
        state = self.env.reset()
        states = torch.zeros((0, pi.policy.backbone.out_dim), device="cuda", dtype=torch.float32)
        actions = torch.zeros((0, pi.policy.action_shape), device="cuda", dtype=torch.float32)
        done = False
        for t in range(pi.policy.dt_max_ep_len):
            # add padding
            actions = torch.cat([actions, torch.zeros((1, pi.policy.action_shape), device="cuda")], dim=0)
            state['pointcloud']['xyz'] = np.expand_dims(state['pointcloud']['xyz'], 0)
            state['pointcloud']['rgb'] = np.expand_dims(state['pointcloud']['rgb'], 0)
            state['pointcloud']['seg'] = np.expand_dims(state['pointcloud']['seg'], 0)
            state['state'] = np.expand_dims(state['state'], 0)
            action, states = pi(
                {
                    "states": state,
                    "dt_states": states,
                    "actions": actions,
                },
                mode="eval"
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)
            # self.env.render("human")
            self.episode_reward += reward
            self.episode_len += 1

            if done:
                break
        if self.use_log:
            self.logger(f'Episode {self.episode_id}: Length {self.episode_len} Reward: {self.episode_reward}')
        self.episode_finish = done
        self.done()
        self.reset()
        if self.use_log:
            print_dict = {}
            print_dict['memory'] = get_total_memory('G', False)
            print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))
            print_info = dict_to_str(print_dict)
            self.logger(f'{print_info}')

    def run(self, pi, num=1, work_dir=None, **kwargs):
        if self.worker_id is None:
            self.logger = get_logger(self.env_name).info
        self.start(work_dir)

        def reset_pi():
            if hasattr(pi, 'reset'):
                assert self.worker_id is None, "Reset policy only works for single thread!"
                reset_kwargs = {}
                if hasattr(self.env, 'level'):
                    # When we run CEM, we need the level of the rollout env to match the level of test env.
                    if self.should_print:
                        self.logger(f"Episode {self.episode_id}, run on level {self.env.level}")
                    reset_kwargs['level'] = self.env.level
                pi.reset(**reset_kwargs)  # Design for recurrent policy and CEM.

        reset_pi()
        for self.episode_id in range(num):
            with torch.no_grad():
                self.single_epoch_evaluation(pi)
        self.finish()
        return self.episode_lens, self.episode_rewards, self.episode_finishes

    def close(self):
        if hasattr(self, 'env'):
            del self.env
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()

    def __del__(self):
        self.close()
