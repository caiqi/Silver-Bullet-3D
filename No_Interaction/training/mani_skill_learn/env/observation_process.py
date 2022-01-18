import numpy as np
import random
import statistics

def process_mani_skill_base(obs, env=None):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}
    random.seed(1024)
    np.random.seed(1024)
    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    obs_mode = env.obs_mode
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

def bucket_process_mani_skill_base(obs, env=None):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}
    random.seed(1024)
    np.random.seed(1024)
    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    obs_mode = env.obs_mode
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
        z = np.round(xyz[:,2]*100) + seg[:, 0] * noise
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
            xyz = np.concatenate((xyz, extra_xyz[part_2]),axis=0)
            rgb = np.concatenate((rgb, extra_rgb[part_2]),axis=0)
            seg = np.concatenate((seg, extra_seg[part_2]),axis=0)
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

def chair_process_mani_skill_base(obs, env=None):
    # obs: iterated dict,
    # e.g. {'agent': [], 'additional_task_info': [], 'rgbd': {'rgb': [], 'depth': [], 'seg': []}}
    # e.g. {'agent': [], 'additional_task_info': [], 'pointcloud': {'rgb': [], 'xyz': [], 'seg': []}}
    random.seed(1024)
    np.random.seed(1024)
    if not isinstance(obs, dict):
        # e.g. [N,d] numpy array, for state observation
        return obs
    obs_mode = env.obs_mode
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
        mask = (xyz[:, 2] <= 0.15) & (rgb[:, 0] > 0.6) & (rgb[:, 1] < 0.05) & (rgb[:, 2] < 0.05)
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