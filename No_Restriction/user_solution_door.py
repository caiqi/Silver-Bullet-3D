import pathlib
from collections import deque

import gym
import numpy as np
import copy

# from mani_skill_learn.env.observation_process import process_mani_skill_base
DEBUG = False


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
        observation = process_mani_skill_base(observation, self.env)
        visual_data = observation[self.obs_mode]
        self._update_buffer(visual_data)
        visual_data = self._get_buffer_content()
        state = observation['agent']
        # Convert dict of array to list of array with sorted key
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


def get_action(move_x=0., move_y=0., move_z=0., rot=0., finger=1.0):
    assert -1. <= move_x <= 1.
    assert -1. <= move_y <= 1.
    assert -1. <= move_z <= 1.
    assert -1. <= rot <= 1.
    assert -1. <= finger <= 1.
    cmd = [move_x, move_y, rot, move_z] + [0. for _ in range(7)] + [finger, finger]
    return cmd


def draw_pt(obs):
    xyz = obs['pointcloud']['xyz']
    seg = obs['pointcloud']['seg']
    num_obj = seg.shape[-1]
    color = [[1, 0.706, 0], [0, 0.651, 0.929], [1, 0, 0], [0.7, 0.7, 0.7]]
    point_array = []
    for i in range(num_obj):
        xyz_obj = xyz[seg[:, i], :]
        point_drawer = open3d.geometry.PointCloud()
        point_drawer.points = open3d.utility.Vector3dVector(xyz_obj)
        point_drawer.paint_uniform_color(color[i])
        point_array.append(point_drawer)
    open3d.visualization.draw_geometries(point_array, width=512, height=512)


def pdsit(pt, pts):
    assert len(pt.shape) == 1
    return np.sqrt(np.sum((pts - pt) ** 2, axis=-1))


def calc_handle_position(obs):
    xyz = obs['pointcloud']['xyz']
    seg = obs['pointcloud']['seg']
    xyz_handle = xyz[seg[:, 0], :]
    xyz_handle_center = np.mean(xyz_handle, axis=0)
#     dist = pdsit(xyz_handle_center, xyz_handle)
#     if np.mean(dist) > 0.1: ## two handle
#         std_x = np.std(xyz_handle[:, 0])
#         std_y = np.std(xyz_handle[:, 1])
#         if std_x > std_y: ## along x axis
#             v = xyz_handle[:, 0]
#         else:
#             v = xyz_handle[:, 1]
#         v_sorted_idx = np.argsort(v)
#         v_sorted_value = v[v_sorted_idx]
#         v_neighbor_diff = v_sorted_value[1:] - v_sorted_value[:-1]
#         v_slice_idx = np.argmax(v_neighbor_diff)
#         idx_handle1 = v_sorted_idx[:v_slice_idx + 1]
#         # idx_handle2 = v_sorted_idx[v_slice_idx+1:]
#         xyz_handle = xyz_handle[idx_handle1, :]
#         xyz_handle_center = np.mean(xyz_handle, axis=0)
    return xyz_handle_center


def calc_orientatation(obs):
    robot_angle = obs['agent'][14]
    return robot_angle


def calc_dist_axis_z(obs):
    xyz_handle_center = calc_handle_position(obs)
    xyz_finger = obs['agent'][0:6]
    h_handle = xyz_handle_center[-1]
    h_finger = (xyz_finger[2] + xyz_finger[5]) / 2
    diff = h_handle - h_finger
    return diff


def calc_dist_axis_y(obs):
    xyz_handle_center = calc_handle_position(obs)
    xyz_finger = obs['agent'][0:6]
    y_handle = xyz_handle_center[1]
    y_finger = (xyz_finger[1] + xyz_finger[1]) / 2
    diff = y_handle - y_finger
    return diff


def calc_dist_axis_x(obs):
    xyz_handle_center = calc_handle_position(obs)
    xyz_finger = obs['agent'][0:6]
    x_handle = xyz_handle_center[0]
    x_finger = (xyz_finger[0] + xyz_finger[0]) / 2
    diff = x_handle - x_finger
    return diff


def check_valid_handle_mask(obs):
    seg = obs['pointcloud']['seg'][:, 0]
    return np.any(seg)


# door function
def calc_handle_direction_init(obs):
    range_x = obs['handle_range'][0]
    range_y = obs['handle_range'][1]
    range_z = obs['handle_range'][2]

    if range_z < range_y:
        d1 = -1 # -
        width = range_z
    else:
        d1 = 1  # |
        width = range_y
    return d1, width


def calc_handle_offset(obs):
    # xyz_finger = obs['agent'][0:6]
    # y_finger = (xyz_finger[1] + xyz_finger[4]) / 2
    xyz = obs['pointcloud']['xyz']
    seg = obs['pointcloud']['seg']
    xyz_door = xyz[seg[:, 1], :]
    xyz_door_center = np.mean(xyz_door, axis=0)
    return xyz_door_center[1]


def calc_door_direction(obs):
    if obs['door_p1'][1] > obs['door_p2'][1]:
        return 1
    else:
        return -1
    # xyz = obs['pointcloud']['xyz']
    # seg = obs['pointcloud']['seg']
    # door_seg = seg[:, 1] & (~seg[:, 0])
    # xyz_door = xyz[door_seg, :]
    # xyz_door_center = np.mean(xyz_door, axis=0)

    # lapse = 0
    # num_point = xyz_door.shape[0]
    # for i in range(num_point):
    #     lapse += np.sign((xyz_door[i, 1] - xyz_door_center[1]) * (xyz_door[i, 0] - xyz_door_center[0])) / num_point

    # if lapse > 0:
    #     return 1
    # else:
    #     return -1


# def calc_door_dist(obs):
#     xyz = obs['pointcloud']['xyz']
#     seg = obs['pointcloud']['seg']
#     door_seg = seg[:, 1] & (~seg[:, 0])
#     xyz_door = xyz[door_seg, :]
#     xyz_door_center = np.mean(xyz_door, axis=0) + 0.001
#     # xyz_door_center = xyz_door.min(axis=0)

#     xyz_finger = obs['agent'][0:6]
#     x_door = xyz_door_center[0]
#     x_finger = (xyz_finger[0] + xyz_finger[0]) / 2
#     diff = x_door - x_finger
#     return diff


def calc_door_dist(obs):
    door_p1 = obs['door_p1']
    door_p2 = obs['door_p2']

    xyz_finger = obs['agent'][0:6]
    x_door = (door_p2[0] + door_p1[0]) / 2
    x_finger = (xyz_finger[0] + xyz_finger[0]) / 2
    diff = x_door - x_finger
    return diff


def get_change_finger_action(finger):
    cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, finger, finger]
    return cmd


def get_change_finger_action_inverse(finger):
    cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, finger, finger]
    return cmd


def calc_dist_3d(obs, x, y, z):
    xyz_handle_center = calc_handle_position(obs)
    xyz_finger = obs['agent'][0:6]
    h_handle = xyz_handle_center[-1]
    x_finger = (xyz_finger[0] + xyz_finger[3]) / 2
    y_finger = (xyz_finger[1] + xyz_finger[4]) / 2
    z_finger = (xyz_finger[2] + xyz_finger[5]) / 2

    return x - x_finger, y - y_finger, z - z_finger


def calc_door_position(obs):
    xyz = obs['pointcloud']['xyz']
    seg = obs['pointcloud']['seg']
    door_seg = seg[:, 1] & (~seg[:, 0])
    xyz_door = xyz[door_seg, :]
    xyz_door_center = np.mean(xyz_door, axis=0)
    return xyz_door_center


def get_positions(obs):
    xyz = obs['pointcloud']['xyz']
    seg = obs['pointcloud']['seg']
    door_seg = seg[:, 1] & (~seg[:, 0])
    xyz_finger = obs['agent'][0:6]
    x_finger = (xyz_finger[0] + xyz_finger[3]) / 2
    y_finger = (xyz_finger[1] + xyz_finger[4]) / 2
    z_finger = (xyz_finger[2] + xyz_finger[5]) / 2
    xyz_finger_center = np.array([x_finger, y_finger, z_finger])
    xyz_handle = xyz[seg[:, 0], :]
    xyz_handle_center = np.mean(xyz_handle, axis=0)
    xyz_door = xyz[door_seg, :]
    xyz_door_center = np.mean(xyz_door, axis=0)
    return xyz_finger_center, xyz_handle_center, xyz_door_center



def move_to_3d(obs, x, y, z, finger):
    thresh = 0.01
    move = 1.0

    dx, dy, dz = calc_dist_3d(obs, x, y, z)
    dx_p, dy_p, dz_p = dx, dy, dz

    if abs(dy) > thresh * 4:
        vy = move * np.sign(dy)
    elif abs(dy) > thresh:
        vy = move / 2 * np.sign(dy)
    else:
        vy = 0

    if abs(dz) > thresh * 4:
        vz = move * np.sign(dz)
    elif abs(dz) > thresh:
        vz = move / 2 * np.sign(dz)
    else:
        vz = 0

    if abs(dx) < thresh * 10 and (vy != 0 or vz != 0):
        vx = 0
    else:
        if abs(dx) > thresh * 4:
            vx = move * np.sign(dx)
        elif abs(dx) > thresh:
            vx = move / 2 * np.sign(dx)
        else:
            vx = 0

    action = get_action(move_x=vx, move_y=vy, move_z=vz, finger=finger)
    return action


def calc_door_rate(obs):
    xyz = obs['pointcloud']['xyz']
    seg = obs['pointcloud']['seg']
    door_seg = seg[:, 1] & (~seg[:, 0])
    xyz_door = xyz[door_seg, :]
    xyz_door_center = np.mean(xyz_door, axis=0)

    dy = 0
    dx = 0
    num_point = xyz_door.shape[0]
    for i in range(num_point):
        dx += abs(xyz_door[i, 0] - xyz_door_center[0]) / num_point
        dy += abs(xyz_door[i, 1] - xyz_door_center[1]) / num_point

    return dx / dy


def update_obs(old_obs, new_obs):
    if not np.any(new_obs['pointcloud']['seg'][:, 0]):
        if 'pointcloud' in old_obs: # first frame
            new_obs['pointcloud'] = old_obs['pointcloud']
    if not np.any(new_obs['pointcloud']['seg'][:, 0]):
        return new_obs
    
    xyz = new_obs['pointcloud']['xyz']
    seg = new_obs['pointcloud']['seg']
    xyz_finger = new_obs['agent'][0:6]
 
    door_seg = seg[:, 1] & (~seg[:, 0])
    if not door_seg.any():
        new_obs['door_p1'] = old_obs['door_p1']
        new_obs['door_p2'] = old_obs['door_p2']
        new_obs['door_rate'] = old_obs['door_rate']
        new_obs['handle_range'] = old_obs['handle_range']
    else:
        
        xyz_handle = xyz[seg[:, 0], :]
        xyz_door = xyz[door_seg, :]

        door_rate = calc_door_rate(new_obs)

        handle_range_x = np.max(xyz_handle[:, 0]) - np.min(xyz_handle[:, 0])
        handle_range_y = np.max(xyz_handle[:, 1]) - np.min(xyz_handle[:, 1])
        handle_range_z = np.max(xyz_handle[:, 2]) - np.min(xyz_handle[:, 2])
        if not 'door_p1' in old_obs:
            new_obs['handle_range'] = [handle_range_x, handle_range_y, handle_range_z]
        else:
            new_obs['handle_range'] = [old_obs['handle_range'][0] * 0.9 + handle_range_x * 0.1,
                                       old_obs['handle_range'][1] * 0.9 + handle_range_y * 0.1,
                                       old_obs['handle_range'][2] * 0.9 + handle_range_z * 0.1]

        if door_rate > 0.04:
            door_max_x_idx = xyz_door[:, 0].argmax()
            door_min_x_idx = xyz_door[:, 0].argmin()
        else:
            xyz_handle_center = np.mean(xyz_handle, axis=0)
            xyz_door_center = np.mean(xyz_door, axis=0)

            if xyz_handle_center[1] > xyz_door_center[1] + 0.1:
                door_max_x_idx = xyz_door[:, 1].argmin()
                door_min_x_idx = xyz_door[:, 1].argmax()
            else:
                door_max_x_idx = xyz_door[:, 1].argmax()
                door_min_x_idx = xyz_door[:, 1].argmin()

        door_color = [40] * xyz_door.shape[0]
        door_color[door_max_x_idx] = 80
        door_color[door_min_x_idx] = 100
        
        door_p1 = [xyz_door[door_max_x_idx, 0], xyz_door[door_max_x_idx, 1]]
        door_p2 = [xyz_door[door_min_x_idx, 0], xyz_door[door_min_x_idx, 1]]

        alpha = 0.5
        if not 'door_p1' in old_obs:
            new_obs['door_p1'] = door_p1
            new_obs['door_p2'] = door_p2
        else:
            new_obs['door_p1'] = [old_obs['door_p1'][0] * alpha + door_p1[0] * (1 - alpha), old_obs['door_p1'][1] * alpha + door_p1[1] * (1 - alpha)]
            new_obs['door_p2'] = [old_obs['door_p2'][0] * alpha + door_p2[0] * (1 - alpha), old_obs['door_p2'][1] * alpha + door_p2[1] * (1 - alpha)]

        # num_point = xyz_door.shape[0]
        # theta = 0
        # for i in range(num_point):
        #     theta = np.arctan((xyz_door[i, 0] - new_obs['door_p1'][0]) / (xyz_door[i, 1] - new_obs['door_p1'][1]))
        # new_obs['door_theta'] = theta
        # print(new_obs['door_theta'])

        # door_rate = (new_obs['door_p2'][0] - new_obs['door_p1'][0]) / (new_obs['door_p2'][1] - new_obs['door_p1'][1])       
        new_obs['door_rate'] = door_rate
        # print(str(new_obs['door_p1']))
        # print(str(new_obs['door_p2']))
        # print(new_obs['door_rate'])
    return new_obs


def rotate_keep_finger(rot, flag_low, alpha1=None, alpha2=None):
    if flag_low:
        vx = -rot * 0.15
        vy = -rot
        scaled_rot = rot * 0.14
    else:
        vx = -rot * 0.85
        vy = -rot
        scaled_rot = rot * 0.22

    if abs(vx) > 1:
        vx = np.sign(vx)
    if abs(vy) > 1:
        vy = np.sign(vy)
    
    return get_action(rot=scaled_rot, move_x=vx, move_y=vy, finger=-1)


def agent_function():
    obs = update_obs({}, (yield [0] * 13))
    while not np.any(obs['pointcloud']['seg'][:, 0]):
        obs = update_obs(obs, (yield [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    finger_open_pos = 0.1
    finger_close_pos = -1.0

    # finger_keep = 0.0001

    rot_thresh = 0.002
    z_thresh = 0.01
    y_thresh = 0.01
    x_thresh = 0.01

    rot_v = 0.3
    move_z = 1.0
    move_y = 1.0
    move_x = 1.0

    action_queue = []

    total_step = 0

    xyz_handle_center = calc_handle_position(obs)
    flag_low = False
    if xyz_handle_center[2] < 0.5:
        flag_low = True
        action_none = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        actions_seq = [
            ([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], 5),
             ([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 10),
            ([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 10),
            ([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 5),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 10),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 0)
        ]
        # for i_step in range(10):
            # obs = update_obs(obs, (yield [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

        max_step = max([x[1] for x in actions_seq])
        for i_step in range(max_step):
            action_init = copy.deepcopy(action_none)
            for act, step in actions_seq:
                if step - i_step > 0:
                    action = [a + b for a, b in zip(action_init, act)]
                    action_init = copy.deepcopy(action)
            if i_step < 10:
                action[3] = 1

            dist = calc_orientatation(obs)
            if abs(dist) > rot_thresh * 4:
                v_curr = rot_v
            elif abs(dist) > rot_thresh:
                v_curr = rot_v / 2
            else:
                v_curr = 0
            action[2] = -v_curr if dist > 0 else v_curr
            obs = update_obs(obs, (yield action))

    # angle
    dist = calc_orientatation(obs)
    dist_prev = dist
    while abs(dist) > rot_thresh:
        if abs(dist) > rot_thresh * 64:
            v_curr = 1.0
        elif abs(dist) > rot_thresh * 4:
            v_curr = rot_v
        else:
            v_curr = rot_v / 2
        action = get_action(rot=-v_curr if dist > 0 else v_curr, finger=finger_open_pos)
        obs = update_obs(obs, (yield action))
        dist = calc_orientatation(obs)
        if dist * dist_prev < 0:
            break
        else:
            dist_prev = dist

    # change finger
    d1, handle_width = calc_handle_direction_init(obs)
    if d1 == 1:
        for _ in range(10):
            action_queue.append((0, 0, -1.0))

    if handle_width > 0.008:
        finger_open_pos_handle = finger_open_pos
    else:
        finger_open_pos_handle = 0.0003
        

    # go to handle
    xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
    x = xyz_handle_center[0]
    y = xyz_handle_center[1]
    z = xyz_handle_center[2]
    dx, dy, dz = calc_dist_3d(obs, x, y, z)
    dx_p, dy_p, dz_p = dx, dy, dz
    step = 0
    while abs(dx) > x_thresh or abs(dy) > y_thresh or abs(dz) > z_thresh:
        action = move_to_3d(obs, x, y, z, finger_open_pos_handle)
        if action_queue:
            a1, a2, a3 = action_queue.pop(0)
            action[-5] = a1
            action[-4] = a2
            action[-3] = a3
        obs = update_obs(obs, (yield action))
        step += 1
        dx, dy, dz = calc_dist_3d(obs, x, y, z)
        if (step > 5 and abs(dx_p - dx) < x_thresh * 0.2 and abs(dy_p - dy) < y_thresh * 0.2 and abs(dz_p - dz) < z_thresh * 0.2):
            break
        else:
            dx_p, dy_p, dz_p = dx, dy, dz

    xyz_handle_center = calc_handle_position(obs)
    if xyz_handle_center[2] > 0.05:
        action = get_action(move_x=-move_x*0.1, finger=finger_open_pos)
        obs = update_obs(obs, (yield action))
        # try forward
        for i in range(3):
            xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
            old_x = xyz_finger_center[0]
            
            action = get_action(move_x=move_x*0.5, finger=finger_close_pos)
            obs = update_obs(obs, (yield action))
            action = get_action(move_x=move_x*0.5, finger=finger_open_pos)
            obs = update_obs(obs, (yield action))

            xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
            new_x = xyz_finger_center[0]
            if abs(new_x - old_x) < x_thresh * 0.2:
                break

        for _ in range(0):
            # action = get_change_finger_action_inverse(finger_close_pos)
            action = get_action(move_x=move_x*0.1, finger=finger_close_pos)
            action[-3] = -1.0
            obs = update_obs(obs, (yield action))

        # grasp
        # action = get_action(finger=finger_close_pos)
        # obs = update_obs(obs, (yield action))

        # pull1
        for i in range(1, 5):
            for _ in range(3):
                action = get_action(move_x=-move_x * 0.1 * i, finger=finger_close_pos)
                obs = update_obs(obs, (yield action))

        # pull2
        for _ in range(20):
            xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
            if abs(xyz_finger_center[0] - xyz_handle_center[0]) > x_thresh * 2 or (obs['door_rate'] > np.tanh(np.pi / 180 * 30)):
                break
            px = xyz_finger_center[0]

            action = get_action(move_x=-move_x, finger=finger_close_pos)
            obs = update_obs(obs, (yield action))

            xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
            nx = xyz_finger_center[0]
            if nx > px:
                break


        for _ in range(0):
            action = get_change_finger_action(finger_open_pos)
            obs = update_obs(obs, (yield action))

        # change finger
        for i in range(8):
            if d1 == -1:
                xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
                action = get_action(move_z=1.0 if xyz_finger_center[2] < xyz_handle_center[2] else -1.0, move_x=-1.0, finger=finger_open_pos)
            else:
                xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
                action = get_action(move_y=1.0 if xyz_finger_center[1] < xyz_handle_center[1] else -1.0, move_x=-1.0, finger=finger_open_pos)
            # action[-3] = 1.0 if i % 2 == 0 else -1.0
            # action[2] = 0.6 if i % 2 == 0 else -0.6
            obs = update_obs(obs, (yield action))
    
    d2 = calc_door_direction(obs)

    for _ in range(1):
        action = get_action(move_y=d2, finger=finger_open_pos)
        obs = update_obs(obs, (yield action))

    go_back_dist = 0.10
    go_left_dist = 0.06
    
    # go back
    xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
    x = obs['door_p2'][0] - go_back_dist
    y = xyz_finger_center[1]
    z = xyz_finger_center[2]
    dx, dy, dz = calc_dist_3d(obs, x, y, z)
    dx_p, dy_p, dz_p = dx, dy, dz
    step = 0
    while abs(dx) > x_thresh or abs(dy) > y_thresh or abs(dz) > z_thresh:
        action = move_to_3d(obs, x, y, z, finger_open_pos)
        obs = update_obs(obs, (yield action))
        step += 1
        dx, dy, dz = calc_dist_3d(obs, x, y, z)
        if (step > 5 and abs(dx_p - dx) < x_thresh * 0.2 and abs(dy_p - dy) < y_thresh * 0.2 and abs(dz_p - dz) < z_thresh * 0.2):
            break
        else:
            dx_p, dy_p, dz_p = dx, dy, dz

    # change finger
    if (d1 == 1 and not flag_low):
        for _ in range(5):
            action_queue.append((0, -d2, 1.0))
        for _ in range(5):
            action_queue.append((0, 0, 1.0))
    elif (d1 == -1 and flag_low):
        for _ in range(5):
            action_queue.append((d2, 1.0, -1.0))
        for _ in range(5):
            action_queue.append((d2, 0.0, -1.0))
    else:
        if not flag_low:
            for _ in range(5):
                action_queue.append((0, -d2, 0))
        else:
            for _ in range(5):
                action_queue.append((d2, 1.0, 0))
            for _ in range(5):
                action_queue.append((d2, 0.0, 0))

    # angle
    dist = calc_orientatation(obs)
    dist_prev = dist
    while abs(dist) > rot_thresh:
        if abs(dist) > rot_thresh * 4:
            v_curr = rot_v
        else:
            v_curr = rot_v / 2
        action = get_action(rot=-v_curr if dist > 0 else v_curr, finger=finger_open_pos)
        if action_queue:
            a1, a2, a3 = action_queue.pop(0)
            action[-5] = a1
            action[-4] = a2
            action[-3] = a3
        obs = update_obs(obs, (yield action))
        dist = calc_orientatation(obs)
        if dist * dist_prev < 0:
            break
        else:
            dist_prev = dist

    # go left/right
    xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
    x = xyz_finger_center[0]
    y = obs['door_p2'][1] - d2 * go_left_dist
    z = xyz_door_center[2]
    dx, dy, dz = calc_dist_3d(obs, x, y, z)
    dx_p, dy_p, dz_p = dx, dy, dz
    step = 0
    while abs(dx) > x_thresh or abs(dy) > y_thresh or abs(dz) > z_thresh:
        action = move_to_3d(obs, x, y, z, finger_open_pos)
        if action_queue:
            a1, a2, a3 = action_queue.pop(0)
            action[-5] = a1
            action[-4] = a2
            action[-3] = a3
        obs = update_obs(obs, (yield action))
        step += 1
        dx, dy, dz = calc_dist_3d(obs, x, y, z)
        if (step > 5 and abs(dx_p - dx) < x_thresh * 0.2 and abs(dy_p - dy) < y_thresh * 0.2 and abs(dz_p - dz) < z_thresh * 0.2):
            break
        else:
            dx_p, dy_p, dz_p = dx, dy, dz

    # move to door
    dist = calc_door_dist(obs)
    dist_prev = dist
    step = 0
    while abs(dist) > x_thresh:
        if abs(dist) > x_thresh * 4:
            v_curr = move_x
        else:
            v_curr = move_x / 2
        action = get_action(move_x=v_curr if dist > 0 else -v_curr, finger=finger_open_pos)
        if action_queue:
            a1, a2, a3 = action_queue.pop(0)
            action[-5] = a1
            action[-4] = a2
            action[-3] = a3
        obs = update_obs(obs, (yield action))
        step += 1
        dist = calc_door_dist(obs)
        if (step > 5 and abs(dist_prev - dist) < x_thresh * 0.5):
            break
        else:
            dist_prev = dist

    if obs['door_rate'] < np.tanh(np.pi * 15 / 180):
        if obs['door_rate'] < np.tanh(np.pi * 5 / 180):
            num = 50
        elif obs['door_rate'] < np.tanh(np.pi * 10 / 180):
            num = 30
        else:
            num = 20
        for i in range(num):
            # action = get_action(move_x=-move_x * i / num, move_y=d2, finger=finger_open_pos)
            if i < 10:
                action = get_action(move_x=-move_x * i / 10, move_y=d2, finger=finger_open_pos)
            else:            
                action = get_action(move_x=-move_x, move_y=d2, finger=finger_open_pos)
            if action_queue:
                a1, a2, a3 = action_queue.pop(0)
                action[-5] = a1
                action[-4] = a2
                action[-3] = a3
    
            obs = update_obs(obs, (yield action))

            xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
            if xyz_finger_center[0] < obs['door_p2'][0] - 0.1 or (d2 == -1 and xyz_finger_center[1] < min(obs['door_p1'][1], obs['door_p2'][1]) - 0.1) or (d2 == 1 and xyz_finger_center[1] > max(obs['door_p1'][1], obs['door_p2'][1]) + 0.1):
                break
        
        # go left/right
        xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
        x = xyz_finger_center[0]
        y = obs['door_p2'][1] - d2 * go_left_dist
        z = xyz_door_center[2]
        dx, dy, dz = calc_dist_3d(obs, x, y, z)
        dx_p, dy_p, dz_p = dx, dy, dz
        step = 0
        while abs(dx) > x_thresh or abs(dy) > y_thresh or abs(dz) > z_thresh:
            action = move_to_3d(obs, x, y, z, finger_open_pos)
            if action_queue:
                a1, a2, a3 = action_queue.pop(0)
                action[-5] = a1
                action[-4] = a2
                action[-3] = a3
            obs = update_obs(obs, (yield action))
            step += 1
            dx, dy, dz = calc_dist_3d(obs, x, y, z)
            if (step > 5 and abs(dx_p - dx) < x_thresh * 0.2 and abs(dy_p - dy) < y_thresh * 0.2 and abs(dz_p - dz) < z_thresh * 0.2):
                break
            else:
                dx_p, dy_p, dz_p = dx, dy, dz

        # move to door
        dist = calc_door_dist(obs)
        dist_prev = dist
        step = 0
        while abs(dist) > x_thresh:
            if abs(dist) > x_thresh * 4:
                v_curr = move_x
            else:
                v_curr = move_x / 2
            action = get_action(move_x=v_curr if dist > 0 else -v_curr, finger=finger_open_pos)
            if action_queue:
                a1, a2, a3 = action_queue.pop(0)
                action[-5] = a1
                action[-4] = a2
                action[-3] = a3
            obs = update_obs(obs, (yield action))
            step += 1
            dist = calc_door_dist(obs)
            if (step > 5 and abs(dist_prev - dist) < x_thresh * 0.5):
                break
            else:
                dist_prev = dist
    
    for i in range(120):
        move_forward = False
        if i < 5:
            action = get_action(move_x=0, move_y=d2, finger=finger_open_pos)
        elif i < 10:
            if i % 3 == 0:
                action = get_action(move_x=-move_x, move_y=d2, finger=finger_open_pos)
            else:
                action = get_action(move_x=move_x, move_y=d2, finger=finger_open_pos)            
        else:
            xyz_finger_center, xyz_handle_center, xyz_door_center = get_positions(obs)
            if xyz_finger_center[0] > (obs['door_p1'][0] + obs['door_p2'][0] * 3) / 4:
                v_x = -move_x
            else:
                v_x = 0 # move_x

            if xyz_finger_center[2] > xyz_door_center[2]:
                v_z = -move_z
            else:
                v_z = move_z

            action = get_action(move_x=v_x, move_y=d2, move_z=v_z, finger=finger_open_pos)
        
            if i > 30:
                if xyz_finger_center[0] < obs['door_p2'][0] - 0.1 or (d2 == -1 and xyz_finger_center[1] < min(obs['door_p1'][1], obs['door_p2'][1]) - 0.1) or (d2 == 1 and xyz_finger_center[1] > max(obs['door_p1'][1], obs['door_p2'][1]) + 0.1):
                    move_forward = True
                    action = get_action(move_x=move_x, finger=finger_open_pos)

        # action = get_action(move_x=move_x * 0.1, move_y=d2, finger=finger_open_pos)
        if action_queue:
            a1, a2, a3 = action_queue.pop(0)
            action[-5] = a1
            action[-4] = a2
            action[-3] = a3
        if flag_low:
#             if i > 30 and not move_forward:
#                 # 4x, 5x, 7x
#                 action[2] = d2 * 0.5
            if i > 10 and not move_forward:
                # 4x, 5x, 7x
                action[2] = d2 * min(1.0, i * 0.5 / 50)
        else:
            if i > 10 and d2 == -1 and not move_forward:
                action[9] = 1.0
                action[5] = -0.5# -0.5
            if i > 30 and d2 == -1 and not move_forward:
                action[9] = 1.0
                action[5] = -1.0# -0.5

            if i > 30 and d2 == 1 and not move_forward:
                action[7] = -1.0
                action[5] = 1.0
        obs = update_obs(obs, (yield action))





    for _ in range(30000):
        dist = calc_orientatation(obs)
        dist = calc_dist_axis_x(obs)
        dist = calc_dist_axis_y(obs)
        dist = calc_dist_axis_z(obs)
        action = get_action()
        obs = update_obs(obs, (yield action))


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.env = gym.make(env_name)
        self.obs_mode = 'pointcloud'  # remember to set this!
        self.env.set_env_mode(obs_mode=self.obs_mode)
        self.stack_frame = 1
        self.obsprocess = ObsProcess(self.env, self.obs_mode, self.stack_frame)
        self.agent = agent_function()
        next(self.agent)

    def reset(self):
        self.agent = agent_function()
        next(self.agent)

    def act(self, observation):
        ##### Replace with your code
        try:
            action = self.agent.send(observation)
        except StopIteration:
            action = [0] * 13
        return action

