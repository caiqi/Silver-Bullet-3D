import cv2
import gym
import mani_skill.env
import numpy as np
import copy

DEBUG = False


def pdist(pt, pts):
    return np.sqrt(np.sum((pts - pt) ** 2, axis=1))


def calc_l2_diff(src, tar):
    l2_dis = np.linalg.norm(src - tar)
    return l2_dis


def get_action(move_x=0., move_y=0., move_z=0., rot=0., finger=1.0, arml=None, armr=None, arm=None):
    if arm is None:
        arm_l = arml if arml is not None else [0. for _ in range(7)]
        arm_r = armr if armr is not None else [0. for _ in range(7)]
    else:
        arm_l = copy.deepcopy(arm)
        arm_r = copy.deepcopy(arm)
        arm_r[2] *= -1
        arm_r[-3] *= -1
        arm_r[-1] *= -1
    cmd = [move_x, move_y, rot, move_z] + arm_r + [finger, finger] + arm_l + [finger, finger]
    return cmd


# -------  Tools for coordinate computation ------- #
def search_bin(height_range, bin_array, top_search_num):
    ratio = 10000
    bin_bank = []
    height_range = [int(height_range[0] * ratio), int(height_range[1] * ratio)]
    for j in range(len(bin_array)):
        if bin_array[j] in range(height_range[0], height_range[1]):
            # return bin_array[j]
            if len(bin_bank) == 0:
                bin_bank.append(bin_array[j])
            else:
                if len(bin_bank) >= top_search_num: break
                add_flag = True
                for b in bin_bank:
                    if abs(b - bin_array[j]) < 50:
                        add_flag = False
                        break
                if add_flag: bin_bank.append(bin_array[j])
    if len(bin_bank) > 0:
        return min(bin_bank)
    else:
        return bin_array[0]


def calc_pre_ground_center(pcd_xyz):
    area_point = pcd_xyz
    min_x = np.min(area_point[:, 0])
    min_y = np.min(area_point[:, 1])
    min_xy = np.array([min_x, min_y])
    area_point = area_point - min_xy
    rect = cv2.minAreaRect(area_point)
    box = cv2.boxPoints(rect)
    return ((box[1] + box[3]) / 2.0) + min_xy, box + min_xy


def calc_box_back_direct(pre_box, chair_back_direct):
    zero_point = np.array([0.0, 0.0])

    direct1 = pre_box[0] - pre_box[1]
    direct1 = direct1 / np.sqrt(direct1.dot(direct1))
    mxy1 = (pre_box[0] + pre_box[1]) / 2.0
    mxy1c = (pre_box[2] + pre_box[3]) / 2.0
    direct2 = pre_box[1] - pre_box[2]
    direct2 = direct2 / np.sqrt(direct2.dot(direct2))
    mxy2 = (pre_box[1] + pre_box[2]) / 2.0
    mxy2c = (pre_box[0] + pre_box[3]) / 2.0

    if calc_l2_diff(mxy1c, zero_point) > calc_l2_diff(mxy1, zero_point): mxy1 = mxy1c
    if calc_l2_diff(mxy2c, zero_point) > calc_l2_diff(mxy2, zero_point): mxy2 = mxy2c
    max_dot_d1 = np.max(np.array(abs(direct1.dot(chair_back_direct)), abs(direct1.dot(-1.0 * chair_back_direct))))
    max_dot_d2 = np.max(np.array(abs(direct2.dot(chair_back_direct)), abs(direct2.dot(-1.0 * chair_back_direct))))
    if max_dot_d1 > max_dot_d2:
        if direct1.dot(chair_back_direct) > 0:
            return direct1, mxy1
        else:
            return -direct1, mxy1
    else:
        if direct2.dot(chair_back_direct) > 0:
            return direct2, mxy2
        else:
            return -direct2, mxy2


def calc_chair_back_direct_box(chair_back_xy, finger_direct):
    chair_back_center, chair_back_box = calc_pre_ground_center(chair_back_xy)
    zero_point = np.array([0.0, 0.0])

    direct1 = chair_back_box[0] - chair_back_box[1]
    mod1 = np.sqrt(direct1.dot(direct1))
    direct2 = chair_back_box[1] - chair_back_box[2]
    mod2 = np.sqrt(direct2.dot(direct2))
    mxy1 = (chair_back_box[0] + chair_back_box[1]) / 2.0
    mxy1c = (chair_back_box[2] + chair_back_box[3]) / 2.0
    mxy2 = (chair_back_box[1] + chair_back_box[2]) / 2.0
    mxy2c = (chair_back_box[0] + chair_back_box[3]) / 2.0

    if calc_l2_diff(mxy1c, zero_point) > calc_l2_diff(mxy1, zero_point): mxy1 = mxy1c
    if calc_l2_diff(mxy2c, zero_point) > calc_l2_diff(mxy2, zero_point): mxy2 = mxy2c
    if mod1 > mod2:
        if direct1.dot(finger_direct) > 0:
            return direct1 / mod1, mxy1
        else:
            return -direct1 / mod1, mxy1
    else:
        if direct2.dot(finger_direct) > 0:
            return direct2 / mod2, mxy2
        else:
            return -direct2 / mod2, mxy2


def calc_cos_value(src, tar):
    src_xy, tar_xy = src[:2], tar[:2]
    src_l, tar_l = np.sqrt(src_xy.dot(src_xy)), np.sqrt(tar_xy.dot(tar_xy))
    dot_value = src_xy.dot(tar_xy)
    cos_value = dot_value / (src_l * tar_l)
    return cos_value


def calc_angle(src, tar):
    src_xy, tar_xy = src[:2], tar[:2]
    src_l, tar_l = np.sqrt(src_xy.dot(src_xy)), np.sqrt(tar_xy.dot(tar_xy))
    dot_value = src_xy.dot(tar_xy)
    cos_value = dot_value / (src_l * tar_l)
    angle = np.arccos(cos_value)
    return angle


def calc_chair_mean_pos(xyz, seg):
    z = xyz[:, 2] + seg[:, 0] * 1000
    chair_mask = (z > 0.15) & (z < 20)
    return xyz[chair_mask], xyz[chair_mask].mean(axis=0)


def calc_finger_mean_pos(obs):
    left_mxy = (obs['agent'][0:3] + obs['agent'][3:6]) / 2.0
    right_mxy = (obs['agent'][6:9] + obs['agent'][9:12]) / 2.0
    return left_mxy, right_mxy


def calc_chair_back_pos(xyz, seg):
    z = xyz[:, 2] + seg[:, 0] * 1000
    chair_mask = (z > 0.15) & (z < 50)
    chair_point = xyz[chair_mask]
    norm_z = (chair_point[:, 2] * 10000).astype(np.int)
    point_bc = np.bincount(norm_z)
    more_array = np.argsort(point_bc)[::-1]
    more = search_bin([0.42, 0.80], more_array, 1)
    ground_z = abs(norm_z - more) < 50
    chair_ground = chair_point[ground_z]
    max_height = len(point_bc)
    chair_back_z = (norm_z > more) & (abs(norm_z - max_height) < 1000)
    chair_back = chair_point[chair_back_z]
    if len(chair_back) == 0:
        chair_back_z = abs(norm_z - more) < 10
        chair_back = chair_point[chair_back_z]
    return chair_back, chair_ground


def get_point_to_origin_normal(direct, p):
    normal1 = np.array([(-1.0) * direct[1], direct[0]])
    normal2 = np.array([direct[1], (-1.0) * direct[0]])
    if calc_cos_value(normal1, -1.0 * p) > calc_cos_value(normal2, -1.0 * p):
        return normal1
    else:
        return normal2


# ----------------------------------------------- #


class BasePolicy(object):
    def __init__(self, opts=None):
        self.obs_mode = 'pointcloud'

    def act(self, observation):
        raise NotImplementedError()

    def reset(self):  # if you use an RNN-based policy, you need to implement this function
        pass


class BaseAction(object):
    def __init__(self, step_len=100000):
        assert step_len > 0
        self.step_len = step_len
        self.step_cnt = 0

    @property
    def isDone(self):
        return not self.step_cnt < self.step_len

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        pass

    def act(self, obs):
        raise NotImplementedError


class ActFixedMove(BaseAction):

    def __init__(self, step_len, move_x=0., move_y=0., move_z=0., rot=0., finger=1.0, arm=None, holding=None):
        super(ActFixedMove, self).__init__(step_len)
        self.vx = move_x
        self.vy = move_y
        self.vz = move_z
        self.rot = rot
        self.finger = finger
        self.arm = arm
        self.holding = holding

    def act(self, obs):
        if not self.isDone:
            self.step_cnt += 1
            if self.holding is not None:
                armR, armL = self.holding.act_impl(obs)
                action = get_action(move_x=self.vx, move_y=self.vy, move_z=self.vz, rot=self.rot, finger=self.finger,
                                    armr=armR, arml=armL)
            else:
                action = get_action(move_x=self.vx, move_y=self.vy, move_z=self.vz, rot=self.rot, finger=self.finger,
                                    arm=self.arm)
            return action
        else:
            raise RuntimeError

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        pass


class TurnAndMove(BaseAction):

    def __init__(self, step_len, move_x=0., move_y=0., move_z=0., rot=0., finger=1.0, arm=None, holding=None):
        super(TurnAndMove, self).__init__(step_len)
        self.vx = move_x
        self.vy = move_y
        self.vz = move_z
        self.rot = rot
        self.finger = finger
        self.arm = arm
        self.holding = holding

        self.cos_value = None
        self.height_range = [0.42, 0.80]
        self.top_ground_search_num = 1
        self.box_guide = True
        self.his_cos_sign = 0.3
        self.his_cos = 10
        self.a_thred = 0.01

    def act(self, obs):
        if not self.isDone:
            self.step_cnt += 1
            action = self.adjust_angle()
            return action
        else:
            raise RuntimeError

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.obs = obs
        self.xyz = self.obs['pointcloud']['xyz']
        self.seg = self.obs['pointcloud']['seg']
        self.left_finger_mxyz, self.right_finger_mxyz = calc_finger_mean_pos(self.obs)
        self.finger_myx = (self.left_finger_mxyz + self.right_finger_mxyz) / 2.0
        self.finger_direct = self.left_finger_mxyz[:2] - self.right_finger_mxyz[:2]
        self.finger_direct = self.finger_direct / np.sqrt(self.finger_direct.dot(self.finger_direct))
        self.finger_normal = get_point_to_origin_normal(self.finger_direct, self.finger_myx)

        if self.step_cnt == 0:
            self.chair_xyz, self.chair_mxyz = calc_chair_mean_pos(self.xyz, self.seg)
            self.pre_ground_center, self.pre_box = calc_pre_ground_center(self.chair_xyz[:, :2])
            self.chair_back_xyz, self.pre_chair_ground_xyz = calc_chair_back_pos(self.xyz, self.seg)
            self.chair_back_direct, self.chair_back_mxy = calc_chair_back_direct_box(self.chair_back_xyz[:, :2],
                                                                                     self.finger_direct)
            self.chair_back_normal = get_point_to_origin_normal(self.chair_back_direct, self.chair_back_mxy)
            self.box_back_direct, self.box_back_mxy = calc_box_back_direct(self.pre_box, self.chair_back_direct)
            self.box_back_normal = get_point_to_origin_normal(self.box_back_direct, self.box_back_mxy)
            self.select_one_direct_mxy()

        if self.box_guide:
            self.cos_value = calc_angle(self.finger_direct, self.box_back_direct)
        else:
            self.cos_value = calc_angle(self.finger_direct, self.chair_back_direct)

    def select_one_direct_mxy(self):
        dot_value = abs(self.box_back_direct.dot(self.chair_back_direct))
        if dot_value < 0.95: self.box_guide = False
        return

    def adjust_angle(self):
        turn = np.array([0.0 for i in range(22)])
        if self.cos_value > self.his_cos:
            self.his_cos_sign *= -1.0
        turn[2] = self.his_cos_sign
        self.his_cos = self.cos_value
        if abs(self.cos_value) > self.a_thred * 2:
            return turn
        elif abs(self.cos_value) < self.a_thred * 2 and abs(self.cos_value) > self.a_thred:
            return turn / 2.0

    @property
    def isDone(self):
        if self.cos_value is None:
            return False
        else:
            return abs(self.cos_value) <= self.a_thred


class ActMoveX(BaseAction):

    def __init__(self, step_len, move_x=0., move_y=0., move_z=0., rot=0., finger=1.0, arm=None, holding=None):
        super(ActMoveX, self).__init__(step_len)
        self.vx = move_x
        self.vy = move_y
        self.vz = move_z
        self.rot = rot
        self.finger = finger
        self.arm = arm
        self.holding = holding

        self.sp_diff = None
        self.box_guide = True
        self.his_x_move = 100
        self.his_x_move_sign = 1.0
        self.sp_x_thred = 0.01

    def act(self, obs):
        if not self.isDone:
            self.step_cnt += 1
            action = self.adjust_x()
            return action
        else:
            raise RuntimeError

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.obs = obs
        self.xyz = self.obs['pointcloud']['xyz']
        self.seg = self.obs['pointcloud']['seg']
        self.base_pos = self.obs['agent'][24:26]
        self.left_finger_mxyz, self.right_finger_mxyz = calc_finger_mean_pos(self.obs)
        self.finger_myx = (self.left_finger_mxyz + self.right_finger_mxyz) / 2.0
        self.finger_direct = self.left_finger_mxyz[:2] - self.right_finger_mxyz[:2]
        self.finger_direct = self.finger_direct / np.sqrt(self.finger_direct.dot(self.finger_direct))
        self.finger_normal = get_point_to_origin_normal(self.finger_direct, self.finger_myx)

        if self.step_cnt == 0:
            self.chair_xyz, self.chair_mxyz = calc_chair_mean_pos(self.xyz, self.seg)
            self.pre_ground_center, self.pre_box = calc_pre_ground_center(self.chair_xyz[:, :2])
            self.chair_back_xyz, self.pre_chair_ground_xyz = calc_chair_back_pos(self.xyz, self.seg)
            self.chair_back_direct, self.chair_back_mxy = calc_chair_back_direct_box(self.chair_back_xyz[:, :2],
                                                                                     self.finger_direct)
            self.chair_back_normal = get_point_to_origin_normal(self.chair_back_direct, self.chair_back_mxy)
            self.box_back_direct, self.box_back_mxy = calc_box_back_direct(self.pre_box, self.chair_back_direct)
            self.box_back_normal = get_point_to_origin_normal(self.box_back_direct, self.box_back_mxy)
            self.select_one_direct_mxy()

        if self.box_guide:
            self.sp_diff = calc_l2_diff(self.base_pos, self.box_back_mxy)
            edge_direct = self.box_back_mxy - self.base_pos
            angle = calc_angle(self.box_back_normal, edge_direct)
        else:
            self.sp_diff = calc_l2_diff(self.base_pos, self.chair_back_mxy)
            edge_direct = self.chair_back_mxy - self.base_pos
            angle = calc_angle(self.chair_back_normal, edge_direct)
        self.sp_diff = np.sin(angle) * self.sp_diff

    def select_one_direct_mxy(self):
        dot_value = abs(self.box_back_direct.dot(self.chair_back_direct))
        if dot_value < 0.95: self.box_guide = False
        return

    def adjust_x(self):
        sp = np.array([0.0 for i in range(22)])
        if self.sp_diff > self.his_x_move: self.his_x_move_sign *= -1.0
        sp[1] = self.his_x_move_sign
        self.his_x_move = self.sp_diff
        if self.sp_diff > self.sp_x_thred * 1.5:
            return sp
        elif self.sp_diff > self.sp_x_thred and self.sp_diff <= self.sp_x_thred * 1.5:
            return sp / 2.0

    @property
    def isDone(self):
        if self.sp_diff is None:
            return False
        else:
            return self.sp_diff <= self.sp_x_thred


class ActAdjustHeight(BaseAction):

    def __init__(self):
        super(ActAdjustHeight, self).__init__()
        self.v = 1.0
        self.dist_Z, self.dist_Z_prev = None, None
        self.thresh_z = 0.05
        self.thresh_re = 3
        self.dist_Z_his = 100
        self.diff_same = 0

    def act(self, obs):
        v = self.v if abs(self.dist_Z) > self.thresh_z * 2 else (self.v / 2)
        v = v if self.dist_Z < 0 else -v
        action = get_action(move_z=v)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.dist_Z_prev, self.dist_Z = self.dist_Z, self.calc_z_dist(obs)
        if self.step_cnt == 0:
            self.dist_Z_prev = self.dist_Z
        if abs(self.dist_Z_prev - self.dist_Z) < 0.001: self.diff_same += 1

    @property
    def isDone(self):
        if self.dist_Z is None:
            return False
        else:
            return abs(self.dist_Z) <= self.thresh_z or (
                        self.dist_Z * self.dist_Z_prev) < 0 or self.diff_same >= self.thresh_re

    def calc_z_dist(self, obs):
        # accu_pts
        xyz = obs['pointcloud']['xyz']
        seg = obs['pointcloud']['seg']
        # remove arm
        mask_arm = seg[:, 0]
        xyz = xyz[np.logical_not(mask_arm), :]
        # remove ground
        mask_floor = xyz[:, 2] < 0.005
        xyz = xyz[np.logical_not(mask_floor), :]
        # remove red point
        dist2target = pdist(np.zeros([3]), xyz)
        mask_red_point = dist2target < 0.16
        xyz = xyz[np.logical_not(mask_red_point), :]

        robot_xy = obs['agent'][24:26]
        chair_body_xyz = xyz[xyz[:, 2] > 0.2, :]
        dist = pdist(robot_xy, chair_body_xyz[:, :2])
        chair_far_xyz = chair_body_xyz[np.argsort(dist)[-100:], :]

        chair_z = np.min(chair_far_xyz[:, 2]) + 0.05
        robot_z = obs['agent'][0:12].reshape([4, 3])[:, 2].mean()
        dist = robot_z - chair_z

        if DEBUG:
            print('ActAdjustHeight: calc_z_dist {} {} {}'.format(chair_z, robot_z, dist))
        return dist


class ActAdjustHeightV2(BaseAction):

    def __init__(self):
        super(ActAdjustHeightV2, self).__init__()
        self.v = 1.0
        self.dist_Z, self.dist_Z_prev = None, None
        self.thresh_z = 0.05
        self.thresh_re = 3
        self.dist_Z_his = 100
        self.diff_same = 0

    def act(self, obs):
        v = self.v if abs(self.dist_Z) > self.thresh_z * 2 else (self.v / 2)
        v = v if self.dist_Z < 0 else -v
        action = get_action(move_z=v)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.obs = obs
        self.xyz = self.obs['pointcloud']['xyz']
        self.seg = self.obs['pointcloud']['seg']
        self.left_finger_mxyz, self.right_finger_mxyz = calc_finger_mean_pos(self.obs)
        self.finger_mxyz = (self.left_finger_mxyz + self.right_finger_mxyz) / 2.0
        if self.step_cnt == 0:
            self.chair_back_xyz, self.pre_chair_ground_xyz = calc_chair_back_pos(self.xyz, self.seg)
        self.dist_Z_prev, self.dist_Z = self.dist_Z, self.calc_z_dist(obs)
        if self.step_cnt == 0:
            self.dist_Z_prev = self.dist_Z
        if abs(self.dist_Z_prev - self.dist_Z) < 0.001: self.diff_same += 1

    @property
    def isDone(self):
        if self.dist_Z is None:
            return False
        else:
            return abs(self.dist_Z) <= self.thresh_z or (
                        self.dist_Z * self.dist_Z_prev) < 0 or self.diff_same >= self.thresh_re

    def calc_z_dist(self, obs):
        chair_z = self.pre_chair_ground_xyz.mean(axis=0)[2]
        robot_z = self.finger_mxyz[2]
        dist = robot_z - chair_z

        if DEBUG:
            print('ActAdjustHeight: calc_z_dist {} {} {}'.format(chair_z, robot_z, dist))
        return dist


class ActInitPose(BaseAction):

    def __init__(self, finger_pose):
        self.finger_pose = finger_pose
        self.action_none = get_action()
        self.actions_seq = (
            (get_action(armr=[0, 0, 0, 1, 0, 0, 0]), 3),
            (get_action(armr=[0, 1, 0, 0, 0, 0, 0]), 15),
            (get_action(arml=[0, 0, 0, 1, 0, 0, 0]), 3),
            (get_action(arml=[0, 1, 0, 0, 0, 0, 0]), 15),
        )
        self.max_step = max([x[1] for x in self.actions_seq])
        super(ActInitPose, self).__init__(self.max_step)

    def act(self, obs):
        if not self.isDone:
            i_step = self.step_cnt
            assert i_step < self.max_step
            action_init = copy.deepcopy(self.action_none)
            for act, step in self.actions_seq:
                if self.step_cnt < step:
                    action = [a + b for a, b in zip(action_init, act)]
                    action_init = copy.deepcopy(action)
            self.step_cnt += 1
            # override finger
            action[-2] = self.finger_pose
            action[-1] = self.finger_pose
            action[-10] = self.finger_pose
            action[-11] = self.finger_pose
            return action
        else:
            raise RuntimeError


class ActRelease(BaseAction):

    def __init__(self, finger_pose):
        self.finger_pose = finger_pose
        self.action_none = get_action()
        self.actions_seq = (
            (get_action(armr=[0, 0, 0, 1, 0, 0, 0]), 3),
            (get_action(armr=[0, -1, 0, 0, 0, 0, 0]), 15),
            (get_action(arml=[0, 0, 0, 1, 0, 0, 0]), 3),
            (get_action(arml=[0, -1, 0, 0, 0, 0, 0]), 15),
        )
        self.max_step = max([x[1] for x in self.actions_seq])
        super(ActRelease, self).__init__(self.max_step)

    def act(self, obs):
        if not self.isDone:
            i_step = self.step_cnt
            assert i_step < self.max_step
            action_init = copy.deepcopy(self.action_none)
            for act, step in self.actions_seq:
                if self.step_cnt < step:
                    action = [a + b for a, b in zip(action_init, act)]
                    action_init = copy.deepcopy(action)
            self.step_cnt += 1
            # override finger
            action[-2] = self.finger_pose
            action[-1] = self.finger_pose
            action[-10] = self.finger_pose
            action[-11] = self.finger_pose
            return action
        else:
            raise RuntimeError


class ActMoveToPlatformRot(BaseAction):

    def __init__(self, finger, holding):
        super(ActMoveToPlatformRot, self).__init__()
        self.v_rot = 0.2
        self.v = 1.0
        self.holding = holding
        self.finger = finger
        self.dist_ang, self.dist_ang_prev = None, None
        self.thresh_ang = 0.01

    def act(self, obs):
        # holding
        armR, armL = self.holding.act_impl(obs)
        # rot
        v_rot = self.v_rot if abs(self.dist_ang) > self.thresh_ang * 4 else (self.v_rot / 2)
        v_rot = v_rot if self.dist_ang < 0 else -v_rot
        action = get_action(rot=v_rot, move_z=-obs['agent'][49] * 5, finger=self.finger, armr=armR, arml=armL)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.dist_ang_prev, self.dist_ang = self.dist_ang, self.calc_angle_dist(obs)
        if self.step_cnt == 0:
            self.dist_ang_prev = self.dist_ang

    @property
    def isDone(self):
        if self.dist_ang is None:
            return False
        else:
            return abs(self.dist_ang) <= self.thresh_ang or (self.dist_ang * self.dist_ang_prev) < 0

    def calc_angle_dist(self, obs):
        robot_pos = obs['agent'][24:26]
        x, y = robot_pos[0], robot_pos[1]
        robot_ang = obs['agent'][26]
        if robot_ang < 0:
            robot_ang += np.pi * 2
        if x < 0 and abs(y) < abs(x):
            target_ang = 0.
        elif x > 0 and abs(y) < abs(x):
            target_ang = np.pi
        elif y > 0 and abs(y) > abs(x):
            target_ang = np.pi * 1.5
        elif y < 0 and abs(y) > abs(x):
            target_ang = np.pi * 0.5
        else:
            raise ValueError
        if target_ang > robot_ang:
            dist_1 = target_ang - robot_ang
            dist_2 = - (robot_ang + 2 * np.pi - target_ang)
            dist = dist_1 if abs(dist_1) < abs(dist_2) else dist_2
        elif target_ang < robot_ang:
            dist_1 = target_ang + (2 * np.pi - robot_ang)
            dist_2 = - (robot_ang - target_ang)
            dist = dist_1 if abs(dist_1) < abs(dist_2) else dist_2
        else:
            dist = 0.
        dist = -dist
        if DEBUG:
            print('ActMoveToPlatform: calc_angle_dist {} {} {}'.format(target_ang, robot_ang, dist))
        return dist


class ActMoveToPlatformXY(BaseAction):

    def __init__(self, finger, holding):
        super(ActMoveToPlatformXY, self).__init__()
        self.v_rot = 0.5
        self.v = 1.0
        self.holding = holding
        self.finger = finger
        self.dist_ang, self.dist_ang_prev = None, None
        self.dist_Y, self.dist_Y_prev = None, None
        self.dist_X, self.dist_X_prev = None, None
        self.thresh_ang = 0.01
        self.thresh_y = 0.01
        self.thresh_x = 0.05
        self.dist_x_offset = 0.1
        self.dist_y_offset = 0.0
        self.rect_center_fallback = None

    def act(self, obs):
        # holding
        armR, armL = self.holding.act_impl(obs)
        # rot
        if not self.isDoneY():
            v_y = self.v if abs(self.dist_Y) > self.thresh_y * 2 else (self.v / 2)
            v_y = v_y if self.dist_Y < 0 else -v_y
        else:
            v_y = 0
        if not self.isDoneX():
            v_x = self.v if abs(self.dist_X) > self.thresh_x * 2 else (self.v / 2)
            v_x = v_x if self.dist_X < 0 else -v_x
        else:
            v_x = 0
        action = get_action(move_x=v_x, move_y=v_y, move_z=-obs['agent'][49] * 5, finger=self.finger, armr=armR,
                            arml=armL)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.dist_Y_prev, self.dist_Y = self.dist_Y, self.calc_y_dist(obs)
        self.dist_X_prev, self.dist_X = self.dist_X, self.calc_x_dist(obs)
        if self.step_cnt == 0:
            self.dist_Y_prev = self.dist_Y
            self.dist_X_prev = self.dist_X

    @property
    def isDone(self):
        return all([self.isDoneX(), self.isDoneY()])

    def isDoneY(self):
        if self.dist_Y is None:
            return False
        else:
            return abs(self.dist_Y) <= self.thresh_y or (self.dist_Y * self.dist_Y_prev) < 0

    def isDoneX(self):
        if self.dist_X is None:
            return False
        else:
            return abs(self.dist_X) <= self.thresh_x or (self.dist_X * self.dist_X_prev) < 0

    def calc_xy_offset_impl(self, obs):
        # accu_pts
        xyz = obs['pointcloud']['xyz']
        seg = obs['pointcloud']['seg']
        # remove arm
        mask_arm = seg[:, 0]
        xyz = xyz[np.logical_not(mask_arm), :]
        # remove ground
        mask_floor = xyz[:, 2] < 0.005
        xyz = xyz[np.logical_not(mask_floor), :]
        # remove red point
        dist2target = pdist(np.zeros([3]), xyz)
        mask_red_point = dist2target < 0.16
        xyz = xyz[np.logical_not(mask_red_point), :]
        # remove chair foot
        mask_chair_foot = xyz[:, 2] <= 0.2
        xyz = xyz[np.logical_not(mask_chair_foot), :]

        xy = xyz[:, :2]
        xy_min = np.min(xy, axis=0)
        xy_shift = xy - xy_min
        rect = cv2.minAreaRect(xy_shift)
        rect_center = rect[0] + xy_min
        return rect_center

    def calc_xy_offset(self, obs):
        try:
            rect = self.calc_xy_offset_impl(obs)
            self.rect_center_fallback = rect
        except Exception:
            rect = self.rect_center_fallback
        return rect

    def calc_x_dist(self, obs):
        rect_center = self.calc_xy_offset(obs)
        xc, yc = rect_center
        robot_pos = obs['agent'][24:26]
        # x, y = robot_pos[0], robot_pos[1]
        x, y = xc, yc
        robot_ang = obs['agent'][26]
        if robot_ang < 0:
            robot_ang += np.pi * 2
        directions = np.asarray([0., 1., 2., 3., 4]) * np.pi / 2
        directions = np.argmin(np.abs(robot_ang - directions))
        if directions == 0 or directions == 4:
            dist = x
        elif directions == 2:
            dist = -x
        elif directions == 1:
            dist = y
        else:
            dist = -y
        dist -= self.dist_x_offset
        if DEBUG:
            print(
                'ActMoveToPlatform: calc_x_dist {} {} {} {} {}'.format(directions, robot_pos, dist, self.dist_x_offset,
                                                                       rect_center))
        return dist

    def calc_y_dist(self, obs):
        rect_center = self.calc_xy_offset(obs)
        xc, yc = rect_center
        robot_pos = obs['agent'][24:26]
        # x, y = robot_pos[0], robot_pos[1]
        x, y = xc, yc
        robot_ang = obs['agent'][26]
        if robot_ang < 0:
            robot_ang += np.pi * 2
        directions = np.asarray([0., 1., 2., 3., 4]) * np.pi / 2
        directions = np.argmin(np.abs(robot_ang - directions))
        if directions == 0 or directions == 4:
            dist = y
        elif directions == 2:
            dist = -y
        elif directions == 1:
            dist = -x
        else:
            dist = x
        dist -= self.dist_y_offset
        if DEBUG:
            print(
                'ActMoveToPlatform: calc_y_dist {} {} {} {} {}'.format(directions, robot_pos, dist, self.dist_x_offset,
                                                                       rect_center))
        return dist


class ArmFeedBack(BaseAction):

    def __init__(self, v=0.1, v_o=0.25):
        self.v_p = v
        self.v_o = v_o
        super(ArmFeedBack, self).__init__(1)
        self.state_refer = None

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        if self.step_cnt == 0:
            self.state_refer = obs['agent'][31:49].reshape(2, -1)

    def act_impl(self, obs):
        state_curr = obs['agent'][31:49].reshape(2, -1)
        state_diff = self.state_refer - state_curr
        v_right = []
        v_left = []
        for i in range(7):
            if abs(state_diff[0, i]) > 1e-2:
                v_right.append(self.v_p if state_diff[0, i] > 0 else -self.v_p)
            else:
                v_right.append(0)
            if abs(state_diff[1, i]) > 1e-2:
                v_left.append(self.v_p if state_diff[1, i] > 0 else -self.v_p)
            else:
                v_left.append(0)
        v_right[1] += self.v_o
        v_right[3] += -self.v_o
        v_left[1] += self.v_o
        v_left[3] += -self.v_o
        return v_right, v_left

    def act(self, obs):
        armr, arml = self.act_impl(obs)
        self.step_cnt += 1
        return get_action(armr=armr, arml=arml)


class PushChairPolicy(object):

    def __init__(self):
        self.circle = None
        self.bucket_h = None
        self.platform_xy = None

        # state
        self.finger_open = 1.0
        self.finger_close = -1.0

        # thresh
        self.rot_thresh = 0.002
        self.z_thresh = 0.01
        self.y_thresh = 0.01
        self.x_thresh = 0.01

        # v
        self.rot_v = 0.25
        self.move_z = 1.0
        self.move_y = 1.0
        self.move_x = 1.0

        # runtime info
        self.step_cnt = 0

        self.arm_feedback = ArmFeedBack(v=0.15, v_o=0.35)

        self.tasks = [
            ActFixedMove(35, move_x=1, move_z=-1),
            ActAdjustHeightV2(),
            ActInitPose(self.finger_open),
            self.arm_feedback,
            ActMoveToPlatformRot(self.finger_open, self.arm_feedback),
            ActMoveToPlatformXY(self.finger_open, self.arm_feedback),
            ActFixedMove(15),
            ActRelease(self.finger_open),
            ActFixedMove(1000),
        ]
        self.obs_accu = []
        self.bucket_center = None
        self.platform_center = None

    def act(self, obs):
        op = self.act_task(obs, self.tasks)
        if op is None:
            op = get_action(finger=self.finger_open)
            print('Warning: empyt op')
        self.step_cnt += 1
        if self.step_cnt > 190:
            op = get_action(finger=self.finger_open)
        return op

    def act_task(self, obs, tasklist):
        op = None
        for task in tasklist:
            if not task.isDone:
                task.update_state(obs, self.bucket_center, self.platform_center)
            if op is not None:
                break
            elif task.isDone:
                if hasattr(task, 'break_list') and task.break_list:
                    break
                else:
                    continue
            else:
                op = task.act(obs)
                break
        return op


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.agent = PushChairPolicy()

    def reset(self):
        self.agent = PushChairPolicy()

    def act(self, observation):
        try:
            action = self.agent.act(observation)
        except Exception:
            action = get_action()
        return action
