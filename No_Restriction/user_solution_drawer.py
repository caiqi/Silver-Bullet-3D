import numpy as np
import copy

DEBUG = False


def get_action(move_x=0., move_y=0., move_z=0., rot=0., finger=1.0):
    assert -1. <= move_x <= 1.
    assert -1. <= move_y <= 1.
    assert -1. <= move_z <= 1.
    assert -1. <= rot <= 1.
    assert -1. <= finger <= 1.
    cmd = [move_x, move_y, rot, move_z] + [0. for _ in range(7)] + [finger, finger]
    return cmd


class BasePolicy(object):
    def __init__(self, opts=None):
        self.obs_mode = 'pointcloud'

    def act(self, observation):
        raise NotImplementedError()

    def reset(self):  # if you use an RNN-based policy, you need to implement this function
        pass


class SlidingAvgPosition(object):

    def __init__(self, lastN=1):
        self.memory = []
        assert lastN >= 1
        self.lastN = lastN

    def update(self, x):
        self.memory.append(x)

    def val(self):
        return sum(self.memory[-self.lastN:]) / self.lastN


class SubTaskPredefinedBase(object):

    def __init__(self, step_len):
        assert step_len > 0
        self.step_len = step_len
        self.step_cnt = 0

    @property
    def isDone(self):
        return not self.step_cnt < self.step_len

    def update_state(self, obs, xyz_handle, finger, multi, low):
        pass


class SubTaskFixedMoveZ(SubTaskPredefinedBase):

    def __init__(self, step_len, v=1.0):
        super(SubTaskFixedMoveZ, self).__init__(step_len)
        self.v = v
        self.op = [0, 0, 0, v, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def act(self, obs):
        if not self.isDone:
            self.step_cnt += 1
            return self.op
        else:
            raise RuntimeError


class SubTaskCombineArm(SubTaskPredefinedBase):

    def __init__(self):
        self.action_none = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.actions_seq = [
            ([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0], 5),
            ([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 11),
            ([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 10),
            ([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 5),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 10),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 0)
        ]
        self.max_step = max([x[1] for x in self.actions_seq])
        super(SubTaskCombineArm, self).__init__(self.max_step)

    def act(self, obs):
        if not self.isDone:
            action_init = copy.deepcopy(self.action_none)
            for act, step in self.actions_seq:
                if step - self.step_cnt > 0:
                    action = [a + b for a, b in zip(action_init, act)]
                    action_init = copy.deepcopy(action)
            self.step_cnt += 1
            return action
        else:
            raise RuntimeError


class SubTaskCombineArmRetryHigh(SubTaskPredefinedBase):

    def __init__(self):
        self.action_none = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.actions_seq = (
            ([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], 15),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 1], 8),
            ([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1], 10),
        )
        self.max_step = max([x[1] for x in self.actions_seq])
        super(SubTaskCombineArmRetryHigh, self).__init__(self.max_step)

    def act(self, obs):
        if not self.isDone:
            i_step = self.step_cnt
            assert i_step < self.max_step
            action_init = copy.deepcopy(self.action_none)
            for act, step in self.actions_seq:
                if self.max_step - i_step <= step:
                    action = [a + b for a, b in zip(action_init, act)]
                    action_init = copy.deepcopy(action)
            self.step_cnt += 1
            return action
        else:
            raise RuntimeError


class SubTaskCombineArmRetryLow(SubTaskPredefinedBase):

    def __init__(self):
        self.action_none = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.actions_seq = (
            ([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1], 15),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1], 10),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 1], 8),
            ([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1], 20),
        )
        self.max_step = max([x[1] for x in self.actions_seq])
        super(SubTaskCombineArmRetryLow, self).__init__(self.max_step)

    def act(self, obs):
        if not self.isDone:
            i_step = self.step_cnt
            assert i_step < self.max_step
            action_init = copy.deepcopy(self.action_none)
            for act, step in self.actions_seq:
                if self.max_step - i_step <= step:
                    action = [a + b for a, b in zip(action_init, act)]
                    action_init = copy.deepcopy(action)
            self.step_cnt += 1
            return action
        else:
            raise RuntimeError


class SubTaskCalibArm(SubTaskPredefinedBase):

    def __init__(self):
        super(SubTaskCalibArm, self).__init__(10000)
        self.tasks = [
            SubTaskFixedMoveZ(10, 1.0),
            SubTaskCombineArm()
        ]
        self.condition = True

    def act(self, obs):
        if not self.isDone:
            for task in self.tasks:
                if not task.isDone:
                    return task.act(obs)
        else:
            raise RuntimeError

    @property
    def isDone(self):
        if self.condition:
            if all([task.isDone for task in self.tasks]):
                return True
            else:
                return False
        else:
            return True

    def update_state(self, obs, xyz_handle, finger, multi, low):
        self.condition = low


class SubTaskCalibAngle(SubTaskPredefinedBase):
    
    def __init__(self, v, threshold, finger_open_pose):
        super(SubTaskCalibAngle, self).__init__(10000)
        self.v = v
        self.threshold = threshold
        self.finger_open_pose = finger_open_pose
        self.debug = False
        self.dist = None
        self.dist_prev = None

    def act(self, obs):
        if abs(self.dist) > self.threshold * 4:
            v_curr = self.v
        else:
            v_curr = self.v / 2
        action = get_action(rot=-v_curr if self.dist > 0 else v_curr, finger=self.finger_open_pose)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xyz_handle, finger, multi, low):
        self.dist_prev = self.dist
        self.dist = self.calc_orientatation(obs)
        if self.step_cnt == 0:
            self.dist_prev = self.dist

    @property
    def isDone(self):
        if self.dist is None:
            return False
        else:
            return abs(self.dist) <= self.threshold or (self.dist * self.dist_prev) < 0

    def calc_orientatation(self, obs):
        robot_angle = obs['agent'][14]
        if self.debug:
            print('robot_angle={:>0.5f}'.format(robot_angle))
        return robot_angle


class SubTaskCalibZ(SubTaskPredefinedBase):

    def __init__(self, v, threshold, finger_open_pose, print_log=False):
        super(SubTaskCalibZ, self).__init__(10000)
        self.v = v
        self.threshold = threshold
        self.finger_open_pose = finger_open_pose
        self.debug = False
        self.dist = None
        self.dist_prev = None
        self.print_log = print_log

    def act(self, obs):
        if abs(self.dist) > self.threshold * 4:
            v_curr = self.v
        else:
            v_curr = self.v / 2
        action = get_action(move_z=v_curr if self.dist > 0 else -v_curr, finger=self.finger_open_pose)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xyz_handle, finger, multi, low):
        self.dist_prev = self.dist
        self.dist = self.calc_dist_axis_z(obs, xyz_handle)
        if self.step_cnt == 0:
            self.dist_prev = self.dist
            if self.print_log:
                print('adjust z')
                print(xyz_handle)
                print(self.dist)

    @property
    def isDone(self):
        if self.dist is None:
            return False
        else:
            return abs(self.dist) <= self.threshold or (self.dist * self.dist_prev) < 0

    def calc_dist_axis_z(self, obs, handle_xyz):
        xyz_handle_center = handle_xyz
        xyz_finger = obs['agent'][0:6]
        h_handle = xyz_handle_center[-1]
        h_finger = (xyz_finger[2] + xyz_finger[5]) / 2
        diff = h_handle - h_finger
        if self.debug:
            print('h_handle={:>0.5f}, h_finger={:>0.5f}, dist={:>0.5f}'.format(h_handle, h_finger, diff))
        return diff


class SubTaskCalibY(SubTaskPredefinedBase):

    def __init__(self, v, threshold, finger_pose):
        super(SubTaskCalibY, self).__init__(10000)
        self.v = v
        self.threshold = threshold
        self.finger_pose = finger_pose
        self.debug = False
        self.dist = None
        self.dist_prev = None

    def act(self, obs):
        if abs(self.dist) > self.threshold * 4:
            v_curr = self.v
        else:
            v_curr = self.v / 2
        action = get_action(move_y=v_curr if self.dist > 0 else -v_curr, finger=self.finger_pose)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xyz_handle, finger, multi, low):
        self.dist_prev = self.dist
        self.dist = self.calc_dist_axis_y(obs, xyz_handle)
        if self.step_cnt == 0:
            self.dist_prev = self.dist

    @property
    def isDone(self):
        if self.dist is None:
            return False
        else:
            return abs(self.dist) <= self.threshold or (self.dist * self.dist_prev) < 0

    def calc_dist_axis_y(self, obs, handle_xyz):
        xyz_handle_center = handle_xyz
        xyz_finger = obs['agent'][0:6]
        y_handle = xyz_handle_center[1]
        y_finger = (xyz_finger[1] + xyz_finger[1]) / 2
        diff = y_handle - y_finger
        if self.debug:
            print('y_handle={:>0.5f}, y_finger={:>0.5f}, dist={:>0.5f}'.format(y_handle, y_finger, diff))
        return diff


class SubTaskCalibX(SubTaskPredefinedBase):

    def __init__(self, v, threshold, finger_pose, dynamic_v=True):
        super(SubTaskCalibX, self).__init__(10000)
        self.v = v
        self.threshold = threshold
        self.finger_pose = finger_pose
        self.debug = False
        self.dist = None
        self.dist_prev = None
        self.dynamic_v = dynamic_v

    def act(self, obs):
        if self.dynamic_v:
            v_curr = self.v if abs(self.dist) > self.threshold * 4 else self.v / 2
        else:
            v_curr = self.v
        action = get_action(move_x=v_curr if self.dist > 0 else -v_curr, finger=self.finger_pose)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xyz_handle, finger, multi, low):
        self.dist_prev = self.dist
        self.dist = self.calc_dist_axis_x(obs, xyz_handle)
        if self.step_cnt == 0:
            self.dist_prev = self.dist

    @property
    def isDone(self):
        if self.dist is None:
            return False
        elif abs(self.dist) <= self.threshold or (self.dist * self.dist_prev) < 0:
            return True
        elif self.step_cnt > 5 and abs(self.dist_prev - self.dist) < self.threshold * 0.2:
            return True
        else:
            return False

    @staticmethod
    def calc_dist_axis_x(obs, handle_xyz):
        xyz_handle_center = handle_xyz
        xyz_finger = obs['agent'][0:6]
        x_handle = xyz_handle_center[0]
        x_finger = (xyz_finger[0] + xyz_finger[0]) / 2
        diff = x_handle - x_finger
        if DEBUG:
            print('x_handle={:>0.5f}, x_finger={:>0.5f}, diff={:>0.5f}'.format(x_handle, x_finger, diff))
        return diff


class SubTaskFixedMove(SubTaskPredefinedBase):

    def __init__(self, step_len, move_x=0., move_y=0., move_z=0., rot=0., finger=1.0):
        super(SubTaskFixedMove, self).__init__(step_len)
        self.action = get_action(move_x=move_x, move_y=move_y, move_z=move_z, rot=rot, finger=finger)

    def act(self, obs):
        if not self.isDone:
            self.step_cnt += 1
            return self.action
        else:
            raise RuntimeError


class SubTaskPull(SubTaskPredefinedBase):

    def __init__(self, v, inc_num, step_num, suffix_num, finger, thresh, pullout_thresh):
        self.v = v
        self.num1 = inc_num
        self.num2 = step_num
        self.num3 = suffix_num
        self.finger = finger
        self.bugger_finger_thresh = thresh
        self.dist = None
        self.handle_xyz_init = None
        self.pullout_thresh = pullout_thresh
        self.pullout = None
        self.break_list = False
        super(SubTaskPull, self).__init__(self.num1 * self.num2 + self.num3)

    def act(self, obs):
        if self.step_cnt < self.num1 * self.num2:
            v_scale = self.step_cnt // self.num2 + 1
            action = get_action(move_x=-self.v * 0.1 * v_scale, finger=self.finger)
        else:
            action = get_action(move_x=-self.v, finger=self.finger)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xyz_handle, finger, multi, low):
        if self.step_cnt > 0:
            self.dist = SubTaskCalibX.calc_dist_axis_x(obs, xyz_handle)
            # print(self.dist, self.bugger_finger_thresh)
        if self.step_cnt == 0:
            self.handle_xyz_init = xyz_handle

    @property
    def butter_finger(self):
        if self.dist is not None:
            return abs(self.dist) > self.bugger_finger_thresh
        else:
            return False

    def isPulledOut(self, handle_xyz):
        if self.pullout is None:
            self.pullout = abs(handle_xyz[0] - self.handle_xyz_init[0]) > self.pullout_thresh
            # print(abs(handle_xyz[0] - self.handle_xyz_init[0]))
        return self.pullout

    @property
    def isDone(self):
        if self.step_cnt == 0:
            return False
        elif self.butter_finger:
            # print('skip butter_finger')
            self.break_list = True
            return True
        elif super(SubTaskPull, self).isDone:
            return True
        else:
            return False


class DrawerPolicy(object):

    def __init__(self):
        # init hyper-params
        self.finger_open_pos = 0.1
        self.finger_close_pos = -1.0
        self.rot_thresh = 0.002
        self.z_thresh = 0.01
        self.y_thresh = 0.01
        self.x_thresh = 0.01
        self.rot_v = 0.1
        self.move_z = 1.0
        self.move_y = 1.0
        self.move_x = 1.0
        self.low_handle_thresh = 0.5
        self.pullout_thresh = 0.1
        # runtime info
        self.xyz_handle_center_fallback = SlidingAvgPosition(1)
        self.step_cnt = 0
        self.butter_finger = None
        self.handle_multi = None
        self.handle_low = None

        self.tasks = [
            SubTaskCalibArm(),
            SubTaskCalibAngle(self.rot_v, self.rot_thresh, self.finger_open_pos),
            SubTaskCalibZ(self.move_z, self.z_thresh, self.finger_open_pos),
            SubTaskCalibY(self.move_y, self.y_thresh, self.finger_open_pos),
            SubTaskCalibX(self.move_x, self.x_thresh, self.finger_open_pos),
            SubTaskCalibZ(self.move_z, self.z_thresh, self.finger_open_pos),
            SubTaskFixedMove(1, finger=self.finger_close_pos),
            SubTaskPull(self.move_x, 10, 4, 5, self.finger_close_pos, self.x_thresh * 4, self.pullout_thresh),
            SubTaskFixedMove(60, move_x=-self.move_x, finger=self.finger_close_pos)
        ]

        self.task_retry_wait = SubTaskFixedMove(7, finger=self.finger_close_pos)

        self.task_retry_grasp = []

        self.task_retry_pull = [
            SubTaskCalibX(self.move_x, self.x_thresh, self.finger_open_pos),
            SubTaskCalibZ(self.move_z, self.z_thresh, self.finger_open_pos),
            SubTaskFixedMove(1, finger=self.finger_close_pos),
            SubTaskPull(self.move_x, 10, 4, 5, self.finger_close_pos, self.x_thresh * 4, self.pullout_thresh),
        ]

    def act(self, obs):
        if self.step_cnt == 0:
            self.__init_env(obs)
        op = self.act_task(obs, self.tasks)
        if op is None and self.tasks[-2].butter_finger:
            if self.task_retry_wait.isDone:
                pullout = self.tasks[-2].isPulledOut(self.__util_func_calc_handle_position(obs))
                op = self.act_task(obs, self.task_retry_grasp if pullout else self.task_retry_pull)
            else:
                op = self.task_retry_wait.act(obs)
        if op is None:
            op = [0. for _ in range(13)]
            # print('Warning: empyt op')
        self.step_cnt += 1
        return op

    def act_task(self, obs, tasklist):
        op = None
        for task in tasklist:
            xyz_handle = self.__util_func_calc_handle_position(obs)
            if not task.isDone:
                task.update_state(obs, xyz_handle, self.butter_finger, self.handle_multi, self.handle_low)
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

    def __init_env(self, obs):
        self.handle_multi = self.__util_func_check_multi_handle(obs)
        self.xyz_handle_center = self.__util_func_calc_handle_position(obs)
        self.handle_low = self.xyz_handle_center[2] < 0.5
        self.task_retry_grasp = [
            SubTaskCombineArmRetryLow() if self.handle_low else SubTaskCombineArmRetryHigh(),
            SubTaskCalibX(self.move_x, self.x_thresh, self.finger_open_pos, False),
            SubTaskFixedMove(2, move_x=self.move_x),
            SubTaskFixedMove(10, move_z=-self.move_z),
            SubTaskFixedMove(20, move_x=-self.move_x),
        ]

    def __util_func_check_multi_handle(self, obs):
        xyz = obs['pointcloud']['xyz']
        seg = obs['pointcloud']['seg']
        xyz_handle = xyz[seg[:, 0], :]
        xyz_handle_center = (np.max(xyz_handle, axis=0) + np.min(xyz_handle, axis=0)) / 2
        dist = np.sqrt(np.sum((xyz_handle_center - xyz_handle) ** 2, axis=-1))
        if np.min(dist) > 0.05:
            # print('multi handle: {}'.format(np.min(dist)))
            return True
        else:
            # print('single handle: {}'.format(np.min(dist)))
            return False

    def __util_func_calc_handle_position(self, obs):
        xyz = obs['pointcloud']['xyz']
        seg = obs['pointcloud']['seg']
        if not np.any(seg[:, 0]):
            return self.xyz_handle_center_fallback.val()
        elif np.sum(seg[:, 0]) == 1:
            return xyz[seg[:, 0], :].squeeze()
        xyz_handle = xyz[seg[:, 0], :]
        if self.handle_multi:
            std_x = np.std(xyz_handle[:, 0])
            std_y = np.std(xyz_handle[:, 1])
            v = xyz_handle[:, 0] if std_x > std_y else xyz_handle[:, 1]
            v_sorted_idx = np.argsort(v)
            v_sorted_value = v[v_sorted_idx]
            v_neighbor_diff = v_sorted_value[1:] - v_sorted_value[:-1]
            v_slice_idx = np.argmax(v_neighbor_diff)
            idx_handle1 = v_sorted_idx[:v_slice_idx + 1]
            xyz_handle = xyz_handle[idx_handle1, :]
        xyz_handle_center = (np.max(xyz_handle, axis=0) + np.min(xyz_handle, axis=0)) / 2
        # update runtime info
        self.xyz_handle_center_fallback.update(xyz_handle_center)
        return xyz_handle_center


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.obs_mode = 'pointcloud'
        self.agent = DrawerPolicy()

    def reset(self):
        self.agent = DrawerPolicy()

    def act(self, observation):
        action = self.agent.act(observation)
        return action
