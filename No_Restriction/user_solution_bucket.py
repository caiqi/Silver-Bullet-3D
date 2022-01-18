import gym
import mani_skill.env
import numpy as np
import copy

DEBUG = False


def pdist(pt, pts):
    return np.sqrt(np.sum((pts - pt) ** 2, axis=1))


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
                action = get_action(move_x=self.vx, move_y=self.vy, move_z=self.vz, rot=self.rot, finger=self.finger, armr=armR, arml=armL)
            else:
                action = get_action(move_x=self.vx, move_y=self.vy, move_z=self.vz, rot=self.rot, finger=self.finger, arm=self.arm)
            return action
        else:
            raise RuntimeError

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        pass


class ActTurnToBucket(BaseAction):

    def __init__(self, v, thresh, finger):
        super(ActTurnToBucket, self).__init__()
        self.v = v
        self.thresh = thresh
        self.thresh_full = self.thresh * 4
        self.finger = finger
        self.dist = None
        self.dist_prev = None

    def act(self, obs):
        v_curr = self.v if abs(self.dist) > self.thresh * 4 else (self.v / 2)
        v_curr = v_curr if self.dist < 0 else -v_curr
        action = get_action(rot=v_curr, finger=self.finger)
        self.step_cnt += 1
        return action

    def update_state(self, obs, xy_bucket=None, xy_platform=None):
        self.dist_prev = self.dist
        self.dist = self.calc_dist(obs, xy_bucket[:2], xy_platform)
        if self.step_cnt == 0:
            self.dist_prev = self.dist

    @property
    def isDone(self):
        if self.dist is None:
            return False
        else:
            return abs(self.dist) <= self.thresh or (self.dist * self.dist_prev) < 0

    def calc_dist(self, obs, xy_bucket, xy_platform):
        robot_angle = obs['agent'][26]
        xy_src = obs['agent'][24:26]
        xy_tar = xy_bucket
        x_src = xy_src[0]
        y_src = xy_src[1]
        x_tar = xy_tar[0]
        y_tar = xy_tar[1]
        if x_tar > x_src:
            theta = np.arctan((y_tar - y_src) / (x_tar - x_src))
        else:
            theta = np.pi - np.arctan(- (y_tar - y_src) / (x_tar - x_src))
        theta_arr = [theta,]
        theta_arr.append(theta + 2 * np.pi * (1 if theta < 0 else -1))
        theta_arr.append(theta + 4 * np.pi * (1 if theta < 0 else -1))
        theta_arr.append(theta + 2 * np.pi * (1 if theta > 0 else -1))
        robot_angle_diff = robot_angle - np.array(theta_arr)
        dist = robot_angle_diff[np.argmin(np.abs(robot_angle_diff))]
        if DEBUG:
            print('ActTurnToBucket', xy_src, xy_tar, theta_arr, robot_angle, dist)
        return dist


class ActInitPose(BaseAction):
    def __init__(self, finger_pose):
        self.finger_pose = finger_pose
        self.action_none = get_action()
        self.actions_seq = (
            (get_action(armr=[-1, 0, 0, 0, 0, 0, 0]), 15),
            (get_action(armr=[0, 1, 0, 0, 0, 0, 0]), 10),
            (get_action(armr=[0, 0, 1, 0, 0, 0, 0]), 15),
            (get_action(armr=[0, 0, 0, -1, 0, 0, 0]), 6),
            (get_action(armr=[0, 0, 0, 0, -1, 0, 0]), 9),
            (get_action(arml=[-1, 0, 0, 0, 0, 0, 0]), 5),
            (get_action(arml=[0, -1, 0, 0, 0, 0, 0]), 10),
            (get_action(arml=[0, 0, 1, 0, 0, 0, 0]), 5),
            (get_action(arml=[0, 0, 0, -1, 0, 0, 0]), 6),
            (get_action(arml=[0, 0, 0, 0, 1, 0, 0]), 8),
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


class ActHoldingUp(BaseAction):
    def __init__(self, finger_pose):
        self.finger_pose = finger_pose
        self.action_none = get_action()
        self.actions_seq = (
            (get_action(armr=[0, 0, -1, 0, 0, 0, 0]), 10),
            (get_action(armr=[0, 0, 0, 0, 0, -1, 0]), 5),
            (get_action(arml=[0, 0, 1, 0, 0, 0, 0]), 10),
            (get_action(arml=[0, 0, 0, 0, 0, -1, 0]), 5),
        )
        self.max_step = max([x[1] for x in self.actions_seq])
        super(ActHoldingUp, self).__init__(self.max_step)

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


class ActArmRelease(BaseAction):
    def __init__(self, finger_pose):
        self.finger_pose = finger_pose
        self.action_none = get_action()
        self.actions_seq = (
            (get_action(armr=[0, 0, 1, 0, 0, 0, 0]), 10),
            (get_action(armr=[0, 0, 0, 0, 0, 1, 0]), 5),
            (get_action(arml=[0, 0, -1, 0, 0, 0, 0]), 10),
            (get_action(arml=[0, 0, 0, 0, 0, 1, 0]), 5),
        )
        self.max_step = max([x[1] for x in self.actions_seq])
        super(ActArmRelease, self).__init__(self.max_step)

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


class ArmFeedBack(BaseAction):

    def __init__(self, v=0.1):
        self.v_p = v
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
        v_right[2] += -0.15
        v_right[-2] += -0.15
        v_left[2] += 0.15
        v_left[-2] += -0.15
        return v_right, v_left

    def act(self, obs):
        armr, arml = self.act_impl(obs)
        self.step_cnt += 1
        return get_action(armr=armr, arml=arml)


class ActMoveToPlatformRot(BaseAction):

    def __init__(self, finger, holding):
        super(ActMoveToPlatformRot, self).__init__()
        self.v_rot = 0.5
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
        self.dist_x_offset = 0.44

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
        action = get_action(move_x=v_x, move_y=v_y, move_z=-obs['agent'][49] * 5, finger=self.finger, armr=armR, arml=armL)
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

    def calc_x_dist(self, obs):
        robot_pos = obs['agent'][24:26]
        x, y = robot_pos[0], robot_pos[1]
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
        if dist > 0:
            dist -= self.dist_x_offset
        else:
            dist += self.dist_x_offset
        if DEBUG:
            print('ActMoveToPlatform: calc_x_dist {} {} {}'.format(directions, robot_pos, dist))
        return dist

    def calc_y_dist(self, obs):
        robot_pos = obs['agent'][24:26]
        x, y = robot_pos[0], robot_pos[1]
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
        if DEBUG:
            print('ActMoveToPlatform: calc_y_dist {} {} {}'.format(directions, robot_pos, dist))
        return dist


class BucketPolicy(object):

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

        self.arm_feedback = ArmFeedBack(v=0.15)

        self.tasks = [
            ActTurnToBucket(self.rot_v, self.rot_thresh, self.finger_open),
            ActInitPose(self.finger_open),
            ActFixedMove(20, move_z=-0.95, move_x=1),
            ActTurnToBucket(self.rot_v, self.rot_thresh, self.finger_open),
            ActHoldingUp(self.finger_open),
            self.arm_feedback,
            ActFixedMove(30, move_z=self.move_z, holding=self.arm_feedback),
            ActMoveToPlatformRot(self.finger_open, self.arm_feedback),
            ActMoveToPlatformXY(self.finger_open, self.arm_feedback),
            ActFixedMove(30, move_z=-self.move_z, holding=self.arm_feedback),
            ActArmRelease(self.finger_open),
            ActFixedMove(20000),
        ]

    def act(self, obs):
        if self.step_cnt == 0:
            self.__init_env(obs)
        op = self.act_task(obs, self.tasks)
        if op is None:
            op = get_action(finger=self.finger_open)
            print('Warning: empyt op')
        self.step_cnt += 1
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

    def __init_env(self, obs):
        self.sbucket_circle_top, self.bucket_h, self.platform_center = self.__util_func_calc_position(obs)
        self.bucket_center = np.asarray([self.sbucket_circle_top[0],
                                         self.sbucket_circle_top[1],
                                         self.sbucket_circle_top[2],
                                         self.bucket_h])

    def __util_func_calc_position(self, obs):
        bucket_beg = 0.05
        xyz = obs['pointcloud']['xyz']
        seg = obs['pointcloud']['seg']
        xyz = xyz[np.logical_not(seg[:, 0]), :]
        xyz = xyz[xyz[:, 2] > bucket_beg, :]
        platform_seg = np.logical_and(np.logical_and(xyz[:, 0] > -0.31, xyz[:, 0] < 0.31),
                                      np.logical_and(xyz[:, 1] > -0.31, xyz[:, 1] < 0.31))
        bucket_xyz = xyz[np.logical_not(platform_seg), :]

        ball_xyz_top5 = bucket_xyz[np.argsort(bucket_xyz[:, 2])[-5:], :]
        ball_xyz_center = np.mean(ball_xyz_top5, axis=0)
        ball_xyz_dist = pdist(ball_xyz_center[:2], bucket_xyz[:, :2])
        ball_xyz_seg = ball_xyz_dist < 0.1
        bucket_xyz = bucket_xyz[np.logical_not(ball_xyz_seg), :]
        platform_center = np.asarray([0., 0.])

        bucket_pt_cnt = []
        bucket_pt_z = bucket_xyz[:, 2]
        for i in range(100):
            bucket_pt_cnt.append(np.sum(bucket_pt_z > (0.11 + i * 0.01)))
            if bucket_pt_cnt[-1] == 0:
                break
        bucket_h = 0.11
        for i in range(1, len(bucket_pt_cnt)):
            if bucket_pt_cnt[i] / bucket_pt_cnt[i - 1] < 0.75:
                bucket_h = bucket_h + 0.01 * i
                break

        bucket_circle_top = np.asarray((0., 0., 0.))
        bucket_circle_top[0] = ball_xyz_center[0]
        bucket_circle_top[1] = ball_xyz_center[1]
        return bucket_circle_top, bucket_h, platform_center


class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        self.agent = BucketPolicy()

    def reset(self):
        self.agent = BucketPolicy()

    def act(self, observation):
        try:
            action = self.agent.act(observation)
        except Exception:
            action = get_action()
        return action


def visual():
    render = False

    env = gym.make('MoveBucket-v0')
    env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
    env._max_episode_steps = 200

    agent = UserPolicy('MoveBucket-v0')

    with open('user_solution_bucket_v7.log.txt', 'r') as f:
        lines = f.readlines()
        seeds = [int(l.strip().split(' ')[0]) for l in filter(lambda x: 'False' in x, lines)]

    with open('user_solution_bucket_v8.log.txt', 'a') as f:
        for level_idx in range(20000, 99999):
            obs = env.reset(level=level_idx)
            agent.reset()
            if render:
                env.render('human')
            for step in range(20000):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                if render:
                    env.render('human')
                if done:
                    break
            print('{} {}'.format(level_idx, info))
            f.write('{} {}\n'.format(level_idx, info))


if __name__ == '__main__':
    visual()
