"""A series of reward functions."""

from flow.core.params import SumoParams
from gym.spaces import Box, Tuple
import numpy as np

from collections import defaultdict
from functools import reduce

from requests import head


ACC = 5.4

def total_lc_reward(env, rl_action):
    reward_dict = {
        'rl_desired_speed': rl_desired_speed(env),
        'rl_action_penalty': rl_action_penalty(env, rl_action),
        'uns4IDM_penalty': unsafe_distance_penalty4IDM(env),
        'meaningless_penalty': meaningless_penalty(env),
        'target_velocity': target_velocity(env),
    }
    return reward_dict


def target_velocity(env):

    return 0


def rl_desired_speed(env, rl):
    vel = np.array(env.k.vehicle.get_speed(rl))
    rl_des = env.initial_config.reward_params.get('rl_desired_speed', 0)
    target_vel = env.env_params.additional_params['target_velocity']
    regulation_speed = env.initial_config.reward_params.get('regulation_speed', 0)

    if rl_des == 0:
        return 0

    if vel < -100:
        return 0.

    if vel <= target_vel:
        cost = vel
        rwd = cost/target_vel
    else:
        cost = regulation_speed-vel
        rwd = cost/(regulation_speed-target_vel)

    return rl_des*rwd


def unsafe_distance_penalty4IDM(env, prev_lane, now_lane, tail_way, tail_speed, prev_lane_nums, now_lane_nums, rl):
    uns4IDM_p = env.initial_config.reward_params.get('uns4IDM_penalty', 0)

    #  Parameter of IDM
    T = 1
    a = b = ACC
    s0 = 5.45
    v = env.k.vehicle.get_speed(rl)
    tw = tail_way

    rwd = 0
    if tail_speed < 0:
        rwd = 0.

    else:
        s_star = 0
        # (prev_lane != now_lane) means that the agent have executed lane-changing
        # (prev_lane_nums == now_lane_nums) means that the number of lanes on the road is same before
        # It only work in the lane reduction scenario
        if (prev_lane != now_lane) and (prev_lane_nums == now_lane_nums):
            if abs(tw) < 1e-3:
                tw = 1e-3
                rwd = -3.

            else:
                follow_vel = tail_speed
                s_star = s0 + max(
                    0, follow_vel * T + follow_vel * (follow_vel - v) / (2 * np.sqrt(a * b)))
                rwd = uns4IDM_p * max(-3, min(0, 1 - (s_star / tw) ** 2))

    return rwd


def leader_unsafe_distance_penalty(env, headway, rl_speed, leader_speed, veh_length):
    leader_uns_p = env.initial_config.reward_params.get('leader_uns_penalty', 0)

    #  Parameter of IDM
    T = 1
    a = b = ACC
    s0 = 5.45

    # exception case: when the agent don't observe the leading vehicle on same lane
    if leader_speed < 0:
        return 0.

    s_star = s0 + max(
        0, rl_speed * T + rl_speed * (rl_speed - leader_speed) / (2 * np.sqrt(a * b)))

    if headway > s_star:
        rwd = 0.

    else:
        rwd = leader_uns_p * max(-5, min(0, 1 - (s_star / headway) ** 2))

    return rwd


def rl_action_penalty(env, actions, rl):
    action_penalty = env.initial_config.reward_params.get('rl_action_penalty', 0)

    if actions is None or action_penalty == 0:
        return 0

    actions = actions[rl]

    #  boolean condition
    if len(actions) == 2:
        direction = actions[1::2]
        for i in range(len(direction)):
            if direction[i] <= -0.333:
                direction[i] = -1
            elif direction[i] >= 0.333:
                direction[i] = 1
            else:
                direction[i] = 0

    elif len(actions) == 4:
        direction = actions[1:]

        if direction[0] == 1:
            direction = np.array([-1])
        elif direction[1] == 1:
            direction = np.array([0])
        else:
            direction = np.array([1])

    reward = 0
    if direction:
        if env.k.vehicle.get_previous_lane(rl) == env.k.vehicle.get_lane(rl):
            reward -= action_penalty

    return reward


def meaningless_penalty(env, prev_lane, now_lane, prev_headway, headway, prev_lane_nums,
                        now_lane_nums, rl, visibility_length):
    mlp = env.initial_config.reward_params.get('meaningless_penalty', 0)
    reward = 0
    lc_pen = 0

    if mlp:
        if (prev_lane != now_lane) and (prev_lane_nums == now_lane_nums):
            headway_criterion = headway - prev_headway
            semi_reward = - (headway_criterion - lc_pen)
            reward -= mlp * (semi_reward / visibility_length)

    return reward