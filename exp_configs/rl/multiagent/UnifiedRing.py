from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter,  SimLaneChangeController
from flow.envs.multiagent import MADRingLCPOEnv
# from flow.networks.ring import RingNetwork
from flow.networks.lane_change_ring import RingNetwork
from flow.utils.registry import make_create_env

import numpy as np
import random
import math
import os

current_file_name_py = os.path.abspath(__file__).split('/')[-1]
current_file_name = current_file_name_py[:-3]

HORIZON = 3000
N_ROLLOUTS = 5
N_CPUS = 30
NUM_AUTOMATED = 1

Acc = 5.4
Max_vel = 72.222

vehicles = VehicleParams()
for j in range(15):
    vehicles.add(
        veh_id="human_{}".format(j),
        acceleration_controller=(IDMController, {
            'v0': 9.138,
            "noise": 0.
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0
        ),
        routing_controller=(ContinuousRouter, {}),
        initial_speed=0,
        num_vehicles=1)

# Add one automated vehicle.
vehicles.add(
    veh_id="rl_{}".format(0),
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=1,
        decel=Acc,
        accel=Acc,
        max_speed=Max_vel
    ),
    initial_speed=0,
    length=4.45,
    num_vehicles=1)


flow_params = dict(
    # name of the experiment
    exp_tag=current_file_name,

    # random seed
    seed=1000,

    # name of the flow environment the experiment is running on
    env_name=MADRingLCPOEnv,

    # name of the network class the experiment is running on
    network=RingNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=90,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "target_velocity": 13.686,
            "ring_length": [670, 670],
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "length": 670,
            "lanes": 3,
            "speed_limit": Max_vel,
            "resolution": 40,
        }, ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom2',
        reward_params={
            'default_reward': -1.,
            'accident_penalty': -5.,
            'rl_desired_speed': .6,
            'meaningless_penalty': .48,
            'uns4IDM_penalty': 0.3,
            'leader_uns_penalty': .6,
            'rl_action_penalty': .1,
            'alpha': 1,
            'regulation_speed':31.389,
        },
    ),
)

create_env, env_name = make_create_env(params=flow_params, version=0)

register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy():
    return DDPGTorchPolicy, obs_space, act_space, {}

POLICY_GRAPHS = {'av': gen_policy()}

def policy_mapping_fn(_):
    return 'av'

POLICIES_TO_TRAIN = ['av']
