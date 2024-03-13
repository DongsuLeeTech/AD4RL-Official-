from pickle import FALSE, TRUE
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController
from flow.envs.multiagent import MADRingLCPOEnv
from flow.networks.bottle_ring import myNetwork
import math

#for MRL
from flow.utils.registry import make_create_env
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.tune.registry import register_env

import os

# name of this file
current_file_name_py = os.path.abspath(__file__).split('/')[-1]
# remove file extension
current_file_name = current_file_name_py[:-3]

HORIZON = 3000
N_ROLLOUTS = 5
N_CPUS = 30

NUM_AUTOMATED = 1
num_human = 32 - NUM_AUTOMATED
humans_remaining = num_human

Acc = 5.4
DAcc = 5.4
Max_vel = 72.222

vehicles = VehicleParams()

import random
seed1 = random.randrange(1, 1005)
seed2 = random.randrange(1, 1005)
print('seed1 :', seed1)
print('seed2 :', seed2)

vehicles = VehicleParams()
for i in range(NUM_AUTOMATED):
    # Add one automated vehicle.
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        initial_speed=0,
        num_vehicles=1,

        car_following_params=SumoCarFollowingParams(
            speed_mode= 0,
            min_gap=1,
            max_speed=Max_vel,
        ),
        length=4.45
    )

# Add a fraction of the remaining human vehicles.
vehicles.add(
    veh_id="human_7d5",
    acceleration_controller=(IDMController, {'v0': 9.139}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1045, model='LC2013', lc_cooperative=1.,
                                            lc_keep_right=100000000.,),
    initial_speed=0,
    num_vehicles=2,
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
    ),
    length=4.45
)

vehicles.add(
    veh_id="human_12d5",
    acceleration_controller=(IDMController, {'v0': 13.493}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1045, model='LC2013',
                                            lc_cooperative=0.,
                                            lc_look_ahead_left=0.1, lc_speed_gain_right=1.0, lc_assertive=1.,
                                            kwargs={"lcLookaheadLeft": 0.1, "lcSpeedGainRight": 10.,
                                                    'lcAssertive': 100.}
                                            ),
    initial_speed=0,
    num_vehicles=20,
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
    ),
    length=4.45,
    color='orange'
)

vehicles.add(
    veh_id="human_10",
    acceleration_controller=(IDMController, {'v0': 14.769}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_controller=(SimLaneChangeController, {}),
        lane_change_params=SumoLaneChangeParams(lane_change_mode=1045, model='LC2013',
                                                lc_cooperative=1.,
                                                lc_look_ahead_left=0.1, lc_speed_gain_right=1.0, lc_assertive=1.,
                                                kwargs={"lcLookaheadLeft": 0.1, "lcSpeedGainRight": 10.,
                                                        'lcAssertive': 100.}
                                                ),
    initial_speed=0,
    num_vehicles=18,
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
    ),
    length=4.45,
    color='blue'
)

vehicles.add(
    veh_id="human_11",
    acceleration_controller=(IDMController, {'v0': 16.258}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_controller=(SimLaneChangeController, {}),
        lane_change_params=SumoLaneChangeParams(lane_change_mode=1045, model='LC2013',
                                                lc_cooperative=1.,
                                                lc_look_ahead_left=0.1, lc_speed_gain_right=1.0, lc_assertive=1.,
                                                kwargs={"lcLookaheadLeft":0.1, "lcSpeedGainRight":10., 'lcAssertive':100.}
                                                ),
    initial_speed=0,
    num_vehicles=4,
    car_following_params=SumoCarFollowingParams(
        speed_mode=7,
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
    ),
    length=4.45,
    color='green'
)

flow_params = dict(
    exp_tag=current_file_name,
    seed=seed1,
    env_name=MADRingLCPOEnv,
    network=myNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=True,
        seed=seed2,
    ),

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=900,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": 505,
            "lane_change_duration": 0,
            "target_velocity": 13.493,
            'sort_vehicles': False
        },
    ),
    net=NetParams(
        additional_params={
            "length": 505,
            "num_lanes": 4,
            "speed_limit": Max_vel,
            "resolution": 40,
        },
    ),

    veh=vehicles,
    initial=InitialConfig(
        spacing='custom2',
        reward_params={
            'default_reward' : -1.,
            'rl_desired_speed': 0.6,
            'meaningless_penalty': 0.48,
            'uns4IDM_penalty': 0.3,
            'leader_uns_penalty': 0.3,
            'rl_action_penalty': 0.1,
            'alpha' : 1,
            'regulation_speed':31.389,
            'accident_penalty': -5.,
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