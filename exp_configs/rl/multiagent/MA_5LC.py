from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.registry import register_env

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv
from flow.networks import RingNetwork
from flow.utils.registry import make_create_env

from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from flow.envs.multiagent import MADRingLCPOEnv

from flow.controllers import RLController, IDMController, ContinuousRouter, SimLaneChangeController

HORIZON = 3000
N_ROLLOUTS = 5
N_CPUS = 30

NUM_AUTOMATED = 1

vehicles = VehicleParams()
Acc = 5.4
DAcc = 5.4
Max_vel = 72.222

import random
seed1 = random.randrange(1, 1005)
seed2 = random.randrange(1, 1005)
print('seed1 :', seed1)
print('seed2 :', seed2)

vehicles = VehicleParams()

vehicles.add(
    veh_id="human_7d5",
    acceleration_controller=(IDMController, {'v0': 9.139}),
    routing_controller=(ContinuousRouter, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    lane_change_params=SumoLaneChangeParams(lane_change_mode=1045, model='LC2013', lc_cooperative=1.,
                                            lc_keep_right=100000000.,),
    initial_speed=0,
    num_vehicles=4,
    car_following_params=SumoCarFollowingParams(
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
        s0=0
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
    num_vehicles=52,
    car_following_params=SumoCarFollowingParams(
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
        s0=0
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
    num_vehicles=48,
    car_following_params=SumoCarFollowingParams(
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
        s0=0
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
    num_vehicles=12,
    car_following_params=SumoCarFollowingParams(
        min_gap=1,
        decel=DAcc,
        accel=Acc,
        max_speed=Max_vel,
        s0=0
    ),
    length=4.45,
    color='green'
)

for i in range(NUM_AUTOMATED):
    vehicles.add(
        veh_id="rl_{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        initial_speed=0,
        num_vehicles=1,

        car_following_params=SumoCarFollowingParams(
            speed_mode=0,
            min_gap=1,
            decel=DAcc,
            accel=Acc,
            max_speed=Max_vel,
        ),
        length=4.45
    )

flow_params = dict(
    exp_tag="MA_5LC",
    env_name=MADRingLCPOEnv,
    network=RingNetwork,

    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=True,
        seed=seed2,
    ),

    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=125,
        clip_actions=False,
        additional_params={
            "max_accel": 1,
            "max_decel": 1,
            "ring_length": 670,
            "lane_change_duration": 0,
            "target_velocity": 13.493,
            'sort_vehicles': False
        },
    ),
    net=NetParams(
        additional_params={
            "length": 670,
            "lanes": 5,
            "speed_limit": Max_vel,
            "resolution": 40,
        },
    ),

    veh=vehicles,
    initial=InitialConfig(
        spacing='custom',

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