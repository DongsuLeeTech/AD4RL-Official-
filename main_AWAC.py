from typing import Any, Dict, List, Optional, Tuple, Union

import os
import sys
import gym
import json
import argparse
import numpy as np
from copy import deepcopy

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env
import getpass

from Algos.AWAC import AWAC
from Utils.utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parse argument used when running a Flow simulation.",
    epilog="python simulate.py EXP_CONFIG")
# required input parameters
parser.add_argument(
    'exp_config', type=str,
)  # Name of the experiment configuration file
parser.add_argument(  # for rllib
    '--algorithm', type=str, default="PPO",
)  # choose algorithm in order to use
parser.add_argument(
    '--num_cpus', type=int, default=1,
)  # How many CPUs to use
parser.add_argument(  # batch size
    '--rollout_size', type=int, default=100,
)  # How many steps are in a training batch.
parser.add_argument(
    '--checkpoint_path', type=str, default=None,
)  # Directory with checkpoint to restore training from.
parser.add_argument(
    '--no_render',
    action='store_true',
)  # Specifies whether to run the simulation during runtime.


# network and dataset setting
parser.add_argument('--seed', type=int, default=0,)  # random seed
parser.add_argument('--dataset', type=str, default=None)  # path to datset
parser.add_argument('--load_model', type=str, default=None,)  # path to load the saved model
parser.add_argument('--logdir', type=str, default='./results/',)  # tensorboardx logs directory

# OfflineRL parameter
parser.add_argument('--buffer', type=int, default=1e6)
parser.add_argument('--horizon', type=int, default=3000)
parser.add_argument('--max-ts', type=int, default=int(1e6))
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--itr', type=int, default=1000000)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--discount', type=float, default=0.99,)
parser.add_argument('--targ-update-freq', type=float, default=2)
parser.add_argument('--num-evaluations', type=int, default=10)

# AWAC algorithm parameter
parser.add_argument('--actor-lr', type=float, default=3e-04,)
parser.add_argument('--q-lr', type=float, default=3e-04,)
parser.add_argument('--awac-lambda', type=float, default=0.3333)
parser.add_argument('--exp-adv-max', type=float, default=100.0)

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)

def main(args, log_dir, replay_buffer):
    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[args.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[args.exp_config])

    # rl part
    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
        multiagent = False
    elif hasattr(module_ma, args.exp_config):
        submodule = getattr(module_ma, args.exp_config)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    flow_params = submodule.flow_params

    import ray
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    alg_run = "PPO"
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    ray.init(num_cpus=16, object_store_memory=200 * 1024 * 1024)

    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    agent = agent_cls(env=gym_name, config=config)

    # warmup correction
    if args.exp_config == 'MA_4BL':
        warmup_ts = 900
    elif args.exp_config == 'MA_5LC':
        warmup_ts = 125
    elif args.exp_config == 'UnifiedRing':
        warmup_ts = 90

    #  Load the Environment and Random Seed
    env = gym.make(gym_name)

    # Setup Random Seed
    env_set_seed(env, args.seed)

    num_inputs = 19
    num_actions = 2
    max_action = 1.0
    print(env.action_space)
    print('state size:', num_inputs)
    print('action size:', num_actions)

    # load Human Driving Data (NGSIM)
    buffer_name = f"{args.dataset}"
    setting = f"{args.dataset}_{args.seed}_{args.batch}"
    replay_buffer.load(f"./buffers/{buffer_name}")

    # Initialize and load policy and Q_net
    policy = AWAC(args, num_inputs, num_actions, max_action)

    done = True

    reward_list = []
    for ep in tqdm(range(args.epochs)):

        q_loss, policy_loss = policy.train(replay_buffer, args.itr)

        evaluations = []
        velocity = []
        timesteps = []
        for _ in range(args.num_evaluations):

            env.seed(args.seed + 100)
            tot_reward = 0.
            state, done = env.reset(), False

            episode_vel = []
            ts = 0
            while ts <= args.max_ts:
                action = policy.select_action(list(state.values()))
                episode_vel.append(list(state.values())[0])
                action = {list(state.keys())[0]: action}
                next_state, reward, done, _ = env.step(action)

                tot_reward += list(reward.values())[0]

                if done['__all__']:
                    timesteps.append(env.unwrapped.k.vehicle.get_timestep(env.unwrapped.k.vehicle.get_ids()[1]) / 100)
                    break
                else:
                    pass

                state = next_state
            velocity.append(np.mean(episode_vel))
            evaluations.append(tot_reward)

        eval_reward = np.mean(evaluations)
        eval_timestep = np.mean(timesteps) - warmup_ts
        correction_reward = np.mean(np.array(evaluations) + np.array(timesteps) - warmup_ts)

        print('----------------------------------------------------------------------------------------')
        print('# epoch: {} # avg.reward: {} # cor.reward: {} # q_loss: {} # actor_loss: {}'.format(ep, eval_reward,
                                                                                                    correction_reward,
                                                                                                    q_loss, policy_loss))
        print('# velocity list: {} over {} evaluations'.format(velocity, args.num_evaluations))
        print('# average episode len: {}'.format(eval_timestep))
        print('----------------------------------------------------------------------------------------')

        summary.add_scalar('actor', policy_loss, ep)
        summary.add_scalar('critic', q_loss, ep)
        summary.add_scalar('avg_reward', eval_reward, ep)
        summary.add_scalar('correction_reward', correction_reward, ep)
        summary.add_scalar('avg_timestep', eval_timestep, ep)
        reward_list.append(eval_reward)

        policy.save(f'./{log_dir}/{setting}', ep)

    np.save(f'./{log_dir}/reward_{setting}', np.array(reward_list), allow_pickle=True)
    summary.close()


def save_checkpoint(state, filename):
    torch.save(state, filename)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


if __name__=="__main__":
    seed_list = [5, 6, 7]
    dataset_list = ['highway-final', 'highway-final-medium', 'highway-final-random',
                    'highway-medium', 'highway-random', 'highway-humanlike']

    for d in dataset_list:
        args.dataset = d
        for seed in seed_list:
            args.seed = seed
            print(f'--------------------Dataset: {args.dataset}--------------------')

            state_dim = 19
            action_dim = 2
            buffer_name = f"{args.dataset}"
            set_seed(args.seed)

            log_dir_name = str(args.dataset) + '_Seed' + str(args.seed) + '_Batch' + str(args.batch)

            from datetime import datetime
            date = datetime.today().strftime("[%Y|%m|%d|%H:%M:%S]")
            log_dir = args.logdir + date + log_dir_name + 'AWAC'
            os.mkdir(log_dir)

            # Tensorboard: Easy Visualization Tool in PyTorch
            summary = SummaryWriter(log_dir)
            filename = './' + log_dir + '/arguments.txt'

            f = open(filename, 'w')
            f.write(str(vars(args)))
            f.close()

            buffer_size = len(np.load(f"./buffers/{buffer_name}/reward.npy"))
            replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size)

            print('-----------------------------------------------------')
            main(args, log_dir, replay_buffer)
            print('-------------------DONE OFFLINE RL-------------------')

            import ray
            ray.shutdown()