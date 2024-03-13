import os
import sys
import gym
import json
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy

import torch
from tensorboardX import SummaryWriter

from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

from Algos.BCQ import BCQ
from Utils.utils import *
from tqdm import tqdm

import uuid
import wandb

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

# Fine tune parameter
parser.add_argument('--fine-tune', action='store_true')
parser.add_argument('--num', type=int)
parser.add_argument('--buffer', type=int, default=1e6)
parser.add_argument('--horizon', type=int, default=3000)
parser.add_argument('--max-ts', type=int, default=int(1e6))

# Offline RL parameter
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--num-evaluations', type=int, default=10)

# RL parameter
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--l2_rate', type=float, default=1e-3)
parser.add_argument('--discount', type=float, default=0.99)
parser.add_argument('--opt_eps', type=float, default=1e-08)

# BCQ algorithm parameter
parser.add_argument("--phi", type=float, default=0.05)
parser.add_argument("--vae_lr", type=float, default=1e-04)
parser.add_argument("--lmbda", type=float, default=0.75)
parser.add_argument('--actor_lr', type=float, default=1e-04,)
parser.add_argument('--critic_lr', type=float, default=1e-04,)

parser.add_argument('--project', default='AD4RL')
parser.add_argument('--group', default='AD4RL-FLOW-v2')
parser.add_argument('--name', default='BCQ')

args = parser.parse_args()
args.device = torch.device("cpu")
print(args.device)
args.render = not args.no_render

def main(args, replay_buffer):
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
    setting = f"{args.dataset}_{args.seed}_{args.batch_size}"
    replay_buffer.load(f"./buffers/{buffer_name}")

    # Initialize and load policy and Q_net
    policy = BCQ(args, num_inputs, num_actions, max_action, args.device)

    done = True

    reward_list = []
    for ep in tqdm(range(args.epochs)):

        vae_tot, qf_tot, actor_tot = policy.train(replay_buffer, 1000000)

        evaluations = []
        velocity = []

        accident_observation = []
        timesteps = []
        for _ in range(args.num_evaluations):

            env.seed(args.seed + 100)
            tot_reward = 0.
            state, done = env.reset(), False

            episode_vel = []
            ts = 0
            while ts <= args.max_ts:
                if args.render:
                    env.render()
                action = policy.select_action(list(state.values()))
                episode_vel.append(list(state.values())[0])
                action = {list(state.keys())[0]: action}
                next_state, reward, done, _ = env.step(action)

                # collect the observation before accident for Q value checking
                if ep >= args.epochs - 1:
                    if env.unwrapped.k.simulation.check_collision():
                        accident_observation.append(state)
                        print(next_state)
                        print(reward)

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
        print('# epoch: {} # avg.reward: {} # cor.reward: {} # qf_loss: {} # actor_loss: {} # vae_loss: {}'.format(ep, eval_reward,
                                                                                    correction_reward, qf_tot, actor_tot, vae_tot))
        print('# velocity list: {} over {} evaluations'.format(velocity, args.num_evaluations))
        print('# average episode len: {}'.format(eval_timestep))
        print('----------------------------------------------------------------------------------------')

        wandb.log(
            {"vanilla_reward": eval_reward, "coorection_reward": correction_reward, "timesteps": eval_timestep},
            step=it)

        # policy.save(f'./{log_dir}/{setting}', ep)

    # np.save(f'./{log_dir}/reward_{setting}', np.array(reward_list), allow_pickle=True)
    # summary.close()

def save_checkpoint(state, filename):
    torch.save(state, filename)

def wandb_init(config:dict) -> None:
    wandb.init(
        config=config,
        project=config['project'],
        group=config['group'],
        name=config['name'],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

if __name__=="__main__":
    seed_list = [0,1,2]
    dataset_list = ['highway-ngsim']

    for d in dataset_list:
        args.dataset = d
        for seed in seed_list:
            args.seed = seed

            print(f'--------------------Dataset: {args.dataset}--------------------')
            state_dim = 19
            action_dim = 2
            buffer_name = f"{args.dataset}"
            set_seed(args.seed)

            args.name = f"{args.name}-Seed{args.seed}-{args.dataset}-{str(uuid.uuid4())[:8]}"
            config = vars(args)
            wandb_init(config)

            buffer_size = len(np.load(f"./buffers/{buffer_name}/reward.npy"))
            replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size)

            print('-----------------------------------------------------')
            main(args, replay_buffer)
            wandb.finish()
            args.name = 'BC'
            print('-------------------DONE OFFLINE RL-------------------')

            import ray
            ray.shutdown()