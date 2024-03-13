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

from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

from Algos.PLAS import LatentPerturbation
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
parser.add_argument('--buffers', type=int, default=1e6)
parser.add_argument('--horizon', type=int, default=3000)
parser.add_argument('--max-ts', type=int, default=int(1e6))

# Offline RL parameter
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--itr', type=int, default=15000)
parser.add_argument('--num-evaluations', type=int, default=5)

# Actor critic parameter
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--batch', type=int, default=64)  # batch size to update
parser.add_argument('--discount', type=float, default=0.99,)  # discounted factor
parser.add_argument('--target-update-interval', type=int, default=2)
parser.add_argument('--l2_rate', type=float, default=1e-3,)
parser.add_argument('--eps', type=float, default=1e-08)
parser.add_argument('--actor_lr', type=float, default=1e-04,)
parser.add_argument('--critic_lr', type=float, default=1e-04,)

# PLAS algorithm parameter
parser.add_argument('--phi', type=float, default=0.05,)
parser.add_argument('--lmbda', type=float, default=0.75,)
parser.add_argument('--vae_lr', type=float, default=1e-04,)
parser.add_argument('--vae-epochs', type=int, default=30)
parser.add_argument('--max-latent-action', type=int, default=2)

parser.add_argument('--project', default='AD4RL')
parser.add_argument('--group', default='AD4RL-FLOW')
parser.add_argument('--name', default='PLAS')


args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    setting = f"{args.dataset}_{args.seed}_{args.batch}"
    replay_buffer.load(f"./buffers/{buffer_name}")

    # Initialize and load policy and Q_net
    policy = LatentPerturbation(args, num_inputs, num_actions, max_action)
    done = True
    # Vae Training
    print('-------------VAE--------------')
    for ep in tqdm(range(args.vae_epochs)):
        vae_tot, recon_tot, KL_tot = policy.train_vae(replay_buffer, 1000000)

        print('----------------------------------------------------------------------------------------')
        print('# epoch: {} # vae_loss: {} # recon_loss: {} # KL_loss: {}'.format(ep, vae_tot,
                                                                                                    recon_tot, KL_tot))
        print('----------------------------------------------------------------------------------------')

    print('-------------Done VAE--------------')
    print()
    print('-------------Model--------------')

    config = vars(args)
    wandb_init(config)

    # Model Training
    done = True
    reward_list = []
    for it in tqdm(range(args.epochs * args.itr)):
        policy.train(replay_buffer)
        if (it + 1) % args.itr == 0:
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
                    if args.render:
                        env.render()
                    action = policy.select_action(list(state.values()))
                    action = {list(state.keys())[0]: action}
                    episode_vel.append(list(state.values())[0])
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
            print('# itr: {} # avg.reward: {} # cor.reward: {}'.format(it, eval_reward, correction_reward))
            print('# velocity list: {} over {} evaluations'.format(velocity, args.num_evaluations))
            print('# average episode len: {}'.format(eval_timestep))
            print('----------------------------------------------------------------------------------------')

            wandb.log({"vanilla_reward": eval_reward, "coorection_reward": correction_reward, "timesteps": eval_timestep},
                      step=it)

            reward_list.append(eval_reward)
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
    seed_list = [5, 6, 7]
    env_list = ['highway-humanlike', 'highway-ngsim']
    for i in env_list:
        args.dataset = i
        print(f'--------------------Dataset: {args.dataset}--------------------')
        for j in seed_list:
            args.seed = j

            state_dim = 19
            action_dim = 2
            buffer_name = f"{args.dataset}"
            set_seed(args.seed)

            args.name = f"{args.name}-Seed{args.seed}-{args.dataset}-{str(uuid.uuid4())[:8]}"

            buffer_size = len(np.load(f"./buffers/{buffer_name}/reward.npy"))
            replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size)

            print('-----------------------------------------------------')
            main(args, replay_buffer)
            wandb.finish()
            args.name = 'PLAS'
            print('-------------------DONE OFFLINE RL-------------------')

            import ray
            ray.shutdown()