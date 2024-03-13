from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import os
import gym
import numpy as np
import torch
import random
from tqdm.auto import tqdm, trange
import math

TensorBatch = List[torch.Tensor]
class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._final_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffers")

        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffers is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._final_states[:n_transitions] = self._to_tensor(data["final_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._returns[:n_transitions] = self._to_tensor(data["returns"][..., None])

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        final_states = self._final_states[indices]
        returns = self._returns[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones, final_states, returns]

def discounted_cumsum(x: np.ndarray, gamma: float, n: int = None) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    if n >= x.shape[0]:
        for t in reversed(range(x.shape[0] - 1)):
            cumsum[t] = x[t] + gamma * cumsum[t + 1]
    else:
        for t in range(0, x.shape[0]):
            n_step_return = 0.0
            for i in range(t, min(t + n, x.shape[0])):
                n_step_return += gamma**(i-t) * x[i]
            cumsum[t] = n_step_return
    return cumsum

def load_trajectories(dataset: dict, gamma: float = 0.99, n: int = None):
    traj, traj_len = [], []
    data_ = defaultdict(list)

    for i in trange(dataset['reward'].shape[0]):
        data_["observations"].append(dataset["state"][i])
        data_["actions"].append(dataset["action"][i])
        data_["next_observations"].append(dataset["next_state"][i])
        data_["rewards"].append(dataset["reward"][i])
        data_["dones"].append(dataset["done"][i])

        if n:
            if i + n >= dataset["reward"].shape[0]:
                data_["n_observations"].append(dataset["next_state"][-1])
            else:
                data_["n_observations"].append(dataset["state"][i + n])
        else:
            pass

        if 1 - dataset["done"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma, n=n
            )

            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    _states = []
    _actions = []
    _rewards = []
    _next_states = []
    _final_states = []
    _dones = []
    _returns = []

    for i in range(len(traj)):
        _states.append(traj[i]['observations'])
        _actions.append(traj[i]['actions'])
        _rewards.append(traj[i]['rewards'])
        _next_states.append(traj[i]['next_observations'])
        _final_states.append(traj[i]['n_observations'])
        _dones.append(traj[i]['dones'])
        _returns.append(traj[i]['returns'])

    return {'observations': np.concatenate((_states),axis=0),
        'actions': np.concatenate((_actions),axis=0),
        'next_observations': np.concatenate((_next_states),axis=0),
        'rewards': np.concatenate(_rewards),
        'terminals': np.concatenate(_dones),
        'final_observations': np.concatenate((_final_states),axis=0),
        'returns': np.concatenate(_returns)}

def raw_dataset_load(dataset: str):
    if dataset == 'replay':
        data_dict = {'state': np.load(f'./buffers/{dataset}/state.npy'), 'action': np.load(f'./buffers/{dataset}/action.npy'),
               'next_state': np.load(f'./buffers/{dataset}/next_state.npy'), 'reward': np.load(f'./buffers/{dataset}/reward.npy').squeeze(-1),
               'done': np.load(f'./buffers/{dataset}/done.npy').squeeze(-1)}
    else:
        data_dict = {'state': np.load(f'./buffers/{dataset}/state.npy'), 'action': np.load(f'./buffers/{dataset}/action.npy'),
               'next_state': np.load(f'./buffers/{dataset}/next_state.npy'), 'reward': np.load(f'./buffers/{dataset}/reward.npy').squeeze(-1),
               'done': np.load(f'./buffers/{dataset}/done.npy').squeeze(-1)}

    return data_dict


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def env_set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
