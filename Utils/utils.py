import os
import cv2
import gym
import numpy as np
import torch

from typing import cast


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}/state.npy", self.state[:self.size])
        np.save(f"{save_folder}/action.npy", self.action[:self.size])
        np.save(f"{save_folder}/next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}/reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}/not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}/ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}/reward.npy", allow_pickle=True)

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        print(np.load(f"{save_folder}/action.npy", allow_pickle=True).shape)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}/next_state.npy", allow_pickle=True)[:self.size]
        self.reward[:self.size] = reward_buffer.reshape(self.size, 1)[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}/done.npy", allow_pickle=True)[:self.size]

    def load4BC(self, save_folder, size=-1):
        buffer_size = np.load(f"{save_folder}/reward.npy", allow_pickle=True).shape[0]

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(buffer_size, size)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]

    def sample4BC(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
        )


import os
import cv2
import gym
import numpy as np
import torch

from typing import cast


class ReplayBuffer_RBVQ(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.Gt = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done, Gt):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.Gt[self.ptr] = Gt

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.Gt[ind]).to(self.device)
        )

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}/reward.npy", allow_pickle=True)

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        print(np.load(f"{save_folder}/action.npy", allow_pickle=True).shape)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}/next_state.npy", allow_pickle=True)[:self.size]
        # self.reward[:self.size] = reward_buffer.reshape(self.size, 1)[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}/done.npy", allow_pickle=True)[:self.size]
        self.Gt[:self.size] = np.load(f"{save_folder}/return.npy", allow_pickle=True)[:self.size]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def env_set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
