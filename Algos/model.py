import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weight_init(p):
    if isinstance(p, nn.Linear):
        torch.nn.init.xavier_uniform(p.weight, gain=1)
        torch.nn.init.constant(p.bias, 0)

# Stochastic
class StochasticActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,
                 action_space=None):
        super(StochasticActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)

        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)

        self.apply(weight_init)
        self.max_action = max_action
        self.epsilon = 1e-06

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)

        print(self.action_scale, self.action_bias)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = self.mean(a)
        log_std = self.log_std(a)
        log_std = torch.clamp(log_std, min=-self.max_action, max=self.max_action)
        return mean, log_std

    def sample(self, state):
        # Policy Distribution; torch.distribution.Normal(loc, scale)
        # loc (float or Tensor) – mean of the distribution (often referred to as mu)
        # scale (float or Tensor) – standard deviation of the distribution (often referred to as sigma)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # rsample(): Reparameterization trick (mean + std * N(0,1)) for backpropagation
        x_t = normal.rsample()
        log_prob = normal.log_prob(x_t)

        # y_t: pi(s); tanh func() as normalize [-1., 1.]
        # action_scale and action_bias moves the action from [-1., 1.] to [action_space.high, action_space.low]
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # log_prob: log pi(s);
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        mean = self.consider_discrete(mean)
        action = self.consider_discrete(action)
        return action, log_prob, mean

    def consider_discrete(self, act):
        action = act
        if action[0][1] > 1 / 3:
            action[0][1] = 1
        elif action[0][1] < -1 / 3:
            action[0][1] = -1
        else:
            action[0][1] = 0

        return action

    def sample_eval(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        mean = self.consider_discrete(mean)

        return mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)

        return super(StochasticActor, self).to(device)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,
                 action_space=None):
        super(DeterministicActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)

        self.mean = nn.Linear(32, action_dim)
        self.noise = torch.Tensor(action_dim)

        self.apply(weight_init)
        self.max_action = max_action
        self.epsilon = 1e-06

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = torch.tanh(self.mean(a)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise

        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicActor, self).to(device)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)
        self.apply(weight_init)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.apply(weight_init)

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        pred = torch.tanh(self.l3(out))
        return pred