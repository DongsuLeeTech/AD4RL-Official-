import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Algos.model import *
from tqdm import tqdm
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self._mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = -10.
        self._max_log_std = 2.
        self._min_action = -max_action
        self._max_action = max_action

    def _get_policy(self, state):
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state, action):
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state):
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state, device, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        policy = self._get_policy(state)
        if deterministic:
            action = policy.mean
        else:
            action = policy.sample()
        return action.cpu().data.numpy()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, action):
        return self._mlp(torch.cat([state, action], dim=-1))

class AWAC(object):
    def __init__(self, args, state_dim, action_dim, max_action):
        self.device = args.device
        self.state_dim = state_dim
        self.act_dim = action_dim

        self.q1 = Critic(state_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=args.q_lr)

        self.q2 = Critic(state_dim, action_dim).to(self.device)
        self.q2_target = copy.deepcopy(self.q2)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=args.q_lr)

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.discount = args.discount
        self.tau = args.tau
        self.batch = args.batch
        self.awac_lambda = args.awac_lambda
        self.exp_adv_max = args.exp_adv_max

        self.it = 0
        self.targ_update_freq = args.targ_update_freq

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor.act(state, self.device, deterministic=True).flatten()

    def train(self, replay_buffer, iterations):
        q_tot_loss, actor_tot_loss = 0, 0

        for it in tqdm(range(iterations// self.batch)):
            # Sample replay buffers / batch
            s, a, ns, r, d = replay_buffer.sample(self.batch)

            with torch.no_grad():
                next_pol_act, _ = self.actor(ns)
                qft = torch.min(self.q1_target(ns, next_pol_act), self.q2_target(ns,next_pol_act))
                qft = r + self.discount * (1 - d) * qft

            q1 = self.q1(s, a)
            q2 = self.q2(s, a)
            q1_loss = F.mse_loss(q1, qft)
            q2_loss = F.mse_loss(q2, qft)
            critic_loss = q1_loss + q2_loss

            with torch.no_grad():
                pol_act, _ = self.actor(s)
                v = torch.min(self.q1(s, pol_act), self.q2(s, pol_act))
                q = torch.min(self.q1(s, a), self.q2(s, a))
                adv = q - v
                weights = torch.clamp_max(torch.exp(adv / self.awac_lambda), self.exp_adv_max)

            act_log_prob = self.actor.log_prob(s, a)
            actor_loss = (-act_log_prob * weights).mean()

            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            critic_loss.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.it += 1

            if self.it % self.targ_update_freq == 0:
                for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                    target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)
                for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                    target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)

            q_tot_loss += critic_loss.item()
            actor_tot_loss += actor_loss.item()

        q_tot_loss /= iterations
        actor_tot_loss /= iterations

        return q_tot_loss, actor_tot_loss

    def save(self, filename, ep):
        torch.save(self.actor.state_dict(), filename + f'_{ep}' + '_actor')
        torch.save(self.q1.state_dict(), filename + f'_{ep}' + '_q1')
        torch.save(self.q1_target.state_dict(), filename + f'_{ep}' + '_q1_target')
        torch.save(self.q2.state_dict(), filename + f'_{ep}' + '_q2')
        torch.save(self.q2_target.state_dict(), filename + f'_{ep}' + '_q2_target')

    def load(self, filename, ep):
        self.actor.load_state_dict(torch.load(filename + f'_{ep}' + '_actor'))
        self.q1.load_state_dict(torch.load(filename + f'_{ep}' + '_q1'))
        self.q1_target.load_state_dict(torch.load(filename + f'_{ep}' + '_q1_target'))
        self.q2.load_state_dict(torch.load(filename + f'_{ep}' + '_q2'))
        self.q2_target.load_state_dict(torch.load(filename + f'_{ep}' + '_q2_target'))
