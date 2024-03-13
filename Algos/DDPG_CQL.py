import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.exploration import OUNoise
from Algos.model import *
# from Algos.SAC import *
from tqdm import tqdm
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))

        return self.l3(q)


class DDPGCQL(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_dim,
                 max_action,
                 action_space,
                 discount=0.99,
                 tau=0.005,
                 ):

        self.device = args.device
        # self.evaluate = args.evaluate
        self.batch = args.batch
        self.target_update_interval = args.target_update_interval

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.exp = OUNoise(action_dim)

        self.it = 0

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def select_exp_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        noise = self.exp.noise()
        return self.actor(obs).cpu().data.numpy().flatten() + 0.05 * noise

    def conservative_q_loss(self, obs, action):
        pol_a = self.actor.forward(obs)
        pol_q = self.critic.forward(obs, pol_a)
        beh_q = self.critic.forward(obs, action)
        return pol_q.mean() - beh_q.mean()


    def actor_loss(self, obs):
        # consider policy action
        action = self.actor.forward(obs)
        return -self.critic.forward(obs, action)


    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        cql_loss = self.conservative_q_loss(state, action)

        with torch.no_grad():
            next_state_action = self.actor_target.forward(next_state)
            qft_next = self.critic_target.forward(next_state, next_state_action)
            next_q = reward + (1 - not_done) * self.discount * qft_next

        qf = self.critic.forward(state, action)

        critic_loss = cql_loss + 0.5 * F.mse_loss(qf, next_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        actor_loss = self.actor_loss(state).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        self.it += 1

    def fine_tune(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch)

        with torch.no_grad():
            next_state_action = self.actor_target.forward(next_state)
            qft_next = self.critic_target.forward(next_state, next_state_action)
            next_q = reward + (1 - not_done) * self.discount * qft_next

        qf = self.critic.forward(state, action)
        critic_loss = F.mse_loss(qf, next_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        actor_loss = self.actor_loss(state).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.it += 1.

        if self.it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return critic_loss.detach().cpu().item(), actor_loss.detach().cpu().item()


    def save(self, filename, ep):
        torch.save(self.actor.state_dict(), filename + f'_{ep}' + '_actor')
        torch.save(self.actor_target.state_dict(), filename + f'_{ep}' + '_actor_target')
        torch.save(self.critic.state_dict(), filename + f'_{ep}' + '_critic')
        torch.save(self.critic_target.state_dict(), filename + f'_{ep}' + '_critic_target')

    def load(self, filename, ep):
        self.actor.load_state_dict(torch.load(filename + f'_{ep}' + '_actor'))
        self.actor_target.load_state_dict(torch.load(filename + f'_{ep}' + '_actor_target'))
        self.critic.load_state_dict(torch.load(filename + f'_{ep}' + '_critic'))
        self.critic_target.load_state_dict(torch.load(filename + f'_{ep}' + '_critic_target'))

def soft_update(target_net, net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)