import numpy as np
from tqdm import tqdm

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

'''
implementation: https://github.com/Wenxuan-Zhou/PLAS
paper: https://arxiv.org/abs/2011.07213
'''

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class ActorPerturbation(nn.Module):
    def __init__(self, state_dim, action_dim, latent_action_dim, max_action, max_latent_action=2, phi=0.05):
        super(ActorPerturbation, self).__init__()

        self.hidden_size = (400, 300, 400, 300)

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, latent_action_dim)

        self.l4 = nn.Linear(state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 32)
        self.l6 = nn.Linear(32, action_dim)

        self.max_latent_action = max_latent_action
        self.max_action = max_action
        self.phi = phi

    def forward(self, state, decoder):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        latent_action = self.max_latent_action * torch.tanh(self.l3(a))

        mid_action = decoder(state, z=latent_action)

        a = F.relu(self.l4(torch.cat([state, mid_action], 1)))
        a = F.relu(self.l5(a))
        a = self.phi * torch.tanh(self.l6(a))
        final_action = (a + mid_action).clamp(-self.max_action, self.max_action)
        return latent_action, mid_action, final_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device, hidden_size=100):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: return a
        return self.max_action * torch.tanh(a)


class LatentPerturbation(object):
    def __init__(self, args, state_dim, action_dim, max_action):
        self.device = args.device
        self.latent_dim = action_dim * 2

        self.actor = ActorPerturbation(state_dim, action_dim, self.latent_dim, max_action,
                                       max_latent_action=args.max_latent_action, phi=args.phi).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.vae = VAE(state_dim, action_dim, self.latent_dim, max_action, args.device).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.vae_lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = args.discount
        self.tau = args.tau
        self.lmbda = args.lmbda
        self.batch = args.batch

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            _, _, action = self.actor(state, self.vae.decode)
        return action.cpu().data.numpy().flatten()

    def vae_loss(self, state, action):
        state = state.to(self.device)
        action = action.to(self.device)
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        return vae_loss, recon_loss, KL_loss

    def train_vae(self, replay_buffer, iterations):
        tot_vae_loss, tot_recon_loss, tot_KL_loss = 0, 0, 0
        iteration = iterations // self.batch

        for it in range(iteration):
            # Sample replay buffers / batch
            state, action, _, _, _ = replay_buffer.sample(self.batch)
            vae_loss, recon_loss, KL_loss = self.vae_loss(state, action)

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            tot_vae_loss += vae_loss
            tot_recon_loss += recon_loss
            tot_KL_loss += KL_loss

        tot_vae_loss /= iteration
        tot_recon_loss /= iteration
        tot_KL_loss /= iteration

        return tot_vae_loss.detach().cpu().numpy(), tot_recon_loss.detach().cpu().numpy(), \
            tot_KL_loss.detach().cpu().numpy()

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        # Critic Training
        with torch.no_grad():
            _, _, next_action = self.actor_target(next_state, self.vae.decode)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + \
                       (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Training
        latent_actions, mid_actions, actions = self.actor(state, self.vae.decode)
        actor_loss = -self.critic.q1(state, actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, ep):
        torch.save(self.actor.state_dict(), filename + f'_{ep}' + '_actor')
        torch.save(self.actor_target.state_dict(), filename + f'_{ep}' + '_actor_target')
        torch.save(self.critic.state_dict(), filename + f'_{ep}' + '_critic')
        torch.save(self.critic_target.state_dict(), filename + f'_{ep}' + '_critic_target')

    def save_vae(self, filename, ep):
        torch.save(self.vae.state_dict(), filename + f'_{ep}' + '_vae')

    def load(self, filename, ep):
        self.actor.load_state_dict(torch.load(filename + f'_{ep}' + '_actor'))
        self.actor_target.load_state_dict(torch.load(filename + f'_{ep}' + '_actor_target'))
        self.critic.load_state_dict(torch.load(filename + f'_{ep}' + '_critic'))
        self.critic_target.load_state_dict(torch.load(filename + f'_{ep}' + '_critic_target'))

    def load_vae(self, filename, ep):
        self.vae.load_state_dict(torch.load(filename + f'_{ep}' + '_vae'))
