import math
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaNet(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 n_critic):
        super().__init__()
        self.InSize = in_feat
        self.OutSize = out_feat
        self.EnsembleSize = n_critic

        self.weight = nn.Parameter(torch.empty(self.EnsembleSize, self.InSize, self.OutSize))
        self.bias = nn.Parameter(torch.empty(self.EnsembleSize, 1, self.OutSize))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.EnsembleSize):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight + self.bias

class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,):
        super().__init__()

        self.action_dim = action_dim
        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)

        self.mu = nn.Linear(32, self.action_dim)
        self.log_std = nn.Linear(32, self.action_dim)

        torch.nn.init.constant_(self.l1.bias, 0.1)
        torch.nn.init.constant_(self.l2.bias, 0.1)
        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)

    def forward(self, state, deterministic, need_log_prob):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        mu, log_std = self.mu(out), self.log_std(out)

        log_std = torch.clip(log_std, -5, 2)
        policy_dist = torch.distributions.Normal(mu, torch.exp(log_std))

        # Deterministic Policy or Stochastic Policy
        if deterministic:
            action = mu
        else:
            # Sampling with Re-parameterized Trick
            action = policy_dist.rsample()

        act, log_prob = torch.tanh(action), None
        eps = 1e-06
        if need_log_prob:
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - act.pow(2) + eps).sum(axis=-1)

        return act * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state, device, eval):
        if eval:
            deter = True
        else:
            deter = False

        state = torch.tensor(state, device=device, dtype=torch.float32)
        return self.forward(state, deter)[0].cpu().numpy()

class Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 n_critic,):
        super().__init__()

        self.CriticNum = n_critic
        self.critic = nn.Sequential(
            VanillaNet(state_dim + action_dim, 64, self.CriticNum),
            nn.ReLU(),
            VanillaNet(64, 32, self.CriticNum),
            nn.ReLU(),
            VanillaNet(32, 1, self.CriticNum),
        )

        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-03, 3e-03)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-03, 3e-03)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        if sa.dim() != 3:
            assert sa.dim() == 2
            sa = sa.unsqueeze(0).repeat_interleave(
                self.CriticNum, dim=0
            )
        assert sa.dim() == 3
        assert sa.shape[0] == self.CriticNum

        return self.critic(sa).squeeze(-1)

class EDAC:
    def __init__(self,
                 args,
                 state_dim,
                 action_dim,
                 max_action,
                 device):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(state_dim, action_dim, args.n_critic)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.batch = args.batch
        self.discount = args.discount
        self.tau = args.tau
        self.eta = args.eta
        self.max_action = max_action

        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor([0.], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.critic_update_num = 0
        self.CriticNum = args.n_critic

        self.target_update_interval = args.target_update_interval
        self.it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, _ = self.actor.forward(state, deterministic=True, need_log_prob=False)
        return    action.cpu().data.numpy().flatten()

    def _alpha_loss(self, state):
        with torch.no_grad():
            action, action_log_prob = self.actor.forward(state, False, need_log_prob=True)

        return (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

    def _actor_loss(self, state):
        action, action_log_prob = self.actor.forward(state, False, need_log_prob=True)
        q_value_dist = self.critic.forward(state, action)
        assert q_value_dist.shape[0] == self.CriticNum
        q_value_min = q_value_dist.min(0).values
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()

        assert action_log_prob.shape == q_value_min.shape
        loss = (self.alpha * action_log_prob - q_value_min).mean()

        return loss, batch_entropy, q_value_std

    def _diversity_loss(self, state, action):
        state = state.unsqueeze(0).repeat_interleave(self.CriticNum, dim=0)
        action = (action.unsqueeze(0).repeat_interleave(self.CriticNum, dim=0).requires_grad_(True))

        q_ens = self.critic.forward(state, action)
        q_action_grad = torch.autograd.grad(q_ens.sum(), action, retain_graph=True, create_graph=True)[0]
        q_action_grad = q_action_grad / (torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10)
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = (torch.eye(self.CriticNum, device=self.device).unsqueeze(0).repeat(q_action_grad.shape[0], 1, 1))

        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()

        return grad_loss / (self.CriticNum - 1)

    def _critic_loss(self, state, action, next_state, reward, not_done):
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor.forward(next_state, False, True)
            tq = self.target_critic.forward(next_state, next_action).min(0).values
            tq = tq - self.alpha * next_action_log_prob

            tq = reward + self.discount * not_done * tq.unsqueeze(-1)

        q = self.critic.forward(state, action)
        loss = ((q - tq.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        div_loss = self._diversity_loss(state, action)

        return loss + self.eta * div_loss

    def train(self, replay_buffer, iterations):
        alpha_tot_loss, actor_tot_loss, critic_tot_loss = 0, 0, 0
        iteration = iterations // self.batch
        for it in tqdm(range(iteration)):
            s, a, ns, r, d = replay_buffer.sample(self.batch)

            alpha_loss = self._alpha_loss(s)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            actor_loss, actor_batch_entropy, q_policy_std = self._actor_loss(s)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.alpha_optim.step()

            critic_loss = self._critic_loss(s, a, ns, r, d)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            actor_tot_loss += actor_loss
            alpha_tot_loss += alpha_loss
            critic_tot_loss += critic_loss

            self.it += 1.

            if self.it % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                # for logging, Q-ensemble std estimate with the random actions:
                # a ~ U[-max_action, max_action]
                max_action = self.max_action
                rand_a = -max_action + 2 * max_action * torch.rand_like(a)
                q_random_std = self.critic.forward(s, rand_a).std(0).mean().item()

        actor_tot_loss /= iterations
        alpha_tot_loss /= iterations
        critic_tot_loss /= iterations

        return   actor_tot_loss.detach().cpu().item(), alpha_tot_loss.detach().cpu().item(), \
            critic_tot_loss.detach().cpu().item()


    def save(self, filename, ep):
        torch.save(self.target_critic.state_dict(), filename + f'_{ep}' + "_target_critic")
        torch.save(self.critic.state_dict(), filename + f'_{ep}' + "_critic")
        torch.save(self.actor.state_dict(), filename + f'_{ep}' + "_actor")

    def load(self, filename, ep):
        self.critic.load_state_dict(torch.load(filename + f'_{ep}' + "_critic"))
        self.target_critic.load_state_dict(torch.load(filename + f'_{ep}' + 'target_critic'))
        self.actor.load_state_dict(torch.load(filename + f'_{ep}' + "_actor"))