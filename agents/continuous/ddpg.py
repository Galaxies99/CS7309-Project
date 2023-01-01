import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict
from utils.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, s):
        return self.max_action * self.layers(s)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace = True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = self.layer1(s)
        x = self.layer2(torch.cat([x, a], 1))
        return x


class Agent(object):
    def __init__(
        self, 
        state_dim,
        action_dim,
        max_action, 
        replay_buffer: ReplayBuffer,
        learning_rate: float,
        batch_size: int,
        gamma: float,
        tau: float,
        opt: edict,
        device = torch.device("cpu"),
        **kwargs
    ):
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        self.actor_optimizer = getattr(torch.optim, opt.type)(
            self.actor.parameters(), 
            lr = self.learning_rate,
            **opt.params
        )
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_optimizer = getattr(torch.optim, opt.type)(
            self.critic.parameters(), 
            lr = self.learning_rate,
            **opt.params
        )

    def action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def optimize(self):
        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
