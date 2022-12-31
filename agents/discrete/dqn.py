import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from utils.replay_buffer import ReplayBuffer



class Model(nn.Module):
    def __init__(
        self, 
        observation_space: spaces.Box, 
        action_space: spaces.Discrete,
        **kwargs
    ):
        super(Model, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = observation_space.shape[0], out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(inplace = True),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(in_features = 64 * 7 * 7, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 512, out_features = action_space.n)
        )
    
    def forward(self, x):
        return self.layers(x)



class Agent(object):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        learning_rate: float,
        batch_size: int,
        gamma: float,
        opt: edict,
        device = torch.device("cpu"),
        **kwargs
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.policy_net = Model(self.observation_space, self.action_space).to(self.device)
        self.target_net = Model(self.observation_space, self.action_space).to(self.device)
        self.update_target_net()
        self.target_net.eval()

        self.optimizer = getattr(torch.optim, opt.type)(
            self.policy_net.parameters(), 
            lr = self.learning_rate,
            **opt.params
        )
    
    def optimize(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(np.array(states) / 255.0).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states) / 255.0).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        with torch.no_grad():
            next_Q = self.target_net(next_states)
            max_next_Q, _ = next_Q.max(1)
            target_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        input_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.smooth_l1_loss(input_Q, target_Q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del states, next_states
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def action(self, state):
        state = torch.from_numpy(np.array(state) / 255.0).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            Q = self.policy_net(state)
            _, action = Q.max(1)
        return action.item()
