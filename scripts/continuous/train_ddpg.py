import os
import gym
import torch
import random
import numpy as np

from utils.gym_wrappers import *
from utils.replay_buffer import *
from utils.configs import merge_cfgs
from easydict import EasyDict as edict
from agents.continuous.ddpg import Agent
from configs.continuous.ddpg import default_cfgs


class Trainer(object):
    def __init__(self, cfgs: edict = {}):
        super(Trainer, self).__init__()
        self.cfgs = merge_cfgs(default_cfgs, cfgs)

        np.random.seed(self.cfgs.seed)
        random.seed(self.cfgs.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(self.cfgs.logs.checkpoint.path) == False:
            os.makedirs(self.cfgs.logs.checkpoint.path)
        if os.path.exists(self.cfgs.logs.print.path) == False:
            os.makedirs(self.cfgs.logs.print.path)

        self.env = gym.make(self.cfgs.environment)
        self.env.seed(self.cfgs.seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.replay_buffer = ReplayBuffer(self.cfgs.agent.replay_buffer_size)
        self.agent = Agent(
            self.state_dim,
            self.action_dim,
            self.max_action,
            self.replay_buffer,
            learning_rate = self.cfgs.agent.learning_rate,
            batch_size = self.cfgs.agent.batch_size,
            gamma = self.cfgs.agent.gamma,
            tau = self.cfgs.agent.tau,
            opt = self.cfgs.agent.optimizer,
            device = self.device
        )

        if self.cfgs.logs.checkpoint.load:
            filename = os.path.join(self.cfgs.logs.checkpoint.path, self.cfgs.logs.checkpoint.name)
            print("Loading a policy network from {}".format(filename))
            self.agent.load_state_dict(torch.load(filename, map_location = self.device))
        
    def training(self):
        rewards = [0.0]
        state = self.env.reset()
        for t in range(self.cfgs.training.num_steps):
            noisy = np.random.normal(0, self.max_action * self.cfgs.training.expl_noise, size = self.action_dim)
            action = (self.agent.action(state) + noisy).clip(-self.max_action, self.max_action)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            rewards[-1] += reward
            if done:
                state = self.env.reset()
                rewards.append(0)
            if t > self.cfgs.training.start_step:
                self.agent.optimize()
            ep_num = len(rewards)
            if done and ep_num % self.cfgs.logs.print.freq == 0:
                mean_reward = round(np.mean(rewards[-101:-1]), 1)
                print("==> Steps: {}, Episodes: {}".format(t, ep_num))
                print("Mean 100 episode reward: {}".format(mean_reward))
                filename = os.path.join(self.cfgs.logs.checkpoint.path, self.cfgs.logs.checkpoint.name)
                torch.save(self.agent.get_state_dict(), filename)
                np.savetxt(os.path.join(self.cfgs.logs.print.path, 'rewards.csv'), rewards, delimiter = ',', fmt = '%1.3f')