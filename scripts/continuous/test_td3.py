import os
import gym
import torch
import random
import numpy as np

from tqdm import tqdm
from utils.gym_wrappers import *
from utils.replay_buffer import *
from utils.configs import merge_cfgs
from easydict import EasyDict as edict
from agents.continuous.td3 import Agent
from configs.continuous.td3 import default_cfgs


class Tester(object):
    def __init__(self, cfgs: edict = {}):
        super(Tester, self).__init__()
        self.cfgs = merge_cfgs(default_cfgs, cfgs)

        np.random.seed(self.cfgs.seed + np.random.randint(100))
        random.seed(self.cfgs.seed + np.random.randint(100))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert os.path.exists(self.cfgs.logs.checkpoint.path)

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
            noise = self.cfgs.agent.noise,
            noise_clip = self.cfgs.agent.noise_clip,
            policy_freq = self.cfgs.agent.policy_freq,
            device = self.device
        )

        filename = os.path.join(self.cfgs.logs.checkpoint.path, self.cfgs.logs.checkpoint.name)
        print("Loading a policy network from {}".format(filename))
        self.agent.load_state_dict(torch.load(filename, map_location = self.device))
    
    def testing(self, episode_num = 100):
        rewards = []
        with tqdm(range(episode_num)) as pbar:
            for _ in pbar:
                state = self.env.reset()
                rewards.append(0.0)
                done = False
                while not done:
                    action = self.agent.action(state).clip(-self.max_action, self.max_action)
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    rewards[-1] += reward
                pbar.set_description('Reward: {}'.format(rewards[-1]))
        return np.array(rewards)