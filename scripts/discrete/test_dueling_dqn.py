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
from agents.discrete.dueling_dqn import Agent
from configs.discrete.dqn import default_cfgs


class Tester(object):
    def __init__(self, cfgs: edict = {}):
        super(Tester, self).__init__()
        self.cfgs = merge_cfgs(default_cfgs, cfgs)

        np.random.seed(self.cfgs.seed)
        random.seed(self.cfgs.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert os.path.exists(self.cfgs.logs.checkpoint.path)
        
        self.env = gym.make(self.cfgs.environment)
        self.env.seed(self.cfgs.seed)

        self.env = NoopResetEnv(self.env, noop_max = 30)
        self.env = MaxAndSkipEnv(self.env, skip = 4)
        self.env = EpisodicLifeEnv(self.env)
        self.env = FireResetEnv(self.env)
        self.env = WarpFrame(self.env)
        self.env = PyTorchFrame(self.env)
        self.env = ClipRewardEnv(self.env)
        self.env = FrameStack(self.env, 4)
        self.env = gym.wrappers.Monitor(
            self.env, 
            self.cfgs.logs.video.path, 
            video_callable = lambda episode_id: episode_id % self.cfgs.logs.video.freq == 0,
            force = True
        )
        self.replay_buffer = ReplayBuffer(self.cfgs.agent.replay_buffer_size)
        self.agent = Agent(
            self.env.observation_space,
            self.env.action_space,
            self.replay_buffer,
            learning_rate = self.cfgs.agent.learning_rate,
            batch_size = self.cfgs.agent.batch_size,
            gamma = self.cfgs.agent.gamma,
            opt = self.cfgs.agent.optimizer,
            device = self.device
        )

        filename = os.path.join(self.cfgs.logs.checkpoint.path, self.cfgs.logs.checkpoint.name)
        print("Loading a policy network from {}".format(filename))
        self.agent.policy_net.load_state_dict(torch.load(filename, map_location = self.device))
        
    def testing(self, episode_num = 20):
        rewards = []
        with tqdm(range(episode_num)) as pbar:
            for _ in pbar:
                state = self.env.reset()
                rewards.append(0.0)
                done = False
                while not done:
                    sample = random.random()
                    if sample > self.cfgs.training.eps_end:
                        action = self.agent.action(state)
                    else:
                        action = self.env.action_space.sample()
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    rewards[-1] += reward
                pbar.set_description('Reward: {}'.format(rewards[-1]))
        return np.array(rewards)
