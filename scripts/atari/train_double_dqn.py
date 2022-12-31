import os
import gym
import torch
import random
import numpy as np

from utils.replay_buffer import *
from utils.gym_wrappers import *
from agents.discrete.double_dqn import Agent
from easydict import EasyDict as edict
from configs.atari.double_dqn import default_cfgs


class Trainer(object):
    def __init__(self, cfgs: edict = {}):
        super(Trainer, self).__init__()
        self.cfgs = edict(default_cfgs.copy())
        self.cfgs.update(cfgs)

        np.random.seed(self.cfgs.seed)
        random.seed(self.cfgs.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(self.cfgs.logs.checkpoint.path) == False:
            os.makedirs(self.cfgs.logs.checkpoint.path)
        if os.path.exists(self.cfgs.logs.video.path) == False:
            os.makedirs(self.cfgs.logs.video.path)
        if os.path.exists(self.cfgs.logs.print.path) == False:
            os.makedirs(self.cfgs.logs.print.path)

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

        if self.cfgs.logs.checkpoint.load:
            filename = os.path.join(self.cfgs.logs.checkpoint.path, self.cfgs.logs.checkpoint.name)
            print("Loading a policy network from {}".format(filename))
            self.agent.policy_net.load_state_dict(torch.load(filename, map_location = self.device))
        
    def training(self):
        eps_timesteps = self.cfgs.training.eps_fraction * self.cfgs.training.num_steps
        rewards = [0.0]
        state = self.env.reset()
        for t in range(self.cfgs.training.num_steps):
            fraction = min(1.0, float(t) / eps_timesteps)
            eps_threshold = self.cfgs.training.eps_begin + fraction * (self.cfgs.training.eps_end - self.cfgs.training.eps_begin)
            sample = random.random()
            if sample > eps_threshold:
                action = self.agent.action(state)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            rewards[-1] += reward
            if done:
                state = self.env.reset()
                rewards.append(0)
            if t > self.cfgs.training.start_step and t % self.cfgs.training.train_opt_freq == 0:
                self.agent.optimize()
            if t > self.cfgs.training.start_step and t % self.cfgs.training.target_upd_freq == 0:
                self.agent.update_target_net()
            ep_num = len(rewards)
            if done and ep_num % self.cfgs.logs.print.freq == 0:
                mean_reward = round(np.mean(rewards[-101:-1]), 1)
                print("==> Steps: {}, Episodes: {}".format(t, ep_num))
                print("Mean 100 episode reward: {}".format(mean_reward))
                print("Exploring: {}%".format(int(100 * eps_threshold)))
                filename = os.path.join(self.cfgs.logs.checkpoint.path, self.cfgs.logs.checkpoint.name)
                torch.save(self.agent.policy_net.state_dict(), filename)
                np.savetxt(os.path.join(self.cfgs.logs.print.path, 'rewards.csv'), rewards, delimiter = ',', fmt = '%1.3f')