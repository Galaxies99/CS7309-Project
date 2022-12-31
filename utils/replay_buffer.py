import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        super(ReplayBuffer, self).__init__()
        self.record = []
        self.size = size
        self.next_idx = 0
    
    def __len__(self):
        return len(self.record)
    
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if self.next_idx >= len(self.record):
            self.record.append(data)
        else:
            self.record[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.size
    
    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.record) - 1, size = batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done = self.record[i]
            states.append(np.array(state, copy = False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy = False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
