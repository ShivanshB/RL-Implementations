import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_samples):
        self.buffer = deque(maxlen=max_samples)
        self.max_samples = max_samples

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)