import collections
import numpy as np
import torch


class ExperienceReplay:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        try:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=False)
        except:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=True)

        states, actions, rewards, next_states, next_actions = map(lambda x: torch.from_numpy(
            np.array(x, dtype='float32')), zip(*[self.buffer[idx] for idx in indices]))
        return states, actions.long(), rewards, next_states, next_actions.long()
