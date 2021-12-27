import sys
sys.path.insert(0, "/home/ayoub/Desktop/Projects/RL-algorithms")

import gym
import minihack
import rl_minihack
import torch
import numpy as np
from torch import nn

from rl_algorithms import DSN

class MyAgent(DSN):

    def decode_state(self, state):
        s = torch.from_numpy(np.array(tuple(map(tuple, state['chars_crop'])))).flatten().unsqueeze(0).float()
        return s

if __name__ == '__main__':
    ALPHA = 1e-1
    GAMMA = 1.01
    EPS = 0 #5e-1
    ITERS = 1

    net = nn.Linear(25, 8)

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,   
        observation_keys=('chars_crop', 'pixel')
    )

    dsn = MyAgent(net, actions=list(range(env.action_space.n)), alpha=ALPHA, gamma=GAMMA, eps=EPS)
    dsn.load('dsn')

    for i in range(ITERS):
        state = env.reset()
        env.render(state, 1e-1)
        n = 0
        done = False
        while not done:
            n += 1
            action = dsn.take_action(state)
            state, reward, done, info = env.step(action)
            dsn.update(reward)
            env.render(state)

        dsn.save('dsn')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()

