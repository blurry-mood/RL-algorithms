import sys
sys.path.insert(0, "/home/ayoub/Desktop/Projects/RL-algorithms")

import gym
import minihack
import rl_minihack
import torch
import numpy as np
from torch import nn

from rl_algorithms import ActorCritic

class MyAgent(ActorCritic):

    def decode_state(self, state):
        s = torch.from_numpy(np.array(tuple(map(tuple, state['chars_crop'])))).flatten().unsqueeze(0).float()
        return s

if __name__ == '__main__':
    ALPHA = 1e-3
    GAMMA = 0.95
    EPS = 1e-1
    ITERS = 10

    p_net = nn.Linear(25, 8)
    v_net = nn.Linear(25, 1)

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        max_episode_steps=10_000,
        obs_crop_h=5,
        obs_crop_w=5,   
        observation_keys=('chars_crop', 'pixel')
    )

    actor_critic = MyAgent(p_net, v_net, actions=list(range(env.action_space.n)), alpha_policy=ALPHA, alpha_q=ALPHA, gamma=GAMMA, eps=EPS)
    actor_critic.load('actor_critic')

    for i in range(ITERS):
        state = env.reset()
        env.render(state)
        n = 0
        done = False
        actor_critic.start_episode()
        while not done:
            n += 1
            action = actor_critic.take_action(state)
            state, reward, done, info = env.step(action)
            l1, l2 = actor_critic.update(state, reward)
            env.render(state)

        actor_critic.save('actor_critic')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()

