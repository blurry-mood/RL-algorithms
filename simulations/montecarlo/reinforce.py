import sys

import torch
sys.path.insert(0, '/home/ayoub/Desktop/Projects/RL-algorithms/')

import gym
import rl_minihack

from rl_algorithms import Reinforce


def test_minihack():
    GAMMA = 0.9
    ALPHA = 2**-12
    EPS = 4e-1
    ITERS = 100

    class MyAgent(Reinforce):
        def decode_state(self, state):
            return torch.from_numpy(state['chars_crop']).flatten().unsqueeze(0).float()

    env = gym.make(
        id="minihack:MiniHack-Room-5x5-v0",
        max_episode_steps=100_000,
        obs_crop_h=3,
        obs_crop_w=3,
        observation_keys=('chars_crop', 'pixel')
    )

    reinforce = MyAgent(torch.nn.Linear(9, env.action_space.n), actions=list(
        range(env.action_space.n)), alpha=ALPHA, gamma=GAMMA, eps=EPS)
    reinforce.load('reinforce_minihack')

    for i in range(ITERS):
        state = env.reset()
        env.render(state)
        n = 0
        done = False
        while not done:
            n += 1
            action = reinforce.take_action(state)
            state, reward, done, info = env.step(action)
            if reward == 0:
                reward = -0.1
            reinforce.update(reward)
            env.render(state)
        
        reinforce.end_episode()
        reinforce.save('reinforce_minihack')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()

    

test_minihack()