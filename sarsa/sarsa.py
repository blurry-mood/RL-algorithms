import gym
import minihack
from nle import nethack

import rl_minihack

import numpy as np
import time

from os.path import join, split
_HERE = split(__file__)[0]

class Sarsa:
    """ Possible observations:
    
    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 'pixel'
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    _STATE = 'chars_crop'

    def __init__(self, actions, alpha, gamma, eps):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = {}
        self.sar = []   # stands for state-action-reward

    def _action_value(self, state, action):
        return self.q.get((state, action), 1e-3*np.random.randn())

    def _get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        qs = {(action, self._action_value(state=state, action=action)) for action in self.actions}
        action = max(qs, key = lambda x: x[1])
        return action[0]

    def update(self, reward, done:bool):
        """ Update state-action value of previous (state, action).
        
        - `reward`: reward received upon the transaction to `next_state` from previous state.
        - `done`: boolean specifying whether the episode ended.
        """
        self.sar[-1][-1] = reward
        if len(self.sar) < 2:
            return

        q = self._action_value(state=self.sar[0][0], action=self.sar[0][1])
        tmp = self.sar[0][2] - q
        tmp += self.gamma * self._action_value(state=self.sar[1][0], action=self.sar[1][1])
        self.q[(self.sar[0][0], self.sar[0][1])] = q + self.alpha * tmp

        del self.sar[0]

        if done:
            self.q[(self.sar[0][0], self.sar[0][1])] = q + self.alpha * (self.sar[0][2] - q)


    def take_action(self, current_state):
        """ Choose an eps-greedy action to be taken when current state is `current_state`. """
        current_state = tuple(map(tuple, current_state[self._STATE]))
        action = self._get_action(current_state, self.eps)
        self.sar.append([current_state, action, 0])
        return action

    def save(self, path):
        """ Load state-action value table in `path`.npy """

        np.save(join(_HERE, path + '.npy'), self.q)

    def load(self, path):
        """ Load state-action value table.
        
        If it doesn't exist, use random q-value.
        """

        try:
            self.q = np.load(join(_HERE, path + '.npy'), allow_pickle='TRUE').item()
        except:
            print("============> No saved learner is found under:", path)
            time.sleep(2)

if __name__ == '__main__':
    ALPHA = 1e-1
    GAMMA = 9e-1
    EPS = 5e-1
    ITERS = 2
    
    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,
        observation_keys=('chars_crop', 'pixel')
    )

    sarsa = Sarsa(list(range(env.action_space.n)), ALPHA, GAMMA, EPS)
    sarsa.load('sarsa')

    for i in range(ITERS):
        state = env.reset()
        env.render(state, 1e-1)
        n = 0
        done = False
        while not done:
            n += 1
            action = sarsa.take_action(state)
            state, reward, done, info = env.step(action)
            sarsa.update(reward, done)                
            env.render(state)

        print('>'*20, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()
    sarsa.save('sarsa')
