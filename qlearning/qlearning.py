import gym
import minihack
from nle import nethack

import rl_minihack

import numpy as np
import time

from os.path import join, split
_HERE = split(__file__)[0]

class QLearner:
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
        self.prev_state = None
        self.prev_action = None

    def _action_value(self, state, action):
        return self.q.get((state, action), 1e-3*np.random.randn())

    def _get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        qs = {(action, self._action_value(state=state, action=action)) for action in self.actions}
        action = max(qs, key = lambda x: x[1])
        return action[0]

    def update(self, next_state, reward):
        """ Update state-action value of previous (state, action).
        
        - `next_state`: the new state.
        - `reward`: reward received upon the transaction to `next_state` from previous state.
        """
        next_state = tuple(map(tuple, next_state[self._STATE]))
        q = self._action_value(state=self.prev_state, action=self.prev_action)
        tmp = reward - q
        tmp += self.gamma * self._action_value(next_state, self._get_action(next_state, 0))
        self.q[(self.prev_state, self.prev_action)] = q + self.alpha * tmp
        self.prev_state = next_state

    def take_action(self, current_state):
        """ Choose an eps-greedy action to be taken when current state is `current_state`. """
        current_state = tuple(map(tuple, current_state[self._STATE]))
        action = self._get_action(current_state, self.eps)
        self.prev_action = action
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
            print("No saved learner is found under:", path)
            time.sleep(2)

if __name__ == '__main__':
    ALPHA = 1e-1
    GAMMA = 9e-1
    EPS = 9e-2
    ITERS = 20

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,   
        observation_keys=('chars_crop', 'pixel')
    )

    qlearner = QLearner(list(range(env.action_space.n)), ALPHA, GAMMA, EPS)
    qlearner.load('qlearner')

    for i in range(ITERS):
        state = env.reset()
        env.render(state, 1e-1)
        n = 0
        done = False
        while not done:
            n += 1
            action = qlearner.take_action(state)
            state, reward, done, info = env.step(action)
            qlearner.update(state, reward)
            env.render(state)

        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()
    qlearner.save('qlearner')

