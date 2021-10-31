import gym
import minihack
from nle import nethack

import numpy as np
import time

from os.path import join, split
_HERE = split(__file__)[0]

class Sarsa:

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

    def update(self, reward):
        """ Update state-action value of previous (state, action).
        
        - `reward`: reward received upon the transaction to `next_state` from previous state.
        """
        self.sar[-1][-1] = reward
        if len(self.sar) < 2:
            return

        q = self._action_value(state=self.sar[0][0], action=self.sar[0][1])
        tmp = self.sar[0][2] - q
        tmp += self.gamma * self._action_value(state=self.sar[1][0], action=self.sar[1][1])
        self.q[(self.sar[0][0], self.sar[0][1])] = q + self.alpha * tmp

        del self.sar[0]

    def take_action(self, current_state):
        """ Choose an eps-greedy action to be taken when current state is `current_state`. """
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
            print("No saved learner is found under:", path)
            time.sleep(2)

if __name__ == '__main__':
    ALPHA = 1e-1
    GAMMA = 9e-1
    EPS = 5e-1
    ITERS = 20
    
    MOVE_ACTIONS = list(range(len(nethack.CompassDirection)))

    """ Possible observations:
    
    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    STATE = ('chars_crop', 'pixel', 'message' )

    sarsa = Sarsa(MOVE_ACTIONS, ALPHA, GAMMA, EPS)

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        observation_keys=STATE,
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5
    )

    for i in range(ITERS):
        state = env.reset()
        state = tuple(map(tuple, state[STATE[0]]))
        n = 0
        sarsa.load('sarsa')
        while True:
            n += 1
            action = sarsa.take_action(state)

            state, reward, done, info = env.step(action)
            state = tuple(map(tuple, state[STATE[0]]))

            sarsa.update(reward)

            arr = env.render('ansi')
            print(arr.replace(" ", "").replace("\n\n", ''))
            
            print('='*20, f'{action=}, {reward=}, {done=}')
            if done:
                break

        sarsa.save('sarsa')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')
        time.sleep(2)