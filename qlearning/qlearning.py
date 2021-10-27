import gym
import minihack
from nle import nethack

import numpy as np
import time

from os.path import join, split
_HERE = split(__file__)[0]

class QLearner:

    def __init__(self, actions, alpha, gamma, eps):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = {}
        self.prev_state = None
        self.prev_action = None

    def action_value(self, state, action):
        return self.q.get((state, action), 1e-3*np.random.randn())

    def _get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        qs = {(action, self.action_value(state=state, action=action)) for action in self.actions}
        action = max(qs, key = lambda x: x[1])
        return action[0]

    def update(self, state, reward):
        q = self.action_value(state=self.prev_state, action=self.prev_action)
        tmp = reward - q
        tmp += self.gamma * self.action_value(state, self._get_action(state, 0))
        self.q[(self.prev_state, self.prev_action)] = q + self.alpha * tmp
        self.prev_state = state

    def take_action(self, state):
        action = self._get_action(state, self.eps)
        self.prev_action = action
        return action

    def save(self, path):
        np.save(join(_HERE, path + '.npy'), self.q)

    def load(self, path):
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
    
    MOVE_ACTIONS = list(range(len(nethack.CompassDirection)))

    """ Possible observations:
    
    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    STATE = ('chars_crop', 'pixel', 'message' )

    qlearner = QLearner(MOVE_ACTIONS, ALPHA, GAMMA, EPS)

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        observation_keys=STATE,
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,   
    )

    for i in range(ITERS):
        state = env.reset()
        state = tuple(map(tuple, state[STATE[0]]))
        n = 0
        qlearner.load('qlearner')
        while True:
            n += 1
            action = qlearner.take_action(state)

            state, reward, done, info = env.step(action)
            state = tuple(map(tuple, state[STATE[0]]))

            qlearner.update(state, reward)
            arr = env.render('ansi')
            print(arr.replace(" ", "").replace("\n\n", ''))
            
            print('='*20, f'{action=}, {reward=}, {done=}')
            if done:
                break

        qlearner.save('qlearner')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')
        time.sleep(2)