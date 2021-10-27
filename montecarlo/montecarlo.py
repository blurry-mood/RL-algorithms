import gym
import minihack
from nle import nethack

import numpy as np
import time

from os.path import join, split
_HERE = split(__file__)[0]

class MonteCarlo:

    def __init__(self, actions, gamma, eps):
        self.actions = actions
        self.gamma = gamma
        self.eps = eps
        self.q = {}
        self.n = {}
        self.state_action_reward = []

    def action_value(self, state, action):
        return self.q.get((state, action), 1e-3*np.random.randn())

    def _get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        qs = {(action, self.action_value(state=state, action=action)) for action in self.actions}
        action = max(qs, key = lambda x: x[1])
        return action[0]

    def record(self, state, action, reward):
        self.state_action_reward.append((state, action, reward))

    def take_action(self, state):
        return self._get_action(state, self.eps)

    def end_episode(self):
        G = 0
        for s, a, r in reversed(self.state_action_reward):
            G = G * self.gamma + r
            n = self.n.get((s,a), 0)
            q = self.q.get((s,a),0)
            self.q[s, a] = G/(n+1) + n/(n+1)*q
        self.state_action_reward = []            

    def save(self, path):
        np.save(join(_HERE, path + '.npy'), self.q)

    def load(self, path):
        try:
            self.q = np.load(join(_HERE, path + '.npy'), allow_pickle='TRUE').item()
        except:
            print("No saved learner is found under:", path)
            time.sleep(2)

if __name__ == '__main__':
    GAMMA = 9e-1
    EPS = 9e-2
    ITERS = 20
    
    MOVE_ACTIONS = list(range(len(nethack.CompassDirection)))

    """ Possible observations:
    
    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    STATE = ('chars_crop', 'pixel', 'message' )

    montecarlo = MonteCarlo(MOVE_ACTIONS, GAMMA, EPS)

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
        montecarlo.load('montecarlo')
        while True:
            n += 1
            action = montecarlo.take_action(state)

            _state, reward, done, info = env.step(action)
            _state = tuple(map(tuple, _state[STATE[0]]))

            montecarlo.record(state, action, reward)
            state = _state

            arr = env.render('ansi')            
            print(arr.replace(" ", "").replace("\n\n", ''))            
            print('='*20, f'{action=}, {reward=}, {done=}')
            if done:
                break
        
        montecarlo.end_episode()
        montecarlo.save('montecarlo')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')
        time.sleep(2)