import gym
import minihack
from nle import nethack

import numpy as np
import time

from os.path import join, split
_HERE = split(__file__)[0]

class ApproxMonteCarlo:

    def __init__(self, dimension:int, actions, gamma, eps, lr):
        self.actions = actions
        self.gamma = gamma
        self.eps = eps
        self.lr = lr

        self.state_action_reward = []
        self.w = np.ones(dimension, dtype=np.float32) # dimension= state.size + action.size + 1

    def _action_value(self, state, action):
        """ Returns both the value and feature vector"""
        x = np.array(state)
        start = np.where(x=='@')
        end = np.where(x=='>')
        start = np.array(start)
        end = np.array(end)
        # x = np.concatenate((start, end), axis=0)
        x = np.insert(start, obj=0, values=[1, action])
        return (x * self.w).sum(), x

    def _get_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        qs = {(action, self._action_value(state=state, action=action)[0]) for action in self.actions}
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
            value, x = self._action_value(s, a)
            self.w = self.w + self.lr * (G - value) * x
        self.state_action_reward = []  

    def save(self, path):
        np.save(join(_HERE, path + '.npy'), self.w)

    def load(self, path):
        try:
            self.w = np.load(join(_HERE, path + '.npy'), allow_pickle='TRUE').item()
        except:
            print("No saved learner is found under:", path)
            # time.sleep(2)

if __name__ == '__main__':
    GAMMA = 9e-1
    EPS = 4e-1
    LR = 1e-2
    DIMENSION = 4
    ITERS = 20
    
    MOVE_ACTIONS = list(range(len(nethack.CompassDirection)))

    """ Possible observations:
    
    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    STATE = ('chars_crop', 'pixel', 'message' )

    approx_montecarlo = ApproxMonteCarlo(DIMENSION, MOVE_ACTIONS, GAMMA, EPS, LR)

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        observation_keys=STATE,
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,   
    )

    for i in range(ITERS):
        env.reset()
        n = 0
        approx_montecarlo.load('approx_montecarlo')
        while True:
            n += 1
            state = env.render('ansi').replace(" ", "").replace("\n", '') 
            state = np.array(list(state)).reshape(5,5)

            action = approx_montecarlo.take_action(state)

            _, reward, done, info = env.step(action)
            approx_montecarlo.record(state, action, reward)

            print('='*20, f'{action=}, {reward=}, {done=}')
            print(state)
            
            if done:
                break

        
        approx_montecarlo.end_episode()
        approx_montecarlo.save('approx_montecarlo')
        print('>'*40, f'Episode {i+1} is finished in {n} steps', 'W =', approx_montecarlo.w)
        time.sleep(2)