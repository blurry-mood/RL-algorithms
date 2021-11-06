import gym
import minihack
from nle import nethack

import rl_minihack

import numpy as np

from copy import deepcopy
import time

import torch
from torch import nn

from os.path import join, split
_HERE = split(__file__)[0]


class QNetwork(nn.Module):

    def __init__(self, num_actions) -> None:
        super().__init__()

        self.model = nn.Sequential(
                                    nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1), nn.PReLU(),
                                    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.PReLU(),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(128, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, num_actions),
                                    nn.PReLU()
                                    )

    def forward(self, x):
        # since the image is just matrix, we need to add batch and channel dimensions,
        # and change type from byte to float
        prob = self.model(x)
        return prob



class DeepSarsa:
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

        self.Q = QNetwork(num_actions=len(actions)) # Q-Network
        self.Q_prime = deepcopy(self.Q)             # The same network used to estimate ground truth values
        self.opt = torch.optim.SGD(self.Q.parameters(), lr=self.alpha)
        self.i = 0  # counter used to trigger the update of Q_prime with Q

        self.sar = []   # stands for state-action-reward

    def _action_value(self, state, action, q:bool=True):
        """ if q is True, the `self.Q` network is used, otherwise, `self.Q_prime` is used"""
        Q = self.Q if q else self.Q_prime
        state = torch.tensor(state).unsqueeze(0).unsqueeze(0).float()
        value = Q(state)[:, action]
        return value if q else value.item()

    def _get_action(self, state, eps):
        """ 
        - state: the state at which the age is.
        - eps: probability of picking a random action. If set to 0, the greedy action is ALWAYS chosen.
        """
        with torch.no_grad():
            if np.random.rand() < eps:
                return np.random.choice(self.actions)
            qs = {(action, self._action_value(state=state, action=action)) for action in self.actions}
            action = max(qs, key = lambda x: x[1])
            return action[0]

    def _step(self, done:bool):
        gt = self.sar[-1][-1]
        if not done:
            gt += self.gamma * self._action_value(state=self.sar[1][0], action=self.sar[1][1], q=False)
        pred = self._action_value(state=self.sar[0][0], action=self.sar[0][1])

        # update Q
        self.opt.zero_grad()
        loss = (pred-gt).pow(2)
        loss.backward()
        self.opt.step()
        return loss

    def update(self, reward, done:bool):
        """ Update state-action value of previous (state, action).
        
        - `reward`: reward received upon the transaction to `next_state` from previous state.
        - `done`: boolean specifying whether the episode ended.
        """
        self.sar[-1][-1] = reward
        if len(self.sar) < 2:
            return

        loss = self._step(False)

        if self.i == 100:
            # update Q_prime
            self.Q_prime = deepcopy(self.Q)
            self.i = 0
        self.i += 1
            
        del self.sar[0]

        if done:
            self._step(True)

        return loss.item()


    def take_action(self, current_state):
        """ Choose an eps-greedy action to be taken when current state is `current_state`. """
        current_state = tuple(map(tuple, current_state[self._STATE]))
        action = self._get_action(current_state, self.eps)
        self.sar.append([current_state, action, 0])
        return action

    def save(self, path):
        """ Load state-action value q-network in `path`.npy """

        torch.save(self.Q.state_dict(), join(_HERE, path + '.pth'))

    def load(self, path):
        """ Load state-action value q-network.
        
        If it doesn't exist, use the random network.
        """

        try:
            self.Q.load_state_dict(torch.load(join(_HERE, path + '.pth')))
        except:
            print("=============>  No saved learner is found under:", path)

if __name__ == '__main__':
    ALPHA = 1e-4
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

    deep_sarsa = DeepSarsa(list(range(env.action_space.n)), ALPHA, GAMMA, EPS)
    deep_sarsa.load('deep_sarsa')

    for i in range(ITERS):
        state = env.reset()
        env.render(state, 1e-1)
        n = 0
        done = False
        while not done:
            n += 1
            action = deep_sarsa.take_action(state)
            state, reward, done, info = env.step(action)
            loss = deep_sarsa.update(reward, done)                
            env.render(state)
            
            print(f'==== {n=}, {loss=}')

        print('>'*20, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()
    deep_sarsa.save('deep_sarsa')
