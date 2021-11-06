
import gym
import minihack
from nle import nethack

import numpy as np

from copy import deepcopy
import time
import rl_minihack

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

class DeepQLearner:
    """ Possible observations:
    
    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 'pixel'
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    _STATE = 'chars_crop'

    def __init__(self, actions, alpha, gamma, eps):
        self.actions = actions  # set of all possible actions
        self.alpha = alpha  # learning rate, used for QNetwork
        self.gamma = gamma  # gamma used to estimate state-action function
        self.eps = eps      # probability of picking a random action versus a greedy one

        self.Q = QNetwork(num_actions=len(actions)) # Q-Network
        self.Q_prime = deepcopy(self.Q)             # The same network used to estimate ground truth values
        self.opt = torch.optim.SGD(self.Q.parameters(), lr=self.alpha)
        self.i = 0  # counter used to trigger the update of Q_prime with Q

        self.prev_state = None  
        self.prev_action = None

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

    def update(self, next_state, reward):
        """ Update state-action value of previous (state, action).
        
        - `next_state`: the new state.
        - `reward`: reward received upon the transaction to `next_state` from previous state.
        """
        next_state = tuple(map(tuple, next_state[self._STATE]))
        if self.prev_state is not None:
            gt = reward + self.gamma * self._action_value(next_state, self._get_action(next_state, 0), q=False)
            pred = self._action_value(self.prev_state, self.prev_action)

            # update Q
            self.opt.zero_grad()
            loss = (pred-gt).pow(2)
            loss.backward()
            self.opt.step()
            
            if self.i == 100:
                # update Q_prime
                self.Q_prime = deepcopy(self.Q)
                self.i = 0
        self.i += 1
        self.prev_state = next_state
        
        try:
            return loss.item()
        except:
            pass
        

    def take_action(self, current_state):
        """ Choose an eps-greedy action to be taken when current state is `current_state`. """
        current_state = tuple(map(tuple, current_state[self._STATE]))
        action = self._get_action(current_state, self.eps)
        self.prev_action = action
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
    ALPHA = 1e-3
    GAMMA = 9e-1
    EPS = 5e-1
    ITERS = 1

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,   
        observation_keys=('chars_crop', 'pixel')
    )

    deep_qlearner = DeepQLearner(list(range(env.action_space.n)), ALPHA, GAMMA, EPS)
    deep_qlearner.load('deep_qlearner')

    for i in range(ITERS):
        state = env.reset()
        env.render(state, 1e-1)
        n = 0
        done = False
        while not done:
            n += 1
            action = deep_qlearner.take_action(state)
            state, reward, done, info = env.step(action)
            loss = deep_qlearner.update(state, reward)
            env.render(state)

            print(f'==== {n=}, {loss=}')

        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()
    deep_qlearner.save('deep_qlearner')