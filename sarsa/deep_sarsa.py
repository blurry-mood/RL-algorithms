import gym
import minihack
from nle import nethack

import numpy as np

from copy import deepcopy
import rl_minihack

import torch

from os.path import join, split

from dqn import *

_HERE = split(__file__)[0]


class DeepSarsa:
    """ Possible observations:

    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 'pixel'
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
    _STATE = 'chars_crop'

    def __init__(self, input_channels, actions, alpha, c=55, gamma=0.9, eps=0.7, t=1024, capacity=1024, bs=32, device='cpu'):
        self.actions = actions  # set of all possible actions
        self.alpha = alpha  # learning rate, used for QNetwork
        self.gamma = gamma  # gamma used to estimate state-action function
        self.eps = eps      # probability of picking a random action versus a greedy one

        self.bs = bs
        self.c = c
        self.t = t

        self.device = device

        self.buffer = ExperienceReplay(capacity)

        self.Q = QNetwork(input_channels=input_channels,
                          num_actions=len(actions)).to(device)  # Q-Network
        # The same network used to estimate ground truth values
        self.Q_prime = deepcopy(self.Q).to(device).eval()
        self.opt = torch.optim.SGD(self.Q.parameters(), lr=self.alpha)
        self.i = 0  # counter used to trigger the update of Q_prime with Q

        self.sar = []

    def _action_value(self, state, action=None, q: bool = True):
        """ if q is True, the `self.Q` network is used, otherwise, `self.Q_prime` is used"""
        Q = self.Q if q else self.Q_prime
        state = state.float().unsqueeze(1)
        # print(state.shape)
        # exit(0)
        n = state.shape[0]
        if action is not None:
            value = Q(state.to(self.device))[list(range(n)), action].cpu()
        else:
            value = Q(state.to(self.device)).cpu()
        return value

    def _get_action(self, state, eps):
        with torch.no_grad():
            if np.random.rand() < eps:  # * 0.5*(np.cos(2 * np.pi * self.i/self.t)+1):
                return np.random.choice(self.actions)
            actions = self._action_value(state=state, q=False).argmax(dim=1)
            return actions

    def update(self, reward):
        """ Update state-action value of previous (state, action).

        - `next_state`: the new state.
        - `reward`: reward received upon the transaction to `next_state` from previous state.
        """
        self.sar[-1][-1] = reward
        if len(self.sar) < 2:
            return

        # register history: state, action, reward, state, action
        state, action, reward, next_state, next_action = (
            *self.sar[0], *self.sar[1][:-1])

        self.buffer.append((
            state.cpu().numpy(),
            action,
            np.array(reward),
            next_state.cpu().numpy(),
            next_action)
        )

        # sample batch_size
        states, actions, rewards, next_states, next_actions = self.buffer.sample(
            self.bs)

        # compute loss
        gt = rewards + self.gamma * \
            self._action_value(next_states.squeeze(1), next_actions, q=False)
        pred = self._action_value(states.squeeze(1), actions, q=True)
        loss = (pred - gt).pow(2).mean()

        # update Q
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.i % self.c == 0:
            # update Q_prim
            self.Q_prime = deepcopy(self.Q).eval()
        self.i += 1

        del self.sar[0]

        try:
            return loss.item()
        except:
            pass

    def take_action(self, current_state, eval=False):
        """ Choose an eps-greedy action to be taken when the current state is `current_state`. """
        current_state = torch.from_numpy(
            np.array(current_state[self._STATE])).unsqueeze(0)
        action = self._get_action(current_state, self.eps if not eval else 0)
        if not eval:
            self.sar.append([current_state, action, 0])
        return action.item()

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
    EPS = 3e-1
    ITERS = 100

    env = gym.make(
        id="MiniHack-Room-5x5-v0",
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,
        observation_keys=('chars_crop', 'pixel')
    )

    deep_sarsa = DeepSarsa(input_channels=1, actions=list(
        range(env.action_space.n)), alpha=ALPHA, gamma=GAMMA, eps=EPS)
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
            loss = deep_sarsa.update(reward)
            env.render(state)

            print(f'==== {n=}, {loss=}')

        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()
    deep_sarsa.save('deep_sarsa')
