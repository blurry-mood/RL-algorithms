import collections
from copy import deepcopy
from typing import List

import torch
from ..Agent import Agent
import numpy as np

from torch import nn


class DSN(Agent):

    def __init__(self, network: nn.Module, actions: List, alpha: float, gamma: float, eps: float, c: int = 128, t: int = 1024, capacity: int = 1024, bs: int = 32, device='cpu'):
        super().__init__()

        self.actions = {i: action for i, action in enumerate(actions)}
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.bs = bs
        self.c = c
        self.t = t

        self.device = device

        self.buffer = ExperienceReplay(capacity, device)
        self.Q = network.to(device)
        self.Q_prime = deepcopy(self.Q).to(device).eval()

        self.loss = nn.MSELoss()
        self.opt = torch.optim.SGD(self.Q.parameters(), lr=self.alpha)
        self.i = 0  # counter used to trigger the update of Q_prime with Q

        self.sar = []

    def _action_value(self, state, action=None, clone: bool = False):
        """ If clone is False, the `self.Q` network is used, otherwise, `self.Q_prime` is used. """
        Q = self.Q if not clone else self.Q_prime
        n = state.shape[0]
        state = state.to(self.device)
        if action is not None:
            value = Q(state)[list(range(n)), action]
        else:
            value = Q(state)
        return value

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state.    """
        with torch.no_grad():
            if np.random.rand() < eps:  # * 0.5*(np.cos(2 * np.pi * self.i/self.t)+1):
                return torch.from_numpy(np.random.choice(list(self.actions.keys()), size=(state.shape[0],)))
            actions = self._action_value(state=state, clone=True).argmax(dim=1)
            return actions

    def update(self, reward: float):
        """ Update state-action value of previous (state, action).

        Args:
            reward (float): Reward received upon the transaction to `state`.
        """
        self.sar[-1][-1] = reward
        if len(self.sar) < 2:
            return

        # register history
        state, action, reward, next_state, next_action = (
            *self.sar[0], *self.sar[1][:-1])
        self.buffer.append((state, action, torch.tensor(
            reward).unsqueeze(0).float(), next_state, next_action))

        # sample batch_size
        states, actions, rewards, next_states, next_actions = self.buffer.sample(
            self.bs)

        # compute loss
        gt = rewards + self.gamma * \
            self._action_value(next_states, next_actions, clone=True)
        pred = self._action_value(states, actions, clone=False)
        loss = self.loss(pred, gt)

        # update Q
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.i == self.c:
            # update Q_prim
            self.i = 0
            self.Q_prime = deepcopy(self.Q).eval()
        self.i += 1

        del self.sar[0]

        try:
            return loss.item()
        except:
            return None

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 

        Args:
            state (Any): The current state representation. After fed to ``decode_state``, the output should be eligible to be a network input.
        """
        state = self.decode_state(state)
        assert state.shape[0] == 1

        action = self._get_action(state, self.eps).cpu()
        self.sar.append([state, action, 0])
        return self.actions[action.item()]

    def save(self, path: str):
        """ Save state-action value table in `path`.npy

        Args:
            path (str): The location of where to store the state-action value table.

        """
        super().save(path)
        torch.save(self.Q.state_dict(), path + '.pth')

    def load(self, path):
        """ Load state-action value table.

        If it doesn't exist, a randomly-initialized table is used.

        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.Q.load_state_dict(torch.load(path + '.pth'))
            self.Q = self.Q.to(self.device)
            self.Q_prime = deepcopy(self.Q).to(self.device).eval()
        except:
            print("No file is found in:", path)


class ExperienceReplay:

    def __init__(self, capacity, device):
        self.buffer = collections.deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        try:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=False)
        except:
            indices = np.random.choice(
                len(self.buffer), batch_size, replace=True)

        states, actions, rewards, next_states, next_actions = map(lambda x: torch.cat(
            x, dim=0).to(self.device), zip(*(self.buffer[idx] for idx in indices)))
        return states, actions, rewards, next_states, next_actions
