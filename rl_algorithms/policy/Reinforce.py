from typing import List

from ..Agent import Agent

import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.distributions.categorical import Categorical


class Reinforce(Agent):

    def __init__(self, network: nn.Module, actions: List, alpha: float, gamma: float, eps: float, device: str = 'cpu'):
        super().__init__()

        self.actions = {i: action for i, action in enumerate(actions)}

        self.gamma = gamma
        self.eps = eps

        self.device = device
        self.network = network.to(device)  # without softmax at the output
        self.optim = torch.optim.SGD(self.network.parameters(), lr=alpha)
        self.softmax = nn.Softmax(dim=1)

        self.state_action_reward = []

    def _policy(self, state: torch.Tensor):
        """ Compute state value for all actions."""
        return self.network(state)

    def _get_action(self, state):
        """ Return an action to be taken from this state based on the policy.    """
        # if np.random.rand() < self.eps:
        #     return torch.from_numpy(np.random.choice(list(self.actions.keys()), size=(state.shape[0],)))
        with torch.no_grad():
            x = self.softmax(self._policy(state=state)*(1 - self.eps))
            action = Categorical(probs=x).sample()
            return action

    def update(self, reward):
        """ Store reward assigned to previous state-action pair.

        Args:
            reward (float): Reward received upon the transaction to the current state.
        """
        self.state_action_reward.append([self._state, self._action, reward])

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 

        Args:
            state (Any): The current state representation. After fed to ``decode_state``, the output should be eligible to be a network input.
        """
        state = self.decode_state(state)
        action = self._get_action(state)
        self._action = action
        self._state = state
        return self.actions[action.item()]

    def end_episode(self):
        """ Optimize policy.

        This method accumulates all experience (state, action, reward, ...),
        computes the return for each state-action pair in the sequence, 
        then optimizes the new policy.

        Return:
            Log of the expected return.
        """
        G = 0
        state_action_return = []
        for s, a, r in reversed(self.state_action_reward):
            G = G * self.gamma + r
            state_action_return.append((s, a, G))
        state_action_return.reverse()

        # prepare loss inputs
        states, actions, returns = zip(*state_action_return)
        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.cat(actions, dim=0).to(self.device)
        returns = torch.from_numpy(np.array(returns)).float().to(self.device)
        gammas = torch.from_numpy(np.array(
            [self.gamma**i for i in range(returns.shape[0])])).float().to(self.device)

        # compute logits & loss
        logits = self._policy(states)
        loss = (cross_entropy(logits, actions, reduction='none')
                * (returns*gammas)).sum()

        # optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # reinitialize env
        self.state_action_reward = []

        return -loss.item()

    def save(self, path: str):
        """ Save state-action value table in `path`.npy

        Args:
            path (str): The location of where to store the state-action value table.

        """
        super().save(path)
        torch.save(self.network.state_dict(), path + '.pth')

    def load(self, path):
        """ Load state-action value table.

        If it doesn't exist, a randomly-initialized table is used.

        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.network.load_state_dict(
                torch.load(path + '.pth'))
            self.network = self.network.to(self.device)
        except:
            print("No file is found in:", path)
        self.state_action_reward = []
