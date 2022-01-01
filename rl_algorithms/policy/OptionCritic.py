from copy import deepcopy
from typing import List

from torch.nn.functional import cross_entropy

from ..Agent import Agent
import numpy as np

import torch
from torch import nn
from torch.distributions.categorical import Categorical


class OptionCritic(Agent):

    def __init__(self, option_net:nn.Module, action_net: nn.Module, termination_net: nn.Module, number_options: int, actions: List, alpha1: float, alpha2: float, gamma: float, eps1: float, eps2: float, device='cpu'):
        super().__init__()

        self.actions = {i: action for i, action in enumerate(actions)}
        self.gamma = gamma
        self.eps1 = eps1
        self.eps2 = eps2

        self.device = device

        self.option_net = option_net.to(device)
        self.action_net = action_net.to(device)
        self.termination_net = termination_net.to(device)

        self.v_loss = nn.MSELoss(reduction='sum')
        self.policy_loss = nn.CrossEntropyLoss(reduction='none')
        self.policy_optim = torch.optim.SGD(
            self.policy_net.parameters(), lr=alpha_policy)
        self.v_optim = torch.optim.SGD(self.v_net.parameters(), lr=alpha_q)
        self.softmax = nn.Softmax(dim=1)

        self.prev_state = None
        self.prev_action = None

    def _termination_prob(self, state):
        return self.beta_net(state)

    def _policy_over_options(self, state: torch.Tensor, option:int=None):
        out = self.option_net(state)
        if option is None:
            return out
        return out[:, option]
    
    def _intra_option_policy(self, state:torch.Tensor, option:int=None, action:int=None):
        out = self.action_net(state)
        if option is None:
            return out
        if action is None:
            return out[:,option]
        return out[:,option,action]

    def _state_value(self, state):
        value = self.v_net(state)
        return value

    def _get_action(self, state):
        """ Return an action to be taken from this state based on the policy.    """
        if np.random.rand() < self.eps:
            return torch.from_numpy(np.random.choice(list(self.actions.keys()), size=(state.shape[0],)))
        with torch.no_grad():
            x = self.softmax(self._policy(state=state))
            action = Categorical(probs=x).sample()
            return action

    def start_episode(self):
        self.I = 1
        self.prev_action = None
        self.prev_state = None

    def update(self, state: torch.Tensor, reward: float):
        """ Update state-action value of previous (state, action).

        Args:
            state (Any): The new state representation.
            reward (float): Reward received upon the transaction to `state`.

        Note:
            - The parameter ``state`` should be a tensor with the leading batch dimension.
        """
        state = self.decode_state(state).to(self.device)

        if self.prev_state is not None:
            gt = reward + self.gamma * self._state_value(state)
            pred = self._state_value(self.prev_state)
            logits = self._policy(self.prev_state)
            delta = (gt-pred).detach()

            v_loss = self.v_loss(pred, gt)
            policy_loss = (self.policy_loss(
                logits, self.prev_action)*self.I*delta).sum()

            # update weights
            self.v_optim.zero_grad()
            self.policy_optim.zero_grad()

            v_loss.backward()
            self.v_optim.step()

            policy_loss.backward()
            self.policy_optim.step()

        self.prev_state = state
        self.I = self.gamma * self.I

        try:
            return v_loss.item(), -policy_loss.item()
        except:
            return None, None

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 

        Args:
            state (Any): The current state representation. After fed to ``decode_state``, the output should be eligible to be a network input.
        """
        state = self.decode_state(state).to(self.device)
        assert state.shape[0] == 1

        action = self._get_action(state)
        self.prev_action = action
        return self.actions[action.item()]

    def save(self, path: str):
        """ Save state-action value table in `path`.npy

        Args:
            path (str): The location of where to store the state-action value table.

        """
        super().save(path)
        torch.save(self.policy_net.state_dict(), path + '_actor.pth')
        torch.save(self.v_net.state_dict(), path + '_critic.pth')

    def load(self, path):
        """ Load state-action value table.

        If it doesn't exist, a randomly-initialized table is used.

        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.policy_net.load_state_dict(torch.load(path + '_actor.pth'))
            self.v_net.load_state_dict(torch.load(path + '_critic.pth'))
        except:
            print("No file is found in:", path)
