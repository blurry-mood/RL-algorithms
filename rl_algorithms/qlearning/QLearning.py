from typing import List
from ..Agent import Agent
import numpy as np


class QLearning(Agent):

    def __init__(self, actions: List, alpha: float, gamma: float, eps: float):
        super().__init__()

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.q = {}
        self.prev_state = None
        self.prev_action = None

    def _action_value(self, state, action):
        """ Compute state-action value of this pair."""
        return self.q.get((state, action), 1e-2*np.random.randn())

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state.    """
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        action = max(self.actions, key=lambda action: self._action_value(
            state=state, action=action))
        return action

    def update(self, state, reward):
        """ Update state-action value of previous (state, action).

        Args:
            state (Any): The new state representation.
            reward (float): Reward received upon the transaction to `state`.

        Note:
            - The parameter ``state`` should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        q = self._action_value(state=self.prev_state, action=self.prev_action)
        tmp = reward - q
        tmp += self.gamma * \
            self._action_value(state, self._get_action(state, 0))
        self.q[(self.prev_state, self.prev_action)] = q + self.alpha * tmp

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 

        Args:
            state (Any): The current state representation. It should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        action = self._get_action(state, self.eps)
        
        self.prev_action = action
        self.prev_state = state
        return action

    def save(self, path: str):
        """ Save state-action value table in `path`.npy

        Args:
            path (str): The location of where to store the state-action value table.

        """
        super().save(path)
        np.save(path + '.npy', self.q)

    def load(self, path):
        """ Load state-action value table.

        If it doesn't exist, a randomly-initialized table is used.

        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.q = np.load(path + '.npy', allow_pickle='TRUE').item()
        except:
            self.q = {}
            print("No file is found in:", path)
