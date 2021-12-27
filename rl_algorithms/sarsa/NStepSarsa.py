from typing import List
from ..Agent import Agent
import numpy as np


class NStepSarsa(Agent):

    def __init__(self, actions: List, alpha: float, gamma: float, eps: float, n: int):
        super().__init__()

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.n = n

        self.q = {}
        self.sar = []

    def _action_value(self, state, action):
        """ Compute state-action value of this pair."""
        return self.q.get((state, action), np.random.randn())

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state.    """
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        action = max(self.actions, key=lambda action: self._action_value(
            state=state, action=action))
        return action

    def update(self, reward):
        """ Update state-action value of previous (state, action).

        Args:
            reward (float): Reward received upon the transaction to `state`.

        """
        self.sar[-1][-1] = reward
        if len(self.sar) < self.n + 1:
            return

        q = self._action_value(state=self.sar[0][0], action=self.sar[0][1])
        G = 0
        for i, (*_, reward) in enumerate(self.sar):
            G += (self.gamma**i) * reward
        self.q[self.sar[0][0], self.sar[0][1]] = q + self.alpha * (G - q)

        del self.sar[0]

    def take_action(self, state):
        """ Choose an eps-greedy action to be taken from this state. 

        Args:
            state (Any): The current state representation. It should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        action = self._get_action(state, self.eps)
        self.sar.append([state, action, 0])
        return action

    def end_episode(self):
        """ Update state-action value of the last `n` (state, action) pairs. 
        """
        for __ in range(len(self.sar)):
            q = self._action_value(state=self.sar[0][0], action=self.sar[0][1])
            G = 0
            for i, (*_, reward) in enumerate(self.sar):
                G += (self.gamma**i) * reward
            self.q[self.sar[0][0], self.sar[0][1]] = q + self.alpha * (G - q)

            del self.sar[0]

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
