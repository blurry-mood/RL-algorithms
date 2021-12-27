from typing import List
from ..Agent import Agent
import numpy as np


class MonteCarlo(Agent):

    def __init__(self, actions: List, gamma: float, eps: float):
        super().__init__()

        self.actions = actions
        self.gamma = gamma
        self.eps = eps

        self.q_n = {}  # q-value & number of prior episodes
        self.state_action_reward = []

    def _action_value(self, state, action):
        """ Compute state-action value of this pair."""
        return self.q_n.get((state, action), (0, 0))[0]

    def _get_action(self, state, eps):
        """ Return an eps-greedy action to be taken from this state.    """
        if np.random.rand() < eps:
            return np.random.choice(self.actions)
        action = max(self.actions, key=lambda action: self._action_value(
            state=state, action=action))
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
            state (Any): The current state representation. It should be an immutable type since it's used as a key.
        """
        state = self.decode_state(state)
        action = self._get_action(state, self.eps)
        self._action = action
        self._state = state
        return action

    def end_episode(self):
        G = 0
        for s, a, r in reversed(self.state_action_reward):
            G = G * self.gamma + r
            n = self.q_n.get((s, a), (0, 0))[1]
            q = self._action_value(s, a)
            self.q_n[s, a] = (G/(n+1) + n/(n+1)*q, n+1)
        self.state_action_reward = []

    def save(self, path: str):
        """ Save state-action value table in `path`.npy

        Args:
            path (str): The location of where to store the state-action value table.

        """
        super().save(path)
        np.save(path + '.npy', self.q_n)

    def load(self, path):
        """ Load state-action value table.

        If it doesn't exist, a randomly-initialized table is used.

        Args:
            path (str): The location of where the state-action value table resides.
        """

        try:
            self.q_n = np.load(path + '.npy', allow_pickle='TRUE').item()
        except:
            self.q_n = {}
            print("No file is found in:", path)
        self.state_action_reward = []
