from abc import ABC, abstractmethod
from typing import Any

from os.path import split, exists
from os import makedirs

class Agent(ABC):

    def __init__(self):
        try:
            self.decode_state(1)
        except NotImplementedError:
            print("Please implement the static method 'decode_state' before procedding.")
            exit(-1)
        except:
            pass

    def save(self, path):
        dir = split(path)[0]
        if dir == '':
            dir = '.'
        if not exists(dir):
            makedirs(dir)

    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    def start_episode(self):
        raise NotImplementedError()

    def end_episode(self):
        raise NotImplementedError()

    @abstractmethod
    def take_action(self, state):
        raise NotImplementedError()

    @abstractmethod
    def update(self, next_state, reward):
        raise NotImplementedError()

    @abstractmethod
    def decode_state(self, state) -> Any:
        raise NotImplementedError()