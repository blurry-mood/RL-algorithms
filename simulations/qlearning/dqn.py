from rl_algorithms import DQN
from torch import nn
import numpy as np
import torch
import rl_minihack
import gym


def test_minihack():
    ALPHA = 1e-1
    GAMMA = 1.01
    EPS = 0  # 5e-1
    ITERS = 100

    class MyAgent(DQN):

        def decode_state(self, state):
            s = torch.from_numpy(np.array(
                tuple(map(tuple, state['chars_crop'])))).flatten().unsqueeze(0).float()
            return s

    env = gym.make(
        id="minihack:MiniHack-Room-5x5-v0",
        max_episode_steps=100_000_000,
        obs_crop_h=5,
        obs_crop_w=5,
        observation_keys=('chars_crop', 'pixel')
    )
    dqn = MyAgent(nn.Linear(25, 5), actions=list(range(env.action_space.n)),
                       alpha=ALPHA, gamma=GAMMA, eps=EPS)
    dqn.load('dqn_minihack')

    for i in range(ITERS):
        state = env.reset()
        env.render(state, 1e-1)
        n = 0
        done = False
        while not done:
            n += 1
            action = dqn.take_action(state)
            state, reward, done, info = env.step(action)
            dqn.update(state, reward)
            env.render(state)

        dqn.save('dqn_minihack')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()


def test_robot_warehouse():
    ALPHA = 1e-2
    GAMMA = 1e-1
    EPS = 4e-1
    ITERS = 10
    N_ACTIONS = 5
    N_AGENTS = 1
    MAX_STEPS = 1e3

    class MyAgent(DQN):

        def decode_state(self, state):
            s = torch.from_numpy(np.array(tuple(state))).flatten().unsqueeze(0).float()
            return s

    env = gym.make("rware:rware-tiny-2ag-v1", max_steps=MAX_STEPS,
                   n_agents=N_AGENTS, request_queue_size=3, sensor_range=1,
                   shelf_columns=3, column_height=0, shelf_rows=1)

    agents = [MyAgent(nn.Linear(111, 5), list(range(N_ACTIONS)), ALPHA, GAMMA, EPS)
              for _ in range(N_AGENTS)]
    for i, agent in enumerate(agents):
        agent.load(f'dqn{i}')

    actions = [0 for _ in agents]
    from time import time_ns

    for episode in range(ITERS):
        states = env.reset()
        n = 0
        done = False
        _reward = 0
        start = time_ns()
        while not done:
            n += 1
            for i, (state, agent) in enumerate(zip(states, agents)):
                actions[i] = agent.take_action(state)

            states, rewards, done, info = env.step(actions)

            for state, reward, agent in zip(states, rewards, agents):
                agent.update(state, reward)
            done = all(done)
            _reward += sum(rewards)

            env.render()

        delta = (time_ns() - start)*1e-9
        _reward = _reward/N_AGENTS/n
        print(
            '>'*40, f'Episode {episode+1} is finished in {n} steps, with average reward {_reward}, done in {delta:.3f} seconds.')

        for i, agent in enumerate(agents):
            agent.save(f'dqn{i}')


test_robot_warehouse()
