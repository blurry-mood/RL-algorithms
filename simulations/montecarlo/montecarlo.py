import sys
sys.path.insert(0, '/home/ayoub/Desktop/Projects/RL-algorithms/')

import gym
import rl_minihack

from rl_algorithms import MonteCarlo


def test_minihack():
    GAMMA = 0.5
    EPS = 2e-1
    ITERS = 20

    class MyAgent(MonteCarlo):
        def decode_state(self, state):
            return tuple(map(tuple, state['chars_crop']))

    env = gym.make(
        id="minihack:MiniHack-Room-5x5-v0",
        max_episode_steps=100_000,
        obs_crop_h=3,
        obs_crop_w=3,
        observation_keys=('chars_crop', 'pixel')
    )

    montecarlo = MyAgent(actions=list(
        range(env.action_space.n)), gamma=GAMMA, eps=EPS)
    montecarlo.load('montecarlo_minihack')

    for i in range(ITERS):
        state = env.reset()
        env.render(state)
        n = 0
        done = False
        while not done:
            n += 1
            action = montecarlo.take_action(state)
            state, reward, done, info = env.step(action)
            montecarlo.update(reward)
            env.render(state)
        
        montecarlo.end_episode()
        montecarlo.save('montecarlo_minihack')
        print('>'*40, f'Episode {i+1} is finished in {n} steps')

    rl_minihack.stop_rendering()

    

test_minihack()