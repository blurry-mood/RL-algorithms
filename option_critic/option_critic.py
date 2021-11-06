from os import terminal_size
from types import new_class
import gym
import minihack
from nle import nethack


MOVE_ACTIONS = list(range(len(nethack.CompassDirection)))

""" Possible observations:

    ['glyphs', 'chars', 'colors', 'specials', 'glyphs_crop', 
    'chars_crop', 'colors_crop', 'specials_crop', 'blstats', 'message']
    """
STATE = ('chars_crop', 'pixel', 'message')

env = gym.make(
    id="MiniHack-Room-5x5-v0",
    observation_keys=STATE,
    max_episode_steps=100_000_000,
    obs_crop_h=5,
    obs_crop_w=5,
    savedir='./runs'
)

option_critic = None
gamma = 0.9
alpha = 1e-2

obs = env.reset()

# extract state from observation
state = option_critic.get_state(obs)
# pick option given obs
option = option_critic.sample_option(obs)

for _ in range(100):
    # given state, take action using intra-option policy
    action = option_critic.sample_action(option, state)

    # take action and receive feedback from environment
    obs, reward, done, info = env.step(action)

    # compute state
    next_state = option_critic.get_state(obs)

    # compute delta
    delta = reward - option_critic.get_value(option, state, action)

    # if not done add value from next state
    if not done:
        termination_pb, next_option = option_critic.get_termination_pb(option, next_state)
        delta += gamma * ((1 - termination_pb)*option_critic.get_value(option, next_state) + 
                            termination_pb*option_critic.get_value(next_option, next_state))

    # update option-state-action value
    option_critic.add_to_value(alpha*delta, option=option, state=state, action=action)




