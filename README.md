# RL-algorithms
Repository with Reinforcement Learning (RL) algorithms tested in different simulations.

# Install
To install the latest stable version:
```console
$ pip install rl-algorithms
```

For a specific version:
```console
$ pip install rl-algorithms==0.0.1
```

To install the latest version available on Github:
```console
$ pip install git+https://github.com/blurry-mood/RL-algorithms
```

# Run simulations
1. Install the desired environment, for example Robotic Warehouse:
```console
$ cd environments
$ sh warehouse_bot.sh
```
2. Run simulation, for instance using a Q-Learning agent:
```console
$ cd simulations/qlearning
$ python qlearning.py
```


# What's available
## Algorithms
- On-Policy Monte Carlo
- Q-Learning
- SARSA
- n-step SARSA
- Deep Q-Learning (DQN)
- Deep Q-Learning with SARSA update rule (DSN)
- REINFORCE
- Actor-Critic

## Environments
- [MiniHack](https://github.com/facebookresearch/minihack)
- [Panda Gym](https://github.com/qgallouedec/panda-gym)
- [Robotic Warehouse](https://github.com/semitable/robotic-warehouse)

# Technical details
Every algorithm is implemented as a subclass of [Agent](https://github.com/blurry-mood/RL-algorithms/blob/main/rl_algorithms/Agent.py). It imperatively needs to implement some methods, namely, `save`, `load`, `take_action`, `update`, and `decode_state`.

Each algorithm re-implements all methods but `decode_state`, the latter is left for user to implement based on target environment. `take_action` method provides a description to what that method should look like (its inputs & outputs).  
Based on the environment, a class extending the desired algorithm class must reimplement `decode_state`.

Here's a concrete example on how to use the package:
```python
from rl_algorithms import QLearning

class MyAgent(QLearning):
    def decode_state(self, state):
        return tuple(state)

qlearning = MyAgent(actions=list(range(10)), alpha=1e-2, gamma=0.85, eps=0.2)
```
For more examples, check the content of files inside `simulations/` folder.

The base agents (algorithms) are implemented in a way that makes them ready to use off-the-shelf; the same methods are called in the same order in the script.  
Here's a script that illustrates the idea:
```python
"""
After defining the agent class & instance (for e.g. named qlearning)
"""

qlearning.load('qlearning_minihack')    # load agent

for episode in range(10):
    state = env.reset()
    env.render(state)
    n = 0
    done = False

    qlearning.start_episode()           # initialize agent for a new episode
    
    while not done:
        n += 1
        
        action = qlearning.take_action(state)   # take action based on state

        state, reward, done, info = env.step(action)
        
        qlearning.update(state, reward) # learn from reward

        env.render(state)

    qlearning.end_episode()             # update agent's internal logic
    qlearning.save('qlearning_minihack')    # save agent in Hard drive

```
The only thing that changes from an algorithm to another are the inputs and outputs of each method.  
Also, some algorithms don't require calling all methods, for Q-Learning `start_episode` and `end_episode` can be safely discarded.


# Bug or Feature
RL-Algorithms is a growing package. If you encounter a **bug** or would like to request a **feature**, please feel free to open an issue [here](https://github.com/blurry-mood/RL-algorithms/issues).