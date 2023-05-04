# ModularRL

ModularRL is a Python library for creating and training reinforcement learning agents using the Proximal Policy Optimization (PPO) algorithm. 
The library is designed to be easily customizable and modular, allowing users to quickly set up and train PPO agents for various environments.

## Installation

```powershell
pip install modular_rl
```

## Features

- Implementation of the PPO algorithm for reinforcement learning
- Customizable agent settings and network architectures
- Modular structure for easy adaptation and extension

## Example Usage

You can use the tester.py script provided in the library to create and train an instance of the AgentPPO class with default or modified settings:

```python
import modular_rl.tester as tester

tester.init()
# or
tester.init_modular()
```

Alternatively, you can create and train an instance of the AgentPPO class directly in your code:

```python
from modular_rl.agents.agent_ppo import AgentPPO
from modular_rl.settings import AgentSettings

def init():
    env = AgentPPO(env=None, setting=AgentSettings.default)
    env.learn()

init()
```

To create and train an instance of the AgentPPO class with modified settings, use the following code:

```python
from modular_rl.agents.agent_ppo import AgentPPO
from modular_rl.settings import AgentSettings

def init_modular():
    env = AgentPPO(env=None, setting=AgentSettings.default_modular)
    env.reset()
    env.learn_reset()
    action, reward, is_done = env.learn_next()
    env.learn_check()
    env.update()

    env.reset()
    env.learn_reset()

    # Proceed with the learning manually.
    initial_state = env.learn_reset()
    action, dist = env.select_action(initial_state)
    env.update_step(initial_state, dist, action, -1)

    env.learn_check()
    env.update()

    env.learn_close()


init_modular()
```

## Key Classes

- AgentPPO: The main agent class implementing the PPO algorithm.
- PolicyNetwork: A customizable neural network for the agent's policy.
- ValueNetwork: A customizable neural network for the agent's value function.
- AgentSettings: A configuration class for setting up the PPO agent.

## License
MIT License