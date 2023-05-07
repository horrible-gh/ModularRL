# ModularRL

ModularRL is a Python library for creating and training reinforcement learning agents using various algorithms. The library is designed to be easily customizable and modular, allowing users to quickly set up and train agents for various environments without being limited to a specific algorithm.

## Installation

```powershell
pip install modular_rl
```

## Features

-   Implementations of various reinforcement learning algorithms,
    such as Proximal Policy Optimization (PPO) and Monte Carlo Tree Search (MCTS)
-   Customizable agent settings and network architectures
-   Modular structure for easy adaptation and extension across different algorithms
-   Model saving and loading functionality for easy reuse of trained models

## Supported Algorithms

-   Proximal Policy Optimization (PPO)
-   Monte Carlo Tree Search (MCTS)

Refer to the respective agent classes for each algorithm:

-   AgentPPO (Modular)
-   AgentMCTS

## Example Usage

You can use the tester.py script provided in the library to create and train an instance of an agent with default or modified settings:

```python
import modular_rl.tester as tester

tester.init_ppo()
# or
tester.init_ppo_modular()

tester.init_mcts()
```

As more algorithms are added, the tester functions will follow the naming convention init*[algorithm_name] or init*[algorithm_name]\_modular.

Please note that not all algorithms support modular training due to the nature of their design.
For such algorithms, you will need to use the non-modular training method provided by the respective agent class.
You can refer to the list of supported algorithms to determine which training method is appropriate.

Alternatively, you can create and train an instance of the AgentPPO(example) class directly in your code:

```python
from modular_rl.agents.agent_ppo import AgentPPO
from modular_rl.settings import AgentSettings

def init():
    env = AgentPPO(env=None, setting=AgentSettings.default)
    env.learn()

init()
```

To create and train an instance of the AgentPPO(example) class with modified settings, use the following code:

```python
from modular_rl.agents.agent_ppo import AgentPPO
from modular_rl.settings import AgentSettings

def init_modular():
    # Semi-automatic (defined record usage)
    # Implement your environment and pass it to 'env' parameter.
    env = AgentPPO(env=None, setting=AgentSettings.default_modular)
    env.reset()
    env.learn_reset()
    action, reward, is_done = env.learn_next()
    env.learn_check()
    env.update()

    # Proceed with the learning manually.
    env.reset()
    # Implement the 'reset' method in your environment.
    '''
    def reset(self):
        ...
        return initial_state
    '''
    env.learn_reset()
    initial_state = env.learn_reset()
    action, _ = env.select_action(initial_state)

    '''
    Note:
    Please implement the resulting state of update_step in the step function of your environment.

    For example:

    def step(self, action):
        ...
        return next_state, reward, is_done, _
    '''

    env.update_step(initial_state, None, action, -1)

    env.learn_check()
    env.update()

    env.learn_close()

init_modular()
```

## Saving and Loading Models

Agents can save and load their models using the save_model(file_name) and load_model(file_name) methods.
The file_name parameter should be the name of the file to save or load the model to/from.

Example:

```python
agent = AgentPPO(env, setting)
agent.train()

agent.save_model("my_saved_model.pth")

loaded_agent = AgentPPO(env, setting)
loaded_agent.load_model("my_saved_model.pth")
```

## Key Classes

-   AgentPPO, AgentMCTS: The main agent classes implementing various reinforcement learning algorithms.
-   PolicyNetwork, ValueNetwork, ActorCriticNetwork: Customizable neural networks for the agent's policy and value functions.
-   AgentSettings: A configuration class for setting up the agents.

## License

MIT License
