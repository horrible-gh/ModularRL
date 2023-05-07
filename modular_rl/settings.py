'''
The AgentSettings class is a configuration class used for setting up various reinforcement learning agents.

It provides default values for various parameters used in the agents, such as the maximum number of episodes,
maximum number of timesteps per episode, update timestep, network architecture, learning rate,
discount factor, early stopping threshold, and whether to end training when the environment is done.

The class currently supports PPO and MCTS algorithms, but it can be extended to include other algorithms in the future.
The default dictionary provides default values for all parameters, while the modular version provides default values
with more flexibility for specific use cases. These default values can be modified by passing in a dictionary of
key-value pairs to the AgentSettings constructor.

'''

from modular_rl.params.ppo import ParamPPO
from modular_rl.params.mcts import ParamMCTS


class AgentSettings:
    default_ppo = ParamPPO.default
    default_ppo_modular = ParamPPO.default_modular
    default_mcts = ParamMCTS.default
    default_mcts_modular = ParamMCTS.default_modular
