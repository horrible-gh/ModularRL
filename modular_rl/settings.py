'''
The AgentSettings class is a configuration class used for setting up the PPO agent. 

It provides default values for various parameters used in the agent, 
such as the maximum number of episodes, maximum number of timesteps per episode, PPO update timestep, 
number of PPO epochs, mini-batch size, network architecture, learning rate, 
discount factor, lambda factor, clipping parameter, early stopping threshold, 
and whether to end training when the environment is done.

The default dictionary provides default values for all parameters, 
while default_modular provides default values for all parameters except done_loop_end. 
These default values can be modified by passing in a dictionary of key-value pairs to the AgentSettings constructor.

'''

from modular_rl.params.ppo import ParamPPO
from modular_rl.params.mcts import ParamMCTS


class AgentSettings:
    default_ppo = ParamPPO.default
    default_ppo_modular = ParamPPO.default_modular
    default_mcts = ParamMCTS.default
    default_mcts_modular = ParamMCTS.default_modular
