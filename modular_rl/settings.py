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


class AgentSettings:
    default = {
        'max_episodes': 30,  # Maximum number of episodes for training
        'max_timesteps': 200,  # Maximum number of timesteps for each episode
        'update_timestep': 2000,  # Update the policy every specified timestep
        'ppo_epochs': 4,  # Number of PPO epochs
        'mini_batch_size': 64,  # Batch size for PPO updates
        'networks': 'medium',  # Size of the hidden layer in neural networks
        'optimizer_speed': 3e-4,  # Learning rate for the optimizer
        'gamma': 0.99,  # Discount factor
        'lam': 0.95,  # Lambda parameter for GAE
        'clip_param': 0.2,  # Clipping parameter for PPO
        # If the average reward is greater than or equal to this value, training is stopped early
        'early_stop_threshold': -1,
        'done_loop_end': False,  # If True, end the episode when the done flag is set
        'reward_print': True,
    }

    default_modular = {
        # Maximum number of episodes for training (-1 for no limit)
        'max_episodes': -1,
        # Maximum number of timesteps for each episode (-1 for no limit)
        'max_timesteps': -1,
        # Update the policy every specified timestep (-1 for no limit)
        'update_timestep': -1,
        'ppo_epochs': 4,  # Number of PPO epochs
        'mini_batch_size': 64,  # Batch size for PPO updates
        'networks': 'medium',  # Size of the hidden layer in neural networks
        'optimizer_speed': 3e-4,  # Learning rate for the optimizer
        'gamma': 0.99,  # Discount factor
        'lam': 0.95,  # Lambda parameter for GAE
        'clip_param': 0.2,  # Clipping parameter for PPO
        # If the average reward is greater than or equal to this value, training is stopped early
        'early_stop_threshold': -1,
        'reward_print': True,
    }
