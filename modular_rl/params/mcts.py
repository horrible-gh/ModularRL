class ParamMCTS:
    default = {
        'max_episodes': 10,  # Maximum number of episodes for training
        'max_timesteps': 200,  # Maximum number of timesteps for each episode
        'update_timestep': 2000,  # Update the policy every specified timestep
        'networks': 'medium',  # Size of the hidden layer in neural networks
        'optimizer_speed': 3e-4,  # Learning rate for the optimizer
        'num_simulations': 10,
        'cpuct': 1.0,
        'temperature': 1.0,
        'gamma': 0.99,  # Discount factor
        # If the average reward is greater than or equal to this value, training is stopped early
        'early_stop_threshold': -1,
        'done_loop_end': False,  # If True, end the episode when the done flag is set
        'reward_print': True,
        'device': None,
        'log_level': 'debug',
    }

    default_modular = {
        # Maximum number of episodes for training (-1 for no limit)
        'max_episodes': -1,
        # Maximum number of timesteps for each episode (-1 for no limit)
        'max_timesteps': -1,
        # Update the policy every specified timestep (-1 for no limit)
        'update_timestep': -1,
        'networks': 'medium',  # Size of the hidden layer in neural networks
        'optimizer_speed': 3e-4,  # Learning rate for the optimizer
        'num_simulations': 800,
        'cpuct': 1.0,
        'temperature': 1.0,
        'gamma': 0.99,  # Discount factor
        # If the average reward is greater than or equal to this value, training is stopped early
        'early_stop_threshold': -1,
        'reward_print': True,
        'device': None,
        'log_level': 'debug',
    }
