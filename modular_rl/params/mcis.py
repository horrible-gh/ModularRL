class ParamMCIS:
    default = {
        'learn_framework': 'pytorch',
        'max_episodes': 10,  # Maximum number of episodes for training
        'max_timesteps': 200,  # Maximum number of timesteps for each episode
        'networks': 'medium',  # Size of the hidden layer in neural networks
        'optimizer_speed': 3e-4,  # Learning rate for the optimizer
        'num_simulations': 10,  # Number of MCIS simulations per move
        'cpuct': 1.0,  # Exploration constant for MCIS
        'temperature': 1.0,  # Temperature for action selection during MCIS
        'gamma': 0.99,  # Discount factor
        # If the average reward is greater than or equal to this value, training is stopped early
        'early_stop_threshold': -1,
        'done_loop_end': False,  # If True, end the episode when the done flag is set
        'reward_print': True,  # If True, print the reward during training
        'device': None,  # Device to run the model on, None for automatic selection
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
    }

    default_modular = {
        'learn_framework': 'pytorch',
        # Maximum number of episodes for training (-1 for no limit)
        'max_episodes': -1,
        # Maximum number of timesteps for each episode (-1 for no limit)
        'max_timesteps': -1,
        'networks': 'medium',  # Size of the hidden layer in neural networks
        'optimizer_speed': 3e-4,  # Learning rate for the optimizer
        'num_simulations': 800,  # Number of MCIS simulations per move
        'cpuct': 1.0,  # Exploration constant for MCIS
        'temperature': 1.0,  # Temperature for action selection during MCIS
        'gamma': 0.99,  # Discount factor
        # If the average reward is greater than or equal to this value, training is stopped early
        'early_stop_threshold': -1,
        'reward_print': True,  # If True, print the reward during training
        'device': None,  # Device to run the model on, None for automatic selection
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
    }
