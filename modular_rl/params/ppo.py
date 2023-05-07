class ParamPPO:
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
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
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
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
    }
