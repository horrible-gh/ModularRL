class ParamMIM:
    default = {
        'score_column': 'score',
        'simulation_iterations': 30,
        'superior_rank_adjustment_factor': 0.9,
        'inferior_rank_adjustment_factor': 0.85,
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
        'max_episodes': 10,  # Maximum number of episodes for training
        'max_timesteps': 200,  # Maximum number of timesteps for each episode
    }
