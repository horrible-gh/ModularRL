class ParamMIM:
    default = {
        # The column name used to track the score for evaluating agent's actions.
        'score_column': 'score',
        # The number of iterations for each simulation during the MIM's action selection process.
        'simulation_iterations': 30,
        # The adjustment factor applied to actions that are ranked superior (i.e., they have a higher score). Lower values result in greater down-weighting.
        'superior_rank_adjustment_factor': 0.95,
        # The adjustment factor applied to actions that are ranked inferior (i.e., they have a lower score). Lower values result in greater up-weighting.
        'inferior_rank_adjustment_factor': 0.9,
        # The adjustment factor for standard deviation in the weight adjustment calculation. Lower values lead to a stronger adjustment.
        'std_deviation_factor': 0.066,
        # The adjustment factor for skewness in the weight adjustment calculation. Lower values lead to a stronger adjustment.
        'skewness_factor': 0.033,
        # The adjustment factor for kurtosis in the weight adjustment calculation. Lower values lead to a stronger adjustment.
        'kurtosis_factor': 0.025,
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
        'max_episodes': 10,  # Maximum number of episodes for training
        'max_timesteps': 200,  # Maximum number of timesteps for each episode
    }

    default_modular = {
        # The column name used to track the score for evaluating agent's actions.
        'score_column': 'score',
        # The number of iterations for each simulation during the MIM's action selection process.
        'simulation_iterations': 1,
        # The adjustment factor applied to actions that are ranked superior (i.e., they have a higher score). Lower values result in greater down-weighting.
        'superior_rank_adjustment_factor': 0.95,
        # The adjustment factor applied to actions that are ranked inferior (i.e., they have a lower score). Lower values result in greater up-weighting.
        'inferior_rank_adjustment_factor': 0.9,
        # The adjustment factor for standard deviation in the weight adjustment calculation. Lower values lead to a stronger adjustment.
        'std_deviation_factor': 0.066,
        # The adjustment factor for skewness in the weight adjustment calculation. Lower values lead to a stronger adjustment.
        'skewness_factor': 0.033,
        # The adjustment factor for kurtosis in the weight adjustment calculation. Lower values lead to a stronger adjustment.
        'kurtosis_factor': 0.025,
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
        'max_episodes': 1,  # Maximum number of episodes for training
        'max_timesteps': 1,  # Maximum number of timesteps for each episode
    }
