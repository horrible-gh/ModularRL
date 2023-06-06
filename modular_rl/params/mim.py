
class ParamMIM:
    default = {
        # The column name used to track the score for evaluating agent's actions.
        'score_column': 'score',
        # The number of iterations for each simulation during the MIM's action selection process.
        'simulation_iterations': 30,
        # The adjustment factor applied to actions that are ranked superior (i.e., they have a higher score). Lower values result in greater down-weighting.
        'superior_rank_adjustment_factor': 0.9,
        # The adjustment factor applied to actions that are ranked inferior (i.e., they have a lower score). Lower values result in greater up-weighting.
        'inferior_rank_adjustment_factor': 0.85,
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
        'superior_rank_adjustment_factor': 0.9,
        # The adjustment factor applied to actions that are ranked inferior (i.e., they have a lower score). Lower values result in greater up-weighting.
        'inferior_rank_adjustment_factor': 0.85,
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
        'max_episodes': 1,  # Maximum number of episodes for training
        'max_timesteps': 1,  # Maximum number of timesteps for each episode
    }
