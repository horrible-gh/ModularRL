class ParamMIM:
    default = {
        'fixed_states':'fixed_states',
        'unknown_spaces':'unknown_spaces',
        'simulation_states':'simulation_states',
        'excluded_states':'excluded_states',
        'my_simulation_number':'my_simulation_number',
        'score_table':'score_table',
        'score_calculation_callback':'score_calculation_callback',
        'score_column':'score',
        'action_sequence':None,
        'action_thresholds_min':0.25,
        'action_thresholds_max':0.8,
        'simulation_iterations':30,
        'log_level': 'debug',  # Log level for the logger
        'log_init_pass': False,  # If True, skip logger initialization
    }
