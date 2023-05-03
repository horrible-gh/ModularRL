class AgentSettings:
    default = {
        'max_episodes': 30,
        'max_timesteps' : 200,
        'update_timestep' : 2000,
        'ppo_epochs' : 4,
        'mini_batch_size':64,
        'networks':'medium',
        'optimizer_speed':3e-4,
        'gamma': 0.99,
        'lam':0.95,
        'clip_param':0.2
    }

    default_modular = {
        'max_episodes': 30,
        'max_timesteps' : -1,
        'update_timestep' : -1,
        'ppo_epochs' : 4,
        'mini_batch_size':64,
        'networks':'medium',
        'optimizer_speed':3e-4,
        'gamma': 0.99,
        'lam':0.95,
        'clip_param':0.2
    }
