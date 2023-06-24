import gym
from modular_rl.settings import AgentSettings
from modular_rl.agents.mcis import AgentMCIS


def init_mcis():
    env = gym.make('CartPole-v0')
    setting = AgentSettings.default_mcis
    setting['num_simulations'] = 10
    setting['max_episodes'] = 10
    setting['log_level'] = 'verb'
    agent = AgentMCIS(env, setting)
    agent.train()


init_mcis()
