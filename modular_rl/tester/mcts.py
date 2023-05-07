import gym
from modular_rl.settings import AgentSettings
from modular_rl.agents.mcts import AgentMCTS


def init_mcts():
    env = gym.make('CartPole-v0')
    setting = AgentSettings.default_mcts
    setting['num_simulations'] = 10
    setting['max_episodes'] = 10
    setting['log_level'] = 'verb'
    agent = AgentMCTS(env, setting)
    agent.train()
