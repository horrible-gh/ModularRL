import gym
from modular_rl.settings import AgentSettings
from modular_rl.agents.mcis import AgentMCIS


def init_mcis():
    env = gym.make('CartPole-v0')
    setting = AgentSettings.default_mcts
    setting['num_simulations'] = 10
    setting['max_episodes'] = 10
    setting['log_level'] = 'verb'
    agent = AgentMCIS(env, setting)
    agent.train()


def init_mcis_modular():
    mcis_agent = AgentMCIS(
        env=None, setting=AgentSettings.default_mcis_modular)

    mcis_agent.reset()

    state = mcis_agent.learn_reset()
    action = mcis_agent.select_action(state)
    next_state = mcis_agent.learn_reset()
    mcis_agent.update_step(state, action, 1, False, next_state)
    mcis_agent.learn_check()

    state = mcis_agent.learn_reset()
    action = mcis_agent.select_action(state)
    next_state = mcis_agent.learn_reset()
    mcis_agent.update_step(state, action, 1, True, next_state)
    mcis_agent.learn_check()

    # update_step function automatically updates if done is given a True option
    # mcis_agent.update()
