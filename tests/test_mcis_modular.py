import gym
from modular_rl.settings import AgentSettings
from modular_rl.agents.mcis import AgentMCIS


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


init_mcis_modular()
