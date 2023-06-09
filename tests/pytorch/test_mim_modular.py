# -*- coding: utf-8 -*-
from modular_rl.agents.mim import AgentMIM
from modular_rl.settings import AgentSettings
from modular_rl.envs.mim import EnvMIM


def init_mim_modular():
    setting = AgentSettings.default_mim_modular
    setting['log_level'] = 'verb'
    env = EnvMIM()
    agent = AgentMIM(env=env, setting=setting)

    agent.reset()
    state = agent.learn_reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    agent.update_step(reward, done)
    agent.learn_check()


init_mim_modular()
