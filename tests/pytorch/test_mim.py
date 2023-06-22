# -*- coding: utf-8 -*-
from modular_rl.agents.mim import AgentMIM
from modular_rl.settings import AgentSettings
from modular_rl.envs.mim import EnvMIM


def init_mim():
    setting = AgentSettings.default_mim
    setting['log_level'] = 'debug'
    agent = AgentMIM(env=EnvMIM(), setting=setting)
    agent.learn()


init_mim()
