# -*- coding: utf-8 -*-
from modular_rl.agents.agent_ppo import AgentPPO
from modular_rl.settings import AgentSettings

def init():
    env = AgentPPO(env=None, setting=AgentSettings.default)
    env.learn()

def init_manual():
    env = AgentPPO(env=None, setting=AgentSettings.default_modular)
    env.reset()
    env.learn_reset()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_check()
    env.update()

    env.reset()
    env.learn_reset()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_next()
    env.learn_next()
    env.learn_check()
    env.update()

    env.learn_close()

    #env.save_model('test.pth')
