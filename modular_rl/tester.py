# -*- coding: utf-8 -*-
from modular_rl.agents.agent_ppo import AgentPPO
from modular_rl.settings import AgentSettings

'''
This code is a Python script that initializes and trains an instance of the AgentPPO class with default settings using the init() function. 
Additionally, the init_modular() function is provided to demonstrate the usage of the AgentPPO class with modified settings.

In init_modular(), an instance of AgentPPO is created with a modified default_modular settings dictionary. 
The function then calls various methods of the instance, such as reset(), learn_reset(), learn_next(), and update() to demonstrate the functionality of the class. 
The instance is then saved to a file using the save_model() method, which is not shown in the provided code.

'''


def init():
    env = AgentPPO(env=None, setting=AgentSettings.default)
    env.learn()


def init_modular():
    env = AgentPPO(env=None, setting=AgentSettings.default_modular)
    env.reset()
    env.learn_reset()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
    env.learn_next()
    env.learn_check()
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

    # env.save_model('test.pth')
