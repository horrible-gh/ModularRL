# -*- coding: utf-8 -*-
from modular_rl.agents.ppo import AgentPPO
from modular_rl.settings import AgentSettings

'''
This code is a Python script that initializes and trains an instance of the AgentPPO class with default settings using the init_ppo() function.
Additionally, the init_ppo_modular() function is provided to demonstrate the usage of the AgentPPO class with modified settings.

In init_ppo_modular(), an instance of AgentPPO is created with a modified default_modular settings dictionary.
The function then calls various methods of the instance, such as reset(), learn_reset(), learn_next(), and update() to demonstrate the functionality of the class.
The instance is then saved to a file using the save_model() method, which is not shown in the provided code.

'''


def init_ppo():
    env = AgentPPO(env=None, setting=AgentSettings.default_ppo)
    env.learn()


init_ppo()
