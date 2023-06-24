# -*- coding: utf-8 -*-
import gym
from modular_rl.agents.ppo import AgentPPO
from modular_rl.settings import AgentSettings

'''
This code is a Python script that initializes and trains an instance of the AgentPPO class with default settings using the init_ppo() function.
Additionally, the init_ppo_modular() function is provided to demonstrate the usage of the AgentPPO class with modified settings.

In init_ppo_modular(), an instance of AgentPPO is created with a modified default_modular settings dictionary.
The function then calls various methods of the instance, such as reset(), learn_reset(), learn_next(), and update() to demonstrate the functionality of the class.
The instance is then saved to a file using the save_model() method, which is not shown in the provided code.

'''


def init_ppo_modular():
    env = gym.make('CartPole-v0')
    setting = AgentSettings.default_ppo_modular
    setting['learn_framework'] = 'tensorflow'
    agent = AgentPPO(env=env, setting=setting)
    agent.reset()
    agent.learn_reset()
    agent.learn_next()
    agent.learn_check()
    agent.learn_next()
    agent.learn_check()
    agent.update()

    agent.reset()
    ppo_manual_step(agent)
    agent.learn_check()
    ppo_manual_step(agent)
    agent.learn_check()
    agent.update()

    agent.learn_close()

    # env.save_model('test.pth')


def ppo_manual_step(agent=AgentPPO):
    initial_state = agent.learn_reset()
    action, _ = agent.select_action(initial_state)
    next_state = agent.learn_reset()
    agent.update_step(next_state, None, action, -1)


init_ppo_modular()
