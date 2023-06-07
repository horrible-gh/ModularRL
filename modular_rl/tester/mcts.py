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


def init_mcts_modular():
    setting = AgentSettings.default_mcts_modular
    setting['log_level'] = 'verb'
    mcts_agent = AgentMCTS(
        env=None, setting=setting)

    mcts_agent.reset()

    state = mcts_agent.learn_reset()
    state, action, reward, done = mcts_agent.select_action(state)
    next_state, reward, done, _, _ = mcts_agent.env.step(action)
    # Modular does not automatically generate the required values and stores them through update_step
    mcts_agent.update_step(state, action, reward, done, next_state)

    if done:
        mcts_agent.update()

    mcts_agent.learn_check()

    state = mcts_agent.learn_reset()
    state, action, reward, done = mcts_agent.select_action(state)
    next_state, reward, done, _, _ = mcts_agent.env.step(action)
    mcts_agent.update_step(state, action, reward, done, next_state)

    if done:
        mcts_agent.update()

    mcts_agent.learn_check()
