import gym
from modular_rl.settings import AgentSettings
from modular_rl.agents.mcts import AgentMCTS


def init_mcts_modular():
    env = gym.make('CartPole-v0')
    setting = AgentSettings.default_mcts_modular
    setting['log_level'] = 'verb'
    mcts_agent = AgentMCTS(
        env=env, setting=setting)

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


init_mcts_modular()
