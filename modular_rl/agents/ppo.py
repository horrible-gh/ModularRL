from modular_rl.agents.pytorch.ppo import PyTorchAgentPPO
from modular_rl.agents.tensorflow.ppo import TensorFlowAgentPPO


class AgentPPO():
    LEARN_PYTORCH = "pytorch"
    LEARN_TENSORFLOW = "tensorflow"

    def __init__(self, env, setting):
        """
        Initialize the AgentMCIS class with the specified environment and settings.

        :param env: The environment to use for training.
        :type env: gym.Env or None
        :param setting: The settings for the MCIS algorithm.
        :type setting: AgentSettings
        """
        if env:
            self.env = env
        self.setting = setting

        learn_framework = self.setting.get(
            "learn_framework", self.LEARN_PYTORCH)

        if learn_framework == self.LEARN_PYTORCH:
            self.agent = PyTorchAgentPPO(env, setting)
        elif learn_framework == self.LEARN_TENSORFLOW:
            pass

    def compute_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        return self.agent.compute_advantages(rewards, values, dones, gamma, lam)

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages):
        self.agent.ppo_iter(mini_batch_size, states, actions,
                            log_probs, returns, advantages)

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        self.agent.ppo_update(ppo_epochs, mini_batch_size, states,
                              actions, log_probs, returns, advantages, clip_param)

    def select_action(self, state):
        return self.agent.select_action(state)

    def learn_step(self, state, timestep):
        return self.agent.learn_step(state, timestep)

    def update_step(self, state, dist, action, timestep, auto_step=True, is_done=False, reward=0, next_state=None):
        return self.agent.update_step(state, dist, action, timestep, auto_step, is_done, reward, next_state)

    def learn(self):
        self.agent.learn()

    def train(self):
        self.agent.train()

    def update(self):
        self.agent.update()

    def learn_next(self):
        return self.agent.learn_next()

    def learn_close(self):
        self.agent.learn_close()

    def learn_check(self):
        self.agent.learn_check()

    def save_model(self, file_name):
        self.agent.save_model(file_name)

    def save(self, file_name):
        self.agent.save(file_name)

    def load_model(self, file_name):
        self.agent.load_model(file_name)

    def load(self, file_name):
        self.agent.load(file_name)

    def reset(self):
        self.agent.reset()

    def learn_reset(self):
        return self.agent.learn_reset()
