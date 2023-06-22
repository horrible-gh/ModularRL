from modular_rl.agents.pytorch.mcts import PyTorchAgentMCTS
from modular_rl.agents.tensorflow.mcts import TensorFlowAgentMCTS


class AgentMCTS():
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
        self.env = env
        self.setting = setting

        learn_framework = self.setting.get(
            "learn_framework", self.LEARN_PYTORCH)

        if learn_framework == self.LEARN_PYTORCH:
            self.agent = PyTorchAgentMCTS(env, setting)
        elif learn_framework == self.LEARN_TENSORFLOW:
            pass

    def select_action(self, state):
        return self.agent.select_action(state)

    def backpropagate(self, search_path, reward, done):
        self.agent.backpropagate(search_path, reward, done)

    def learn(self):
        self.agent.learn()

    def train(self):
        self.agent.train()

    def compute_loss(self, state, action, reward, next_state, done):
        return self.agent.compute_loss(state, action, reward, next_state, done)

    def update(self):
        self.agent.update()

    def learn_close(self):
        self.agent.learn_close()

    def learn_check(self):
        self.agent.learn_check()

    def check_tensor(self, state):
        return self.agent.check_tensor(state)

    def save_model(self, file_name):
        self.agent.save_model(file_name)

    def save(self, file_name):
        self.agent.save(file_name)

    def load_model(self, file_name):
        self.agent.load_model(file_name)

    def load(self, file_name):
        self.agent.load(file_name)

    def update_step(self, state, action, reward, done, next_state):
        self.agent.update_step(state, action, reward, done, next_state)

    def reset(self):
        self.agent.reset()

    def learn_reset(self):
        return self.agent.learn_reset()
