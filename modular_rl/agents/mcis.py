from modular_rl.agents.pytorch.mcis import PyTorchAgentMCIS
from modular_rl.agents.tensorflow.mcis import TensorFlowAgentMCIS


class AgentMCIS():
    def __init__(self, env, setting):
        """
        Initialize the AgentMCIS class with the specified environment and settings.

        :param env: The environment to use for training.
        :type env: gym.Env or None
        :param setting: The settings for the MCIS algorithm.
        :type setting: AgentSettings
        """

        if setting['learn_framework'] == 'pytorch':
            self.agent = PyTorchAgentMCIS(env, setting)
        elif setting['learn_framework'] == 'tensorflow':
            pass
