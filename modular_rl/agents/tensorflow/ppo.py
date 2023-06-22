import tensorflow as tf
from modular_rl.agents.tensorflow._agent import Agent


class TensorFlowAgentPPO(Agent):
    def __init__(self, env, setting):
        """
        Initialize the AgentMCIS class with the specified environment and settings.

        :param env: The environment to use for training.
        :type env: gym.Env or None
        :param setting: The settings for the MCIS algorithm.
        :type setting: AgentSettings
        """

        super().__init__(env, setting)
        super().init_actor_critic()

        # mcis parameters
        self.num_simulations = setting.get('num_simulations', 800)
        self.cpuct = setting.get('cpuct', 1.0)
        self.temperature = setting.get('temperature', 1.0)

        self.device = setting.get('device', None)
        if self.device == None:
            # TensorFlow automatically uses the GPU if one is available, otherwise it uses the CPU.
            # Therefore, there is no need to manually set the device as in PyTorch.
            pass
