
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from modular_rl.networks.actor_critic import ActorCriticNetwork
from modular_rl.util.node import Node
from LogAssist.log import Logger


class AgentMCTS:
    def __init__(self, env, setting):
        # Environment
        self.env = env
        self.setting = setting
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Neural Network
        self.network = ActorCriticNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=setting.get('optimizer_speed', 3e-4))

        # MCTS parameters
        self.num_simulations = setting.get('num_simulations', 800)
        self.cpuct = setting.get('cpuct', 1.0)
        self.temperature = setting.get('temperature', 1.0)
        self.gamma = setting.get('gamma', 0.99)

        # Training parameters
        self.max_episodes = setting.get('max_episodes', 1000)
        self.max_timesteps = setting.get('max_timesteps', 1000)

        self.device = setting.get('device', None)
        if self.device == None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        # Episode parameters
        self.total_reward = 0
        self.episode = 0

        # Logger initialize
        self.log_level = setting.get('log_level', 'debug')
        Logger.init(self.log_level, None, None, None, True, False)

    def select_action(self, state):
        """
        Select an action using MCTS.

        :param state: The current state.
        :type state: numpy.ndarray
        :return: The selected action.
        :rtype: int
        """
        state = self._check_state(state)
        state_tensor = torch.FloatTensor(state)
        action_probs, _ = self.network(state_tensor)
        action_probs = action_probs.detach().numpy().flatten()
        root = Node(state, action_probs)
        Logger.verb(
            'agents:mcts:select_action:self.num_simulations', self.num_simulations)
        Logger.verb(
            'agents:mcts:select_action:root', root)
        Logger.verb(
            'agents:mcts:select_action:action_probs', action_probs)
        Logger.verb('agents:mcts:select_action:node', f'Root node: {root}')
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            # Logger.verb(
            #    'agents:mcts:select_action:search_path', search_path)
            # Selection
            while node.expanded():
                action, node = node.select_child(self.cpuct)
                search_path.append(node)
            Logger.verb('agents:mcts:select_action:After selection',
                        f' {search_path[-1]}')

            # Expansion
            if len(search_path) > 1:
                parent, action = search_path[-2], search_path[-1].action
            else:
                # or some other suitable defaults
                parent, action = search_path[0], None

            # Logger.verb('mcts:select_action:action', action)
            step_output = self.env.step(action) if action is not None else (
                parent.state, 0, False, None)
            step_output_num = len(step_output)

            if step_output_num == 4:
                state, reward, done, _ = step_output
            elif step_output_num == 5:
                state, reward, done, _, _ = step_output

            if not done:
                # Logger.verb(
                #    'agents:mcts:select_action:state', state)
                if not torch.is_tensor(state):
                    state_tensor = torch.from_numpy(
                        state).float().to(self.device)
                else:
                    state_tensor = state.to(self.device)

                action_probs, value = self.network(state_tensor)
                action_space = self.env.action_space.n
                node.expand(action_space, action_probs)

            # Backpropagation
            self.backpropagate(search_path, reward, done)

        chosen_action = root.select_action(self.temperature)
        chosen_action_reward = root.children[chosen_action].total_value
        chosen_action_state = root.children[chosen_action].state
        # Assuming that a non-zero reward indicates a terminal state
        done = (chosen_action_reward != 0)
        return chosen_action_state, chosen_action, chosen_action_reward, done

    def backpropagate(self, search_path, reward, done):
        """
        Backpropagate the value estimates back to the root node.

        :param search_path: The nodes visited during the search.
        :type search_path: list of Node
        :param reward: The reward obtained after the search.
        :type reward: float
        :param done: Whether the episode has ended.
        :type done: bool
        """
        for node in reversed(search_path):
            node.update_stats(reward)
            if not done:
                _, reward = self.network(node.state)
                reward = reward.item()

    def train(self):
        """
        Train the agent.
        """
        self.total_reward = 0
        for episode in range(self.max_episodes):
            state = self.env.reset()

            for t in range(self.max_timesteps):
                state = self._check_state(state)
                state_tensor = torch.FloatTensor(state).squeeze(0)
                next_state, action, reward, done = self.select_action(
                    state_tensor)

                # step_output = self.env.step(action.item())
                # step_output_num = len(step_output)
#
                # if step_output_num == 4:
                #    next_state, reward, done, _ = step_output
                # elif step_output_num == 5:
                #    next_state, reward, done, _, _ = step_output

                self.total_reward += reward
                Logger.debug(
                    'agents:mcts:train', f"Episode: {self.episode}, Reward: {reward},  Total Reward: {self.total_reward}")

                if done:
                    break

                # Update the network
                self.optimizer.zero_grad()
                actor_loss, critic_loss = self.compute_loss(
                    state_tensor, action, reward, next_state, done)
                loss = actor_loss + critic_loss
                loss.backward()
                self.optimizer.step()

                state = next_state

            self.episode += 1
            # print(f"Episode: {self.episode}, Reward: {self.total_reward}")

    def _check_state(self, state):
        Logger.verb('agents:mcts:_check_state', state)
        state_num = len(state)
        if state_num == 2:
            state, _ = state  # Unpack the tuple
        return state

    def compute_loss(self, state, action, reward, next_state, done):
        # Predict action probabilities and values
        action_probs, values = self.network(state)

        # Compute the value loss
        target_values = reward + self.gamma * \
            self.network(next_state)[1] * (1 - done)
        critic_loss = F.mse_loss(values, target_values.detach())

        # Compute the policy loss
        m = Categorical(action_probs)
        logprobs = m.log_prob(action)
        actor_loss = -logprobs * (target_values - values).detach()

        return actor_loss, critic_loss
