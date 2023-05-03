
# -*- coding: utf-8 -*-
import gym
from modular_rl.networks import PolicyNetwork, ValueNetwork
import torch.optim as optim
import torch
from torch.distributions import Categorical
import numpy as np

class AgentPPO:
    def __init__(self, env, setting):
        # 환경 준비
        self.env = env if env else gym.make('CartPole-v0')
        self.setting = setting
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # 신경망 인스턴스 및 옵티마이저 생성
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim,setting.get('networks', 'middle'))
        self.value_net = ValueNetwork(self.state_dim,setting.get('networks', 'middle'))
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=setting.get('optimizer_speed', 3e-4))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=setting.get('optimizer_speed', 3e-4))

        # 학습 파라미터 설정
        self.max_episodes = setting.get('max_episodes', 30)
        self.max_timesteps = setting.get('max_timesteps',200)
        self.update_timestep = setting.get('update_timestep',200)
        self.ppo_epochs = setting.get('ppo_epochs',4)
        self.mini_batch_size = setting.get('mini_batch_size',64)
        self.gamma = setting.get('gamma',0.99)
        self.lam = setting.get('lam',0.95)
        self.clip_param = setting.get('clip_param',0.2)

        self.state = None

    # PPO 알고리즘 구현
    def compute_advantages(self, rewards, values, done, gamma=0.99, lam=0.95):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - done[t]) - values[t]
            advantages[t] = delta + gamma * lam * last_advantage * (1 - done[t])
            last_advantage = advantages[t]
        return advantages

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size - 1, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist = Categorical(self.policy_net(state))
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (return_ - self.value_net(state)).pow(2).mean()
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def learn_step(self, state, timestep):
        # 기존 학습 코드를 이 함수에 넣고 필요한 부분만 수정합니다.

        state_num = len(state)
        if state_num==1:
            state = state  # Unpack the tuple
        elif state_num==2:
            state, _ = state  # Unpack the tuple

        state_tensor = torch.tensor(state, dtype=torch.float32)
        dist = Categorical(self.policy_net(state_tensor))
        action = dist.sample()

        step_output = self.env.step(action.item())
        step_output_num = len(step_output)

        if step_output_num==4:
            next_state, reward, is_done, _ = self.env.step(action.item())
        elif step_output_num==5:
            next_state, reward, is_done, _, _ = self.env.step(action.item())

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(is_done)
        self.log_probs.append(dist.log_prob(action))

        self.state = next_state
        self.total_reward += reward
        if timestep > 0:
            timestep += 1

        if self.update_timestep > 0 and timestep > 0 and (timestep % self.update_timestep == 0):
            self.update()

    def update(self):
        states_tensor = torch.tensor(np.array(self.states, dtype=np.float32))
        actions_tensor = torch.tensor(np.array(self.actions))
        rewards_tensor = torch.tensor(np.array(self.rewards, dtype=np.float32))
        next_states_tensor = torch.tensor(np.array(self.next_states, dtype=np.float32))
        done_tensor = torch.tensor(np.array(self.done, dtype=np.float32))
        log_probs_tensor = torch.stack(self.log_probs)

        values = self.value_net(states_tensor).detach().squeeze()
        next_values = self.value_net(next_states_tensor).detach().squeeze()

        advantages = self.compute_advantages(rewards_tensor.numpy(), np.append(values.numpy(), next_values[-1].item()), done_tensor.numpy(), self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

        returns = np.add(advantages[:-1], values[:-1])
        self.ppo_update(self.ppo_epochs, self.mini_batch_size, states_tensor, actions_tensor, log_probs_tensor, returns, advantages_tensor)

        self.reset()

    def learn(self):
        # 학습 루프
        timestep = 0
        rewards_sum = 0
        test = 0
        if self.max_episodes > 0 and self.max_timesteps > 0:
            for episode in range(self.max_episodes):
                state = self.env.reset()
                self.total_reward = 0

                self.reset()

                for t in range(self.max_timesteps):
                    self.learn_step(state, timestep)

                rewards_sum += self.total_reward
                print(f'Episode: {episode}, Total Reward: {self.total_reward}, {rewards_sum}')

            self.env.close()
        else:
            self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.done = []
        self.log_probs = []

    def learn_reset(self):
        self.state = self.env.reset()
        self.total_reward = 0

    def learn_next(self):
        self.learn_step(self.state, -1)

    def learn_close(self) :
        self.env.close()

    def learn_check(self):
        print(f'Total Reward: {self.total_reward}')

    def save(self, file_name):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, file_name)

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
