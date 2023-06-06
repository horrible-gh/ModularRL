import random
import numpy as np
from LogAssist.log import Logger
from modular_rl.agents._custom import AgentCustom
from modular_rl.params.mim import ParamMIM


class AgentMIM(AgentCustom):
    def __init__(self, env, setting):
        super().__init__(env, setting)
        self.fixed_states_name = 'fixed_states'
        self.unknown_spaces_name = 'unknown_spaces'
        self.simulation_states_name = 'simulation_states'
        self.excluded_states_name = 'excluded_states'
        self.my_simulation_number_name = 'my_simulation_number'
        self.score_table_name = 'score_table'
        self.score_calculation_callback_name = 'score_calculation_callback'
        self.score_column = setting.get(
            'score_column', ParamMIM.default['score_column'])
        self.simulation_iterations = setting.get(
            'simulation_iterations', ParamMIM.default['simulation_iterations'])
        self.judgement_flexibility = setting.get(
            'judgement_flexibility', ParamMIM.default['judgement_flexibility'])
        self.simulation_iteration_indexes_count = 0
        self.standard_deviations_avg = 0
        self.skews_avg = 0
        self.kurtosises_avg = 0
        self.reset()

    def simulate(self, state):
        required_state_elements = {
            self.fixed_states_name: 'fixed_state',
            self.unknown_spaces_name: 'unknown_spaces',
            self.simulation_states_name: 'simulation_states',
            self.excluded_states_name: 'excluded_states',
            self.score_calculation_callback_name: 'score_calculation_callback',
            self.my_simulation_number_name: 'my_simulation_number',
            self.score_table_name: 'score_table',
        }

        for element_key, error_obj in required_state_elements.items():
            if state.get(element_key) is None:
                raise Exception(
                    f"'{error_obj}' not found in the provided state for simulation.")

        fixed_states = state[self.fixed_states_name]
        unknown_spaces = state[self.unknown_spaces_name]
        simulation_states = state[self.simulation_states_name]
        excluded_states = state[self.excluded_states_name]
        score_calculation_callback = state[self.score_calculation_callback_name]
        self.my_simulation_number = state[self.my_simulation_number_name]
        score_table = state[self.score_table_name]
        remaining_states = set(simulation_states) - set(excluded_states)
        simulation_table = []

        Logger.verb(
            'mim:simulate', f'{self.simulation_states_name},{list(remaining_states)}, {simulation_states}')

        self.reset()

        if len(fixed_states) != len(unknown_spaces):
            return simulation_table

        for fixed_state in fixed_states:
            remaining_states -= set(fixed_state)

        for i, fixed_state in enumerate(fixed_states):
            simulation_table.append([0 for _ in score_table])
            self.simulation_iteration_indexes.append(0)
            for _ in range(self.simulation_iterations):
                self.simulation_iteration_indexes[i] += 1
                # Logger.verb('mim:simulate:fixed_states,unkonwn_spaces,remaining_states ',f'{fixed_states}, {unknown_spaces}, {list(remaining_states)}, {i}')
                sample_states = random.sample(
                    list(remaining_states), unknown_spaces[i])
                simulated_result = fixed_state + sample_states
                # Logger.verb('mim:simulate',simulated_result)
                score_obj = score_calculation_callback(simulated_result)
                score = score_obj[self.score_column] if self.score_column else score_obj
                for idx in range(len(score_table)-1):
                    # Logger.verb('mim:simulate',f'{score}, {score_table}')
                    if score_table[idx] <= score <= score_table[idx+1]:
                        simulation_table[i][idx] += 1
                        break
                if unknown_spaces[i] == 0:
                    break

        Logger.verb('mim:simulation_table', simulation_table)
        for index in range(len(simulation_table)):
            sim_result = 0
            for sub_index in range(len(simulation_table[index])):
                sim_result += simulation_table[index][sub_index] * \
                    sub_index + 0.001
            self.simulation_totals.append(sim_result)
            self.simulation_averages.append(
                sim_result / self.simulation_iteration_indexes[index])
            self.standard_deviations.append(np.std(simulation_table[index]))
            skewness, kurtosis = self.calc_skewness_kurtosis(
                simulation_table[index])
            self.skews.append(skewness)
            self.kurtosises.append(kurtosis)

        Logger.verb('simulator:simulation_totals', self.simulation_totals)
        Logger.verb('simulator:simulation_averages', self.simulation_averages)
        Logger.verb('simulator:standard_deviations', self.standard_deviations)
        Logger.verb('simulator:skews', self.skews)
        Logger.verb('simulator:kurtosises', self.kurtosises)

    def calc_skewness_kurtosis(self, data):
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data)

        skewness = (1/n) * sum((x - mean)**3 for x in data) / (std_dev**3)
        kurtosis = (1/n) * sum((x - mean)**4 for x in data) / (std_dev**4) - 3

        return skewness, kurtosis

    def rank_array(self, array):
        temp = array.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = len(array) - (np.arange(len(array)))

        return ranks

    def calculate_weight_adjustment_factors(self, simulation_size):
        weight_adjustment_factors = [1] * simulation_size
        factors = [0.125, 0.075, 0.05]
        averages = [self.standard_deviations_avg,
                    self.skews_avg, self.kurtosises_avg]
        values = [self.standard_deviations, self.skews, self.kurtosises]

        for index in range(simulation_size):
            for factor, average, value in zip(factors, averages, values):
                if average < value[index]:
                    weight_adjustment_factor = factor * average / value[index]
                    weight_adjustment_factors[index] += weight_adjustment_factor - factor
        return weight_adjustment_factors

    def update_averages(self, total_size):
        factors = [self.standard_deviations_avg,
                   self.skews_avg, self.kurtosises_avg]
        values = [self.standard_deviations, self.skews, self.kurtosises]
        for factor, value in zip(factors, values):
            factor = (
                factor * self.simulation_iteration_indexes_count + sum(value)) / total_size
        self.simulation_iteration_indexes_count = total_size

    def calculate_action_weights(self, adjusted_averages, skip_myself):
        Logger.verb('mim:calculate_action_weights:adjusted_averages',
                    adjusted_averages)
        Logger.verb('mim:calculate_action_weights:my_simulation_average',
                    self.my_simulation_average)
        averages_table = []
        if skip_myself:
            averages_table.insert(0, self.my_simulation_average)
        averages_table.extend(adjusted_averages)
        Logger.verb('mim:calculate_action_weights:averages_table',
                    averages_table)
        ranks = self.rank_array(np.array(averages_table))
        Logger.verb('mim:calculate_action_weights:ranks', f'{ranks}')

        base_weight = int(1 / self.env.action_space * 100)
        action_scale = self.env.action_space / len(ranks)
        action_weights = [base_weight for _ in range(self.env.action_space)]
        check_weights = [False for _ in range(self.env.action_space)]

        Logger.verb('mim:calculate_action_weights:check_base_weight',
                    f'{base_weight},{action_scale},{action_weights}')

        my_rank = ranks[0]
        my_average = self.my_simulation_average
        my_action_rank = my_rank
        my_action_rank = self.env.action_space - \
            round(my_rank * action_scale)

        for rank_idx in range(len(ranks)):
            Logger.verb('mim:calculate_action_weights:rank_idx',
                        rank_idx)
            if rank_idx != 0:
                rank_diff = abs(my_rank - ranks[rank_idx])
                Logger.verb('mim:calculate_action_weights:rank_diff',
                            rank_diff)
                action_num = self.env.action_space - \
                    round(ranks[rank_idx] * action_scale)
                Logger.verb('mim:calculate_action_weights:action_num',
                            action_num)
                if my_rank < ranks[rank_idx]:
                    average_weight = (averages_table[action_num] /
                                      my_average) * (0.5 ** rank_diff)
                else:
                    average_weight = (my_average /
                                      averages_table[action_num]) * (0.5 ** rank_diff)

                Logger.verb('mim:calculate_action_weights:scale,num',
                            f'{action_scale}, {action_num},{average_weight}')

                if check_weights[action_num] == False:
                    check_weights[action_num] = True
                    expect_weight = action_weights[action_num] * \
                        average_weight
                    weight_diff = action_weights[action_num] - expect_weight
                    Logger.verb('mim:calculate_action_weights:weight_diff',
                                f'{weight_diff}, {expect_weight}')
                    action_weights[action_num] -= weight_diff
                    if action_weights[action_num] < 1:
                        action_weights[action_num] = 1
                    action_weights[my_action_rank] += weight_diff
                    Logger.verb('mim:calculate_action_weights:scale,num',
                                f'{action_num}, {action_weights[action_num]}, {my_action_rank}, {action_weights[my_action_rank]}, {average_weight}')

        for action_num in range(len(action_weights)):
            if action_weights[action_num] == 0:
                prev_action_weights = action_weights[action_num-1]
                next_action_weights = action_weights[action_num+1]
                middle_weight = (prev_action_weights + next_action_weights) / 2
                action_weights[action_num] = round(middle_weight)

        Logger.verb('mim:calculate_action_weights:action_weights',
                    action_weights)
        return action_weights

    def calculate_weights(self):
        skip_myself = False
        skip_count = 0

        if self.simulation_iteration_indexes[self.my_simulation_number] <= 1:
            skip_myself = True
            skip_count = 1

        self.my_simulation_average = self.simulation_averages[self.my_simulation_number]

        if skip_myself:
            del self.simulation_iteration_indexes[self.my_simulation_number]
            del self.simulation_averages[self.my_simulation_number]
            del self.standard_deviations[self.my_simulation_number]
            del self.skews[self.my_simulation_number]
            del self.kurtosises[self.my_simulation_number]

        simulation_size = len(self.simulation_averages)
        simulation_total_size = self.simulation_iteration_indexes_count + simulation_size

        self.update_averages(simulation_total_size)
        weight_adjustment_factors = self.calculate_weight_adjustment_factors(
            simulation_size)

        adjusted_averages = [average * factor for average,
                             factor in zip(self.simulation_averages, weight_adjustment_factors)]
        action_weights = self.calculate_action_weights(
            adjusted_averages, skip_myself)

        return action_weights

    def select_action(self, state):
        action_list = []
        if isinstance(self.env.action_space, list):
            action_table = self.env.action_space
        else:
            action_table = list(range(self.env.action_space))
        self.env.action_space = len(action_table)
        self.simulate(state)
        action_weights = self.calculate_weights()
        Logger.verb('mim:select_action', action_weights)
        for action_index in range(len(action_weights)):
            for _ in range(int(action_weights[action_index])):
                action_list.append(action_index)

        random.shuffle(action_list)
        choice = random.choice(action_list)
        return action_table[choice]

    # def update_step(self, state, action, reward, done, next_state):
    def update_step(self, reward, done):
        """
        Updates the provided state, action, reward, done, and next_state.
        :param state: The current state of the environment.
        :type state: numpy.ndarray
        :param action: The action taken by the agent.
        :type action: int
        :param reward: The reward for the current step.
        :type reward: float
        :param done: Flag to mark if the episode is done or not.
        :type done: bool
        :param next_state: The next state after the current action.
        :type next_state: numpy.ndarray
        """
        self.update_reward(reward)
        # self.states.append(state)
        # self.actions.append(action)
        # self.rewards.append(reward)
        # self.dones.append(done)
        # self.next_states.append(next_state)
        if done:
            self.update()

    def update(self):
        pass

    def reset(self):
        self.simulation_totals = []
        self.simulation_iteration_indexes = []
        self.simulation_averages = []
        self.standard_deviations = []
        self.skews = []
        self.kurtosises = []

        # self.

    def train(self):
        self.learn()

    def learn(self):
        """
        Train the agent.
        """
        self.total_reward = 0
        for episode in range(self.max_episodes):
            self.episode = episode
            state = self.learn_reset()
            for t in range(self.max_timesteps):
                state = self._check_state(state)

                # Select an action
                action = self.select_action(state)
                Logger.verb('agents:mim:learn:action', action)
                # Take a step in the environment
                next_state, reward, done = self.env.step(action)
                # Update the agent's experience
                self.update_step(reward, done)
                # Update the network parameters at the end of the episode
                if done:
                    self.update()
                    self.learn_check()
                    break
                state = next_state
            self.episode += 1

    def load(self, file_name):
        self.load_model(file_name)

    def load_model(self, file_name):
        data_arr = np.load(file_name)
        self.simulation_iteration_indexes_count = data_arr[0]
        self.standard_deviations_avg = data_arr[1]
        self.skews_avg = data_arr[2]
        self.kurtosises_avg = data_arr[3]

    def save(self, file_name):
        self.save_model(file_name)

    def save_model(self, file_name):
        data_arr = [
            self.simulation_iteration_indexes_count,
            self.standard_deviations_avg,
            self.skews_avg,
            self.kurtosises_avg,
        ]
        np.save(file_name, data_arr)
