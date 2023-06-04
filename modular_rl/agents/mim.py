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
        for p in range(len(simulation_table)):
            sim_result = 0
            for idx in range(len(simulation_table[p])):
                sim_result += simulation_table[p][idx] * idx
            self.simulation_totals.append(sim_result)
            self.simulation_averages.append(
                sim_result / self.simulation_iteration_indexes[p])
            self.standard_deviations.append(np.std(simulation_table[p]))
            skewness, kurtosis = self.calc_skewness_kurtosis(
                simulation_table[p])
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

    def rank_array(array):
        temp = array.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = len(array) - (np.arange(len(array)))

        return ranks

    def state_analyze(self):
        skip_myself = False
        skip_count = 0
        Logger.verb('mim:state_analyze',
                    f'{self.simulation_iteration_indexes}, {self.my_simulation_number}')
        if self.simulation_iteration_indexes[self.my_simulation_number] <= 1:
            skip_myself = True
            skip_count = 1

        reliability = [1 for p in range(len(
            self.simulation_iteration_indexes) - skip_count)]

        simulation_after_averages = [0 for p in range(len(
            self.simulation_iteration_indexes) - skip_count)]

        action_weight = [0 for weight in range(self.env.action_space)]

        my_simulation_number = self.my_simulation_number
        my_simulation_average = self.simulation_averages[my_simulation_number]
        my_simulation_after_average = 0
        my_standard_deviation = self.standard_deviations[my_simulation_number]
        my_skew = self.skews[my_simulation_number]
        my_kurtosises = self.kurtosises[my_simulation_number]
        my_rank = len(self.simulation_averages)

        if skip_myself:
            del self.simulation_iteration_indexes[my_simulation_number]
            del self.simulation_averages[my_simulation_number]
            del self.standard_deviations[my_simulation_number]
            del self.skews[my_simulation_number]
            del self.kurtosises[my_simulation_number]

        simulation_size = len(self.simulation_averages)
        simulation_total_size = self.simulation_iteration_indexes_count + simulation_size

        self.standard_deviations_avg = (self.standard_deviations_avg * self.simulation_iteration_indexes_count + sum(
            self.standard_deviations)) / simulation_total_size

        self.skews_avg = (self.skews_avg * self.simulation_iteration_indexes_count + sum(
            self.skews)) / simulation_total_size

        self.kurtosises_avg = (self.kurtosises_avg * self.simulation_iteration_indexes_count + sum(
            self.kurtosises)) / simulation_total_size

        self.simulation_iteration_indexes_count = simulation_total_size

        for p in range(simulation_size):
            if self.standard_deviations_avg < self.standard_deviations[p]:
                reliability[p] -= 0.25
                reliability[p] += 0.25 * self.standard_deviations_avg / \
                    self.standard_deviations[p]
            if self.skews_avg < self.skews[p]:
                reliability[p] -= 0.15
                reliability[p] += 0.15 * self.skews_avg / self.skews[p]
            if self.kurtosises_avg < self.kurtosises[p]:
                reliability[p] -= 0.1
                reliability[p] += 0.1 * \
                    self.kurtosises_avg / self.kurtosises[p]
            simulation_after_averages[p] = self.simulation_averages[p] * \
                reliability[p]

        if skip_myself:
            my_simulation_after_average = my_simulation_average
        else:
            my_simulation_after_average = simulation_after_averages[my_simulation_number]
            del simulation_after_averages[my_simulation_number]

        avg_list = []
        avg_list.append(my_simulation_after_average)
        avg_list.extend(simulation_after_averages)
        ranks = self.rank_array(avg_list)
        action_scale = self.env.action_space / len(ranks)
        action_weight = [0 for action in range(self.env.action_space)]

        my_rank = ranks[0]
        action_weight[len(avg_list) - int(my_rank * action_scale) -
                      1] = int(my_simulation_after_average * 10)
        my_weight = int(my_simulation_after_average * 10)

        for p in range(len(avg_list)):
            if skip_myself == False and p == 0:
                continue
            if my_rank < ranks[p]:
                weight = avg_list[p] / my_simulation_after_average * my_weight
            else:
                weight = my_simulation_after_average / avg_list[p] * my_weight
            action_weight[len(avg_list) - int(ranks[p] *
                                              action_scale) - 1] = weight

        for action_num in range(len(action_weight)):
            if action_weight[action_num] == 0:
                prev_action_weight = action_weight[action_num-1]
                next_action_weight = action_weight[action_num+1]
                middle_weight = (prev_action_weight + next_action_weight) / 2
                action_weight[action_num] = middle_weight
        return action_weight

        # for p in range(simulation_size):
        #    if skip_myself == False and p == my_simulation_number:
        #        continue
        #    if my_simulation_after_average + (my_simulation_after_average * self.judgement_flexibility) > simulation_after_averages[p]:
        #        my_rank -= 1
        # if my_rank == (simulation_size + skip_count):
        #    return 0
        # elif my_rank == 1:
        #    return 1
        # else:
        #    return (1 / simulation_size) * ((simulation_size + skip_count) - my_rank)

    def select_action(self, state):
        if isinstance(self.env.action_space, list):
            action_table = self.env.action_space
        else:
            action_table = [action for action in range(self.env.action_space)]
        self.env.action_space = len(action_table)
        self.simulate(state)
        action_weight = self.state_analyze()

        # choice = random.choice(action_table)
        # select_skew_random = 0
        # for i in range(len(action_table)):
        #    if action_table[i] == choice:
        #        select_skew_random = i
        #        break
        # action_idx = round((len(action_table) - 1) * select_weight)
        # action_middle = int(len(action_table) / 2)
        # if action_middle < select_skew_random:
        #    if select_weight != 0 and select_weight != 1:
        #        action_idx += 1
        return action_table[action_idx]

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
