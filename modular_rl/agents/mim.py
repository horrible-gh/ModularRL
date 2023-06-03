import random
import numpy as np
from LogAssist.log import Logger
from modular_rl.agents._custom import AgentCustom
from modular_rl.params.mim import ParamMIM


class AgentMIM(AgentCustom):
    def __init__(self, env, setting):
        super().__init__(env, setting)
        self.fixed_states_name = setting.get('fixed_states', ParamMIM.default['fixed_states'])
        self.unknown_spaces_name = setting.get('unknown_spaces', ParamMIM.default['unknown_spaces'])
        self.simulation_states_name = setting.get('simulation_states', ParamMIM.default['simulation_states'])
        self.excluded_states_name = setting.get('excluded_states', ParamMIM.default['excluded_states'])
        self.my_simulation_number = setting.get('my_simulation_number', ParamMIM.default['my_simulation_number'])
        self.score_table_name = setting.get('score_table', ParamMIM.default['score_table'])
        self.score_calculation_callback_name = setting.get('score_calculation_callback', ParamMIM.default['score_calculation_callback'])
        self.score_column = setting.get('score_column', ParamMIM.default['score_column'])
        self.simulation_iterations = setting.get('simulation_iterations', ParamMIM.default['simulation_iterations'])
        self.standard_deviation_average = 0
        self.standard_deviation_counts = 0
        self.skew_average = 0
        self.skew_counts = 0
        self.kurtosise_average = 0
        self.kurtosise_counts = 0


    def simulate(self,state) :
        fixed_states = state[self.fixed_states_name]
        unknown_spaces = state[self.unknown_spaces_name]
        simulation_states = state[self.simulation_states_name]
        excluded_states = state[self.excluded_states_name]
        score_calculation_callback = state[self.score_calculation_callback_name]
        score_table= state[self.score_table_name]
        remaining_states = set(simulation_states) - set(excluded_states)
        simulation_table = []

        Logger.verb('mim:simulate',f'{self.simulation_states_name},{list(remaining_states)}, {simulation_states}')


        simulation_totals = []
        simulation_averages = []
        simulation_iteration_indexes = []
        standard_deviations = []
        skews = []
        kurtosises = []

        if len(fixed_states) != len(unknown_spaces):
            return simulation_table

        for fixed_state in fixed_states:
            remaining_states -= set(fixed_state)

        for i, fixed_state in enumerate(fixed_states):
            simulation_table.append([0 for _ in score_table])
            simulation_iteration_indexes.append(0)
            for _ in range(self.simulation_iterations):
                simulation_iteration_indexes[i] += 1
                #Logger.verb('mim:simulate:fixed_states,unkonwn_spaces,remaining_states ',f'{fixed_states}, {unknown_spaces}, {list(remaining_states)}, {i}')
                sample_states = random.sample(list(remaining_states), unknown_spaces[i])
                simulated_result = fixed_state + sample_states
                #Logger.verb('mim:simulate',simulated_result)
                score_obj = score_calculation_callback(simulated_result)
                score = score_obj[self.score_column] if self.score_column else score_obj
                for idx in range(len(score_table)-1):
                    #Logger.verb('mim:simulate',f'{score}, {score_table}')
                    if score_table[idx] <= score <= score_table[idx+1]:
                        simulation_table[i][idx] += 1
                        break
                if unknown_spaces[i]==0:
                    break

        Logger.verb('mim:simulation_table',simulation_table)
        for p in range(len(simulation_table)):
            sim_result = 0
            for idx in range(len(simulation_table[p])):
                sim_result += simulation_table[p][idx] * idx
            simulation_totals.append(sim_result)
            simulation_averages.append(sim_result / simulation_iteration_indexes[p])
            standard_deviations.append(np.std(simulation_table[p]))
            skewness, kurtosis = self.calc_skewness_kurtosis(simulation_table[p])
            skews.append(skewness)
            kurtosises.append(kurtosis)

        Logger.verb('simulator:simulation_totals',simulation_totals)
        Logger.verb('simulator:simulation_averages',simulation_averages)
        Logger.verb('simulator:standard_deviations',standard_deviations)
        Logger.verb('simulator:skews',skews)
        Logger.verb('simulator:kurtosises',kurtosises)
        return simulation_iteration_indexes, simulation_averages, standard_deviations, skews, kurtosises


    def calc_skewness_kurtosis(self, data):
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data)

        skewness = (1/n) * sum((x - mean)**3 for x in data) / (std_dev**3)
        kurtosis = (1/n) * sum((x - mean)**4 for x in data) / (std_dev**4) - 3

        return skewness, kurtosis

    def state_analyze(self, simulation_iteration_indexes, simulation_averages, standard_deviations, skews, kurtosises):
        skip_myself = False
        if simulation_iteration_indexes[self.my_simulation_number] <=1:
            skip_myself = True

        my_simulation_average = simulation_averages[self.my_simulation_number]
        my_standard_deviation = standard_deviations[self.my_simulation_number]
        my_skew = skews[self.my_simulation_number]
        my_kurtosises = kurtosises[self.my_simulation_number]

        if skip_myself:
            del simulation_averages[self.my_simulation_number]
            del standard_deviations[self.my_simulation_number]
            del skews[self.my_simulation_number]
            del kurtosises[self.my_simulation_number]

        avg_standard_deviations = sum(standard_deviations) / len(standard_deviations)
        avg_skews = sum(skews) / len(skews)
        avg_kurtosises = sum(kurtosises) / len(kurtosises)


    def select_action(self, state):
        simulation_iteration_indexes, simulation_averages, standard_deviations, skews, kurtosises = self.simulate(state)



    def update_step(self, state, action, reward, done, next_state):
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

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

        if done:
            self.update()

    def train(self):
        self.learn()

    def learn(self):
        state = self.env.reset()
        Logger.verb('mim:learn:state', state)
        self.select_action(state)

    def save_model(self):
        pass
