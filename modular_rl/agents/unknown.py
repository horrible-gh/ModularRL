from modular_rl.agents._custom import AgentCustom
from modular_rl.params.unknown import SimulationParameters
import random
from scipy.stats import skew, kurtosis
import numpy as np

class AgentSimulator(AgentCustom):
    def __init__(self, env, setting):
        super().__init__(env, setting)
        self.fixed_states = setting.get('fixed_states', SimulationParameters.default['fixed_states'])
        self.unknown_spaces = setting.get('unknown_spaces', SimulationParameters.default['unknown_spaces'])
        self.simulation_states = setting.get('simulation_states', SimulationParameters.default['simulation_states'])
        self.excluded_states = setting.get('excluded_states', SimulationParameters.default['excluded_states'])
        self.my_simulation_number = setting.get('my_simulation_number', SimulationParameters.default['my_simulation_number'])
        self.score_table = setting.get('score_table', SimulationParameters.default['score_table'])
        self.score_calculation_callback = setting.get('score_calculation_callback', SimulationParameters.default['score_calculation_callback'])
        self.score_column = setting.get('score_column', SimulationParameters.default['score_column'])
        self.simulation_iterations = setting.get('simulation_iterations', SimulationParameters.default['simulation_iterations'])



    def simulate(self,state) :
        fixed_states = state[self.fixed_states]
        unknown_spaces = state[self.unknown_spaces]
        simulation_states = state[self.simulation_states]
        excluded_states = state[self.excluded_states]
        remaining_states = set(simulation_states) - set(excluded_states)
        simulation_table = []

        simulation_total = []
        standard_deviations = []
        skews = []
        kurtosises = []

        if len(fixed_states) != len(unknown_spaces):
            return simulation_table

        for fixed_state in fixed_states:
            remaining_states -= set(fixed_state)

        for i, fixed_state in enumerate(fixed_states):
            for _ in range(self.simulation_iterations):
                sample_states = random.sample(list(remaining_states), unknown_spaces[i])
                simulated_result = fixed_state + sample_states
                score_obj = self.score_calculation_callback(simulated_result)
                score = score_obj[self.score_column] if self.score_column else score_obj
                for idx in range(len(self.score_table)-1):
                    if self.score_table[idx] <= score <= self.score_table[idx+1]:
                        simulation_table.append([0 for _ in self.score_table])
                        simulation_table[i][idx] += 1
                    if unknown_spaces[i]==0:
                        break
        for p in simulation_table:
            sim_result = 0
            for idx in simulation_table[p]:
                sim_result += simulation_table[p][idx] * idx
            simulation_total.append(sim_result)
            standard_deviations.append(np.std(simulation_table[p]))
            skews.append(skew(simulation_table[p]))
            kurtosises.append(kurtosis(simulation_table[p]))
        return None


    def select_action(self, state):
        self.simulate(state)


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
