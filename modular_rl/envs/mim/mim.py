import random
from modular_rl.envs._custom import CustomEnv
from modular_rl.envs.mim.card_evaluator import CardEvaluator
from LogAssist.log import Logger
import copy


class EnvMIM(CustomEnv):
    def __init__(self) -> None:
        self.players = 4
        self.fixed_states = []
        self.unknown_spaces = [0, 3, 3, 3]
        self.simulation_states = []
        self.excluded_states = []
        self.risk_table = [0, 0, 0]
        self.score_table = [
            CardEvaluator.NO_PAIR_BASE,
            CardEvaluator.ONE_PAIR_BASE,
            CardEvaluator.TWO_PAIR_BASE,
            CardEvaluator.THREE_OF_A_KIND_BASE,
            CardEvaluator.STRAIGHT_BASE,
            CardEvaluator.BACK_STRAIGHT_BASE,
            CardEvaluator.MOUNTAIN_BASE,
            CardEvaluator.FLUSH_BASE,
            CardEvaluator.FULL_HOUSE_BASE,
            CardEvaluator.FOUR_OF_A_KIND_BASE,
            CardEvaluator.STRAIGHT_FLUSH_BASE,
            CardEvaluator.BACK_STRAIGHT_FLUSH_BASE,
            CardEvaluator.ROYAL_STRAIGHT_FLUSH_BASE
        ]
        self.state = {
            'fixed_states': self.fixed_states,
            'unknown_spaces': self.unknown_spaces,
            'simulation_states': self.simulation_states,
            'excluded_states': self.excluded_states,
            'my_simulation_number': 0,
            'score_table': self.score_table,
            'score_calculation_callback': lambda cards: CardEvaluator.card_evaluator(cards),
            'risk_table': self.risk_table,
        }

        self.action_space = 3
        # self.action_space = [0, 1, 2]

        self.shuffle()

    def shuffle(self):
        self.simulation_states = list(range(52))
        self.allocate_max = 52
        random.shuffle(self.simulation_states)
        self.current_card = 0

    def reset(self):
        self.fixed_states.clear()
        self.simulation_states.clear()

        self.shuffle()
        self.state['simulation_states'] = self.simulation_states

        for player in range(self.players):
            self.fixed_states.append([])

        for i in range(4):
            for player in range(self.players):
                self.fixed_states[player].append(self.card_allocate())

        self.fixed_states[0].append(self.card_allocate())
        self.fixed_states[0].append(self.card_allocate())
        self.fixed_states[0].append(self.card_allocate())

        Logger.verb('mim:reset', f'{self.simulation_states}')

        return self.state

    def card_allocate(self):
        if len(self.simulation_states) == 0:
            return None
        else:
            hand = random.choice(self.simulation_states)
            self.simulation_states.remove(hand)
            return hand

    def step(self, action):
        scores = []
        max_p = len(self.fixed_states)
        rank = max_p

        step_states = copy.deepcopy(self.fixed_states)
        Logger.verb('envs:mim:step', step_states)
        for c in range(3):
            for p in range(len(step_states)):
                if p == 0:
                    continue
                step_states[p].append(self.card_allocate())

        for p in range(len(step_states)):
            scores.append(CardEvaluator.card_evaluator(
                step_states[p])['score'])

        my_score = scores[0]
        for p in range(len(step_states)):
            if p == 0:
                continue
            if my_score > scores[p]:
                rank -= 1

        if rank == 1:
            if action == 2:
                reward = max_p
            elif action == 1:
                reward = max_p / 2
            else:
                reward = -2
        elif rank == 2:
            if action == 2:
                reward = max_p - 1
            elif action == 1:
                reward = max_p / 2
            else:
                reward = -1
        elif rank == 3:
            if action == 2:
                reward = -1
            elif action == 1:
                reward = 1
            else:
                reward = 1
        elif rank == 4:
            if action == 2:
                reward = -2
            elif action == 1:
                reward = -1
            else:
                reward = 1
        elif rank == 5:
            if action == 2:
                reward = -2
            elif action == 1:
                reward = -2
            else:
                reward = 1
        return step_states, reward, True
