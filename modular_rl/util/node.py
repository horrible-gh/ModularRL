import math
from LogAssist.log import Logger
import numpy as np


class Node:
    def __init__(self, state, prior, action=None, done=False):
        self.state = state
        self.prior = prior
        self.action = action  # 액션 정보를 저장하는 속성 추가
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.total_value = 0
        self.done = done  # Assign the done attribute


    def __repr__(self):
        return f'Node(state={self.state}, total_value={self.total_value}, visit_count={self.visit_count}, is_expanded={self.expanded()}, action_probs={self.action}, children={self.children})'

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count  # 총 보상을 방문 횟수로 나누어 평균 보상을 반환

    def select_child(self, cpuct):
        best_score = -math.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            uct_score = child.value() + cpuct * child.prior * math.sqrt(self.visit_count) / \
                (1 + child.visit_count)
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_space, child_priors, child_done):  # Add a child_done parameter
        for action, prior in zip(range(action_space), child_priors):
            if action not in self.children:
                self.children[action] = Node(self.state, prior, action, child_done)  # Pass the child_done to the new Node

    def update_stats(self, reward):
        self.total_value += reward  # 총 보상 업데이트
        self.visit_count += 1  # 방문 횟수 증가

    def select_action(self, temperature=1):
        visit_counts = np.array(
            [child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        else:
            visit_counts = visit_counts ** (1/temperature)  # apply temperature
            # normalize to get probabilities
            visit_counts = visit_counts / sum(visit_counts)
            action = np.random.choice(actions, p=visit_counts)

        return action
