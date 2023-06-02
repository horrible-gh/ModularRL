# -*- coding: utf-8 -*-
class CardEvaluator:
    ROYAL_STRAIGHT_FLUSH_BASE = 5000000
    BACK_STRAIGHT_FLUSH_BASE = 3000000
    STRAIGHT_FLUSH_BASE = 1000000
    STRAIGHT_FLUSH_MULTIPLY = 10000
    FOUR_OF_A_KIND_BASE = 500000
    FOUR_OF_A_KIND_MULTIPLY = 10000
    FULL_HOUSE_BASE = 300000
    FULL_HOUSE_MULTIPLY = 10000
    FLUSH_BASE = 100000
    FLUSH_MULTIPLY = 10000
    MOUNTAIN_BASE = 80000
    BACK_STRAIGHT_BASE = 70000
    STRAIGHT_BASE = 50000
    STRAIGHT_MULTIPLY = 1000
    THREE_OF_A_KIND_BASE = 5000
    THREE_OF_A_KIND_MULTIPLY = 1000
    TWO_PAIR_BASE = 1000
    TWO_PAIR_MULTIPLY = 100
    TWO_PAIR_MULTIPLY_SUB = 10
    ONE_PAIR_BASE = 300
    ONE_PAIR_MULTIPLY = 10
    NO_PAIR_MULTIPLY = 10
    NO_PAIR_BASE = 0

    def card_num_to_str(card):
        card_table = {
            '1': 'A',
            '11': 'J',
            '12': 'Q',
            '13': 'K',
            '14': 'A',
        }
        return card_table.get(str(card), card)

    def add_ace_value(cards, add_mode=False, plus_only=False):
        conv_cards = []
        for i, card in enumerate(cards):
            if card < 4:  # 0부터 3까지의 값일 경우
                if add_mode == True:
                    conv_cards.append(cards[i])
                if plus_only == False:
                    conv_cards.append(cards[i] + 52)  # 52를 더해줌으로써 Ace의 숫자를 높힘
            else:
                conv_cards.append(cards[i])
        return conv_cards

    def is_flush_types(cards):
        suits = [card % 4 for card in cards]  # 무늬 리스트
        max_suit = max(set(suits), key=suits.count)  # 가장 많은 무늬 추출
        matched_cards = set(filter(lambda x: x %
                            4 == max_suit, cards))  # 같은 무늬 카드 추출
        if len(matched_cards) < 5:  # 추출한 카드가 5장 미만이면 플러시 계열이 아님
            return 0

        # 추출한 카드 5장이 로얄 스트레이트 플러시인지 확인
        royal_straight_cards = [1, 10, 11, 12, 13]
        matched_royal_straight_cards = [
            card for card in matched_cards if (card // 4) + 1 in royal_straight_cards
        ]

        if len(matched_royal_straight_cards) == 5:
            return {
                'description': 'Royal Straight Flush',
                'score': CardEvaluator.ROYAL_STRAIGHT_FLUSH_BASE + max_suit,
                'probability': 3,
                'expectation': 0,
            }

        # 추출한 카드 5장이 백스트레이트 플러시인지 확인
        back_straight_cards = [1, 2, 3, 4, 5]
        matched_back_straight_cards = [
            card for card in matched_cards if (card // 4) + 1 in back_straight_cards
        ]

        if len(matched_back_straight_cards) == 5:
            return {
                'description': 'Back Straight Flush',
                'score': CardEvaluator.BACK_STRAIGHT_FLUSH_BASE + max_suit,
                'probability': 2.9,
                'expectation': 0,
            }

        # 추출한 카드 5장이 스트레이트 플러시인지 확인
        values = sorted([card // 4 for card in matched_cards])  # 숫자 리스트
        for i in range(len(values) - 4):
            if values[i:i+5] == list(range(values[i], values[i]+5)):
                return {
                    'description': f'{CardEvaluator.card_num_to_str((values[i]+5))} Straight Flush',
                    'score': (values[i]+5) * CardEvaluator.STRAIGHT_FLUSH_MULTIPLY
                    + CardEvaluator.STRAIGHT_FLUSH_BASE + max_suit,
                    'probability': 2.5,
                    'expectation': 0,
                }

        # 추출한 카드가 플러시인지 확인
        matched_cards = list(matched_cards)
        matched_cards = CardEvaluator.add_ace_value(matched_cards)
        max_flush_num = max(matched_cards) // 4 + 1
        score = CardEvaluator.FLUSH_BASE + \
            (max_flush_num * CardEvaluator.FLUSH_MULTIPLY) + max_suit
        return {
            'description': f'{CardEvaluator.card_num_to_str(max_flush_num)} Flush',
            'score': score,
            'probability': 0.8,
            'expectation': 0,
        }

    def is_four_of_a_kind(cards):
        conv_cards = CardEvaluator.add_ace_value(cards)
        nums = [card // 4 for card in conv_cards]  # 카드 숫자 리스트
        max_nums = max(set(nums), key=nums.count)  # 가장 많은 숫자 추출
        matched_cards = set(filter(lambda x: x //
                                   4 == max_nums, conv_cards))  # 같은 숫자 카드 추출
        if len(matched_cards) < 4:  # 추출한 카드가 4장 이하면 포카드 아님
            return 0

        return {
            'description': f'{CardEvaluator.card_num_to_str(max_nums+1)} Four Of A Kind',
            'score': CardEvaluator.FOUR_OF_A_KIND_MULTIPLY * (max_nums + 1) + CardEvaluator.FOUR_OF_A_KIND_BASE,
            'probability': 2,
            'expectation': 0,
        }

    def is_full_house(cards):
        conv_cards = CardEvaluator.add_ace_value(cards)
        nums = [card // 4 for card in conv_cards]
        three_of_a_kind = []

        # 세 장의 같은 숫자 그룹 찾기
        for i in set(nums):
            if nums.count(i) == 3:
                three_of_a_kind.append(i)

        # 세 장의 같은 숫자 그룹이 2개 이상인 경우
        if len(three_of_a_kind) >= 2:
            three_of_a_kind.sort(reverse=True)  # 숫자가 높은 순으로 정렬
            high_group = three_of_a_kind[0] + 1  # 가장 높은 숫자 그룹
            low_group = three_of_a_kind[1] + 1  # 두 번째로 높은 숫자 그룹

            # 두 장의 같은 숫자 그룹을 찾기 위해 세 장의 같은 숫자 그룹을 제거한 리스트 생성
            reduced_nums = [num for num in nums if num not in three_of_a_kind]

            for j in set(reduced_nums):
                if reduced_nums.count(j) == 2:
                    return {
                        'description': f'{CardEvaluator.card_num_to_str(high_group)},{CardEvaluator.card_num_to_str(low_group)} Full House',
                        'score': CardEvaluator.FULL_HOUSE_MULTIPLY * high_group + CardEvaluator.FULL_HOUSE_BASE,
                        'probability': 1,
                        'expectation': 0,
                    }

        # 세 장의 같은 숫자 그룹이 하나만 있는 경우
        elif len(three_of_a_kind) == 1:
            for j in set(nums):
                if nums.count(j) == 2 and j != three_of_a_kind[0]:
                    return {
                        'description': f'{CardEvaluator.card_num_to_str(three_of_a_kind[0] + 1)},{CardEvaluator.card_num_to_str(j + 1)} Full House',
                        'score': CardEvaluator.FULL_HOUSE_MULTIPLY * three_of_a_kind[0] + 1 + CardEvaluator.FULL_HOUSE_BASE,
                        'probability': 1,
                        'expectation': 0,
                    }

        return 0

    def is_straight_types(cards):
        conv_cards = CardEvaluator.add_ace_value(cards, True, False)
        conv_cards = sorted(conv_cards, key=lambda x: (not x, x))  # 카드 정렬
        values = [card // 4 for card in conv_cards]  # 숫자 리스트
        # suit = [card % 4 for card in conv_cards]  # 무늬 리스트
        score = 0
        dic_score = 0

        # 중복을 제거한 숫자 리스트 생성
        unique_values = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        values = unique_values

        # 다섯 장의 카드가 연속되어 있는지 확인
        for i in range(len(values) - 4):
            if values[i:i+5] == list(range(values[i], values[i]+5)):
                # 가장 낮은 숫자와 해당 카드의 무늬 정보 반환
                min_value = values[i]

                min_cards = [card for card in conv_cards if card //
                             4 == min_value]  # 가장 낮은 숫자의 카드들을 찾음
                min_card = min(min_cards)  # 가장 낮은 무늬 값을 가진 카드를 찾음
                min_suit = min_card % 4  # 가장 높은 카드의 무늬를 찾음

                max_value = values[i+4]
                max_cards = [card for card in conv_cards if card //
                             4 == max_value]  # 가장 높은 숫자의 카드들을 찾음
                max_card = max(max_cards)  # 가장 높은 무늬 값을 가진 카드를 찾음
                max_suit = max_card % 4  # 가장 높은 카드의 무늬를 찾음
                if min_value == 0:
                    score = CardEvaluator.BACK_STRAIGHT_BASE + min_suit
                    dic_score = {
                        'description': f'Back Straight',
                        'score': score,
                        'probability': 0.45,
                        'expectation': 0,
                    }
                elif max_value == 13:
                    score = CardEvaluator.MOUNTAIN_BASE + max_suit
                    dic_score = {
                        'description': f'Mountain Straight',
                        'score': score,
                        'probability': 0.5,
                        'expectation': 0,
                    }
                else:
                    next_score = max_value * CardEvaluator.STRAIGHT_MULTIPLY + \
                        CardEvaluator.STRAIGHT_BASE + max_suit
                    if score < next_score:
                        score = next_score
                        dic_score = {
                            'description': f'{CardEvaluator.card_num_to_str(max_value+1)} Straight',
                            'score': score,
                            'probability': 0.4,
                            'expectation': 0,
                        }

        return dic_score

    def is_three_of_a_kind(cards):
        conv_cards = CardEvaluator.add_ace_value(cards)
        nums = [card // 4 for card in conv_cards]  # 카드 숫자 리스트
        max_nums = max(set(nums), key=nums.count)  # 가장 많은 숫자 추출
        matched_cards = set(filter(lambda x: x //
                                   4 == max_nums, conv_cards))  # 같은 숫자 카드 추출
        if len(matched_cards) < 3:  # 추출한 카드가 3장 이하면 트리플 아님
            return 0
        if len(cards) < 7:
            expectation = 1
        else:
            expectation = 0
        return {
            'description': f'{CardEvaluator.card_num_to_str(max_nums+1)} Three Of A Kind',
            'score': CardEvaluator.THREE_OF_A_KIND_MULTIPLY * max_nums + CardEvaluator.THREE_OF_A_KIND_BASE,
            'probability': 0.1,
            'expectation': expectation,
        }

    def is_pair_types(cards):
        conv_cards = CardEvaluator.add_ace_value(cards)
        nums = [card // 4 for card in conv_cards]
        pairs = []
        high_group = 0
        low_group = 0

        # 두 장의 같은 숫자 그룹 찾기
        for i in set(nums):
            if nums.count(i) == 2:
                pairs.append(i)

        # 두 장의 같은 숫자 그룹이 1벌 이상인 경우
        pair_groups = len(pairs)
        if pair_groups >= 1:
            pairs.sort(reverse=True)  # 숫자가 높은 순으로 정렬
            high_group = pairs[0]  # 가장 높은 숫자 그룹
            if pair_groups >= 2:
                low_group = pairs[1]  # 두 번째로 높은 숫자 그룹

            # 가장 높은 숫자 그룹의 카드 번호를 베이스 점수로 사용
            high_card = max([card for card in conv_cards if card //
                            4 == high_group], key=lambda card: card % 4)
            if pair_groups >= 2:
                base_score = CardEvaluator.TWO_PAIR_MULTIPLY * \
                    high_group + CardEvaluator.TWO_PAIR_BASE
            else:
                base_score = CardEvaluator.ONE_PAIR_MULTIPLY * \
                    high_group + CardEvaluator.ONE_PAIR_BASE
            if pair_groups >= 2:
                # 두 번째로 높은 숫자 그룹의 카드 번호를 서브 점수로 사용
                sub_score = low_group * CardEvaluator.TWO_PAIR_MULTIPLY_SUB

            # 가장 높은 숫자 그룹의 무늬 점수를 계산
            high_suit = high_card % 4
            suit_score = high_suit

            if pair_groups >= 2:
                if len(cards) < 7:
                    expectation = 0.75
                else:
                    expectation = 0
                return {
                    'description': f'{CardEvaluator.card_num_to_str(high_group+1)},{CardEvaluator.card_num_to_str(low_group+1)} Two Pair',
                    'score': base_score + sub_score + suit_score,
                    'probability': 0.05,
                    'expectation': expectation,
                }
            if pair_groups == 1:
                if len(cards) < 7:
                    expectation = 0.5
                else:
                    expectation = 0
                return {
                    'description': f'{CardEvaluator.card_num_to_str(high_group+1)} One Pair',
                    'score': base_score + suit_score,
                    'probability': 0.01,
                    'expectation': expectation,
                }
        return 0

    def no_pair(cards):
        conv_cards = sorted(CardEvaluator.add_ace_value(cards), reverse=True)
        if len(cards) < 7:
            expectation = 0.5
        else:
            expectation = 0
        return {
            'description': f'{CardEvaluator.card_num_to_str((conv_cards[0]//4)+1)} Top',
            'score': (CardEvaluator.NO_PAIR_MULTIPLY * (conv_cards[0] // 4)) + (conv_cards[0] % 4),
            'probability': 0.0003,
            'expectation': expectation,
        }

    def card_evaluator(cards):
        # print([card // 4 for card in cards])
        # print([card % 4 for card in cards])
        cards_len = len(cards)
        score = {
            'description': 'none',
            'score': 0,
            'probability': 0.00,
            'expectation': 0.00,
        }
        # Royal Straight Flush
        if cards_len >= 5:
            score = CardEvaluator.is_flush_types(cards)
            if score != 0:
                return score
        if cards_len >= 4:
            score = CardEvaluator.is_four_of_a_kind(cards)
            if score != 0:
                return score
        if cards_len >= 5:
            score = CardEvaluator.is_full_house(cards)
            if score != 0:
                return score
            score = CardEvaluator.is_straight_types(cards)
            if score != 0:
                return score
        if cards_len >= 3:
            score = CardEvaluator.is_three_of_a_kind(cards)
            if score != 0:
                return score
        if cards_len >= 2:
            score = CardEvaluator.is_pair_types(cards)
            if score != 0:
                return score
        score = CardEvaluator.no_pair(cards)
        return score
