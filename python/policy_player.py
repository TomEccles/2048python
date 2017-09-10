import random

from board_features import *
from predict_moves import PolicyNetwork


def move_from_distribution(dist):
    cumul = 0
    r = random.random()
    for i, m in enumerate(all_moves):
        cumul += dist[i]
        if cumul > r:
            return m
    print(dist)
    raise Exception("Didn't get a move" + dist)


class PolicyPlayer(object):
    def __init__(self, predictor=None):
        self.predictor = PolicyNetwork() if predictor is None else predictor

    def get_move(self, board):
        options = self.predictor.run_forward(board_as_feature_array(board))
        return move_from_distribution(options)

    def feed_observations(self, move_boards, moves, values, passes):
        board_arrays = numpy.array([board_as_feature_array(board) for board in move_boards])

        self.predictor.feed_reinforcement(board_arrays,
                                          numpy.array([move_as_one_hot_encoding(l, b) for (l, b) in
                                                       zip(moves, board_arrays)]),
                                          numpy.array([[v] for v in values]),
                                          passes)

    def save(self, path):
        self.predictor.save(path)