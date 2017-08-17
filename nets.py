import numpy

from board import all_moves
from board_features import board_as_feature_array, move_as_one_hot_encoding, board_as_feature_array_with_sum
from predict_moves import PriorNet
from predict_values import ValuerNet


def get_values(boards):
    ends = [i - 1 for i in range(1, len(boards)) if boards[i].nonzeroes() == 1]
    ends.append(len(boards) - 1)
    game = 0
    values = []
    for index, board in enumerate(boards):
        if index > ends[game]:
            game += 1
        values.append((ends[game] - index)/1000)
    return values


class Nets(object):
    def __init__(self, prior_weight, value_weight, predictor_to_load, valuer_to_load):
        self.prior_weight = prior_weight
        self.predictor = PriorNet()
        if predictor_to_load is not None:
            self.predictor.load(predictor_to_load)

        self.value_weight = value_weight
        self.valuer = ValuerNet()
        if valuer_to_load is not None:
            self.valuer.load(valuer_to_load)

    def get_prior_weight(self):
        return self.prior_weight

    def get_value_weight(self):
        return self.value_weight

    def get_priors(self, board):
        if self.prior_weight == 0:
            return [0.25 for _ in all_moves]
        return self.predictor.run_forward(board_as_feature_array(board))

    def get_values(self, boards):
        if self.value_weight == 0:
            return [0 for i in boards]
        if not boards:
            return []
        return self.valuer.run_forward([board_as_feature_array_with_sum(board) for board in boards])

    def feed_observations(self, move_boards, moves, appear_boards, passes):
        self.valuer.feed_observations(numpy.array([board_as_feature_array_with_sum(board) for board in appear_boards]),
                                      numpy.array(get_values(appear_boards)),
                                      passes)
        self.predictor.feed_observations(numpy.array([board_as_feature_array(board) for board in move_boards]),
                                         numpy.array([move_as_one_hot_encoding(l) for l in moves]),
                                         passes)

    def validate_observations(self, move_boards, moves, appear_boards):
        self.predictor.validate_observations(numpy.array([board_as_feature_array(board) for board in move_boards]),
                                             numpy.array([move_as_one_hot_encoding(l) for l in moves]))
        self.valuer.validate_observations(numpy.array([board_as_feature_array_with_sum(board) for board in appear_boards]),
                                          numpy.array(get_values(appear_boards)))
        sums = [numpy.sum(b.board)/1000 for b in appear_boards]
        values = get_values(appear_boards)
        diffs = [sums[i] + values[i] for i in range(len(sums))]
        average_diff = sum(diffs) / len(diffs)
        preds = [max(average_diff - s, 0) for s in sums]
        errors = [(preds[i] - values[i])**2 for i in range(len(preds))]
        loss = sum(errors) / len(errors)
        print("loss from sums, rms: %.4f %.4f" % (loss, loss ** 0.5))

    def save(self, predict_path, value_path):
        self.predictor.save(predict_path)
        self.valuer.save(value_path)