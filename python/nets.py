import numpy
from sklearn import linear_model
from sklearn import metrics
from board import all_moves
from board_features import board_as_feature_array, move_as_one_hot_encoding, board_as_feature_array
from predict_moves import PriorNet
from predict_values import ValuerNet

value_normalisation = 1000


def get_values(boards):
    ends = [i - 1 for i in range(1, len(boards)) if boards[i].nonzeroes() == 1]
    ends.append(len(boards) - 1)
    game = 0
    values = []
    for index, board in enumerate(boards):
        if index > ends[game]:
            game += 1
        values.append((ends[game] - index) / value_normalisation)
    return values


def linear_loss(train_x, train_y, test_x, test_y):
    model = linear_model.LinearRegression()
    model.fit(numpy.array(train_x).reshape(-1, 1), train_y)
    pred_y = model.predict(numpy.array(test_x).reshape(-1, 1))
    return metrics.mean_squared_error(pred_y, test_y)


class Nets(object):
    def __init__(self, prior_weight, value_weight, predictor_to_load=None, valuer_to_load=None):
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
            return [0 for _ in boards]
        if not boards:
            return []
        return self.valuer.run_forward([board_as_feature_array(board) for board in boards])

    def feed_observations(self, move_boards, moves, appear_boards, passes):
        self.valuer.feed_observations(numpy.array([board_as_feature_array(board) for board in appear_boards]),
                                      numpy.array(get_values(appear_boards)),
                                      passes)

        board_arrays = numpy.array([board_as_feature_array(board) for board in move_boards])
        self.predictor.feed_observations(board_arrays,
                                         numpy.array([move_as_one_hot_encoding(l, b) for (l, b) in zip(moves, board_arrays)]),
                                         passes)

    def validate_observations(self, move_boards, moves, appear_boards):
        board_arrays = numpy.array([board_as_feature_array(board) for board in move_boards])
        self.predictor.validate_observations(
            numpy.array(board_arrays),
            numpy.array([move_as_one_hot_encoding(l, b) for (l, b) in zip(moves, board_arrays)]))
        self.valuer.validate_observations(
            numpy.array([board_as_feature_array(board) for board in appear_boards]),
            numpy.array(get_values(appear_boards)))

        move_sums = [numpy.sum(b.board) for b in move_boards]
        move_values = get_values(move_boards)

        appear_sums = [numpy.sum(b.board) for b in appear_boards]
        appear_values = get_values(move_boards)

        loss = linear_loss(move_sums, move_values, appear_sums, appear_values)
        print("loss from sums, rms: %.4f %.4f" % (loss, loss ** 0.5))

    def save(self, predict_path, value_path):
        self.predictor.save(predict_path)
        self.valuer.save(value_path)
