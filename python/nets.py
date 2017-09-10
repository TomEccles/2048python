import numpy
from sklearn import linear_model
from sklearn import metrics
from board import all_moves
from board_features import board_as_feature_array, move_as_one_hot_encoding, board_as_feature_array
from predict_moves import PolicyNetwork
from predict_values import ValuerNet
from game_player import GameBatchResult

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
    def __init__(self, prior_weight, value_weight, policy_to_load=None, valuer_to_load=None):
        self.prior_weight = prior_weight
        self.policy_net = PolicyNetwork()
        if policy_to_load is not None:
            self.policy_net.load(policy_to_load)

        self.value_weight = value_weight
        self.value_net = ValuerNet()
        if valuer_to_load is not None:
            self.value_net.load(valuer_to_load)

    def get_prior_weight(self):
        return self.prior_weight

    def get_value_weight(self):
        return self.value_weight

    def get_priors(self, board):
        if self.prior_weight == 0:
            return [0.25 for _ in all_moves]
        return self.policy_net.run_forward(board_as_feature_array(board))

    def get_values(self, boards):
        if self.value_weight == 0:
            return [0 for _ in boards]
        if not boards:
            return []
        return self.value_net.run_forward([board_as_feature_array(board) for board in boards])

    def train_policy_net(self, game_batch, passes):
        """

        :type game_batch: GameBatchResult
        """
        board_arrays = numpy.array([board_as_feature_array(board) for board in game_batch.to_move_boards])
        self.policy_net.feed_observations(board_arrays,
                                          numpy.array(
                                             [move_as_one_hot_encoding(l, b) for (l, b) in zip(game_batch.moves, board_arrays)]),
                                          passes)

    def train_valuer(self, game_batch, passes):
        """

        :type game_batch: GameBatchResult
        """
        self.value_net.feed_observations(numpy.array([board_as_feature_array(board) for board in game_batch.to_appear_boards]),
                                         numpy.array(get_values(game_batch.to_appear_boards)),
                                         passes)

    def feed_observations(self, game_batch, passes):
        self.train_policy_net(game_batch, passes)
        self.train_valuer(game_batch, passes)

    def validate_observations(self, game_batch, train_batch):
        """

        :type game_batch: GameBatchResult
        """
        board_arrays = numpy.array([board_as_feature_array(board) for board in game_batch.to_move_boards])
        self.policy_net.validate_observations(
            numpy.array(board_arrays),
            numpy.array([move_as_one_hot_encoding(l, b) for (l, b) in zip(game_batch.moves, board_arrays)]))
        self.value_net.validate_observations(
            numpy.array([board_as_feature_array(board) for board in game_batch.to_appear_boards]),
            numpy.array(get_values(game_batch.to_appear_boards)))

        train_sums = [numpy.sum(b.board) for b in train_batch.to_appear_boards]
        train_values = get_values(train_batch.to_appear_boards)
        appear_sums = [numpy.sum(b.board) for b in game_batch.to_appear_boards]
        appear_values = get_values(game_batch.to_appear_boards)

        loss = linear_loss(train_sums, train_values, appear_sums, appear_values)
        print("loss from sums, rms: %.4f %.4f" % (loss, loss ** 0.5))

    def save(self, policy_path=None, value_path=None):
        self.policy_net.save(policy_path)
        self.value_net.save(value_path)
