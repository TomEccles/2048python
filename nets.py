from board_features import board_as_feature_array
from predict_moves import PriorNet
from predict_values import ValuerNet


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
        return self.predictor.run_forward(board_as_feature_array(board))