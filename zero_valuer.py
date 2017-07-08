from board import all_moves


class ZeroValuer(object):
    def get_prior_weight(self):
        return 0

    def get_value_weight(self):
        return 0

    def get_priors(self, board):
        return [0.25 for _ in all_moves]