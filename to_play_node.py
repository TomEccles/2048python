from board import all_moves
from board_features import board_as_feature_array
from indented_printer import IndentedPrinter
from node import Node
from rollout import *
from to_appear_node import ToAppearNode


class NodeWithPrior(ToAppearNode):
    def __init__(self, board, parent, prior, nets):
        super().__init__(board, parent, nets)
        self.prior = prior

    def value(self):
        return (self.score + 1000 + self.prior*self.nets.get_prior_weight()) / (self.games + 1)


class ToPlayNode(Node):
    def __init__(self, board, parent, nets, is_root=False):
        super().__init__(board, parent, nets)
        self.children = None
        self.is_root = is_root

    def print(self, depth):
        printer = IndentedPrinter(depth)
        printer.print("ToPlayNode")
        printer.print(self.board.to_string())
        printer.print("Games: %d, Score: %d")
        if self.children is not None:
            [c.print(depth + 1) for c in self.children]

    def rollout(self):
        return rolloutFromMove(self.board.copy())

    def bestChild(self):
        options = self.get_possible_children()
        return None if not options else max(options, key=lambda child: child.games)

    def getChildNodeToEvaluate(self):
        options = self.get_possible_children()
        return None if not options else max(options, key=lambda child: child.value())

    def get_possible_children(self):
        if self.children is not None:
            return self.children

        # Bias the evaluation towards likely candidates - unless we're the root node, in which case we should be fair
        priors = \
            self.nets.get_priors(self.board.board) \
            if not self.is_root \
            else [0.25 for _ in all_moves]

        boards = []
        for move in all_moves:
            moved = self.board.move(move)
            if moved != self.board:
                boards.append((moved, priors[move]))
        self.children = [NodeWithPrior(board, self, prior, self.nets) for board, prior in boards]

        return self.children
