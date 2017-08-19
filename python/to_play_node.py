from board import all_moves
from indented_printer import IndentedPrinter
from node import Node
from rollout import *
from to_appear_node import ToAppearNode


class NodeWithPriorAndValue(ToAppearNode):
    def __init__(self, board, parent, prior, value, nets):
        super().__init__(board, parent, value, nets)
        self.prior = prior

    def value(self):
        return (self.score + 1000 + self.prior * self.nets.get_prior_weight()) / (self.games + 1)


class ToPlayNode(Node):
    def __init__(self, board, parent, nets):
        super().__init__(board, parent, nets)
        self.children = None

    def print(self, depth):
        printer = IndentedPrinter(depth)
        printer.print("ToPlayNode")
        printer.print(self.board.to_string())
        printer.print("Games: %d, Score: %d")
        if self.children is not None:
            [c.print(depth + 1) for c in self.children]

    def rollout(self):
        return rollout_from_move(self.board.copy())

    def bestChild(self):
        options = self.get_possible_children()
        return None if not options else max(options, key=lambda child: child.games)

    def getChildNodeToEvaluate(self):
        options = self.get_possible_children()
        return None if not options else max(options, key=lambda child: child.value())

    def get_possible_children(self):
        if self.children is not None:
            return self.children

        priors = self.nets.get_priors(self.board)

        boards = []
        for move in all_moves:
            moved = self.board.move(move)
            if moved != self.board:
                boards.append((moved, priors[move]))
        values = self.nets.get_values([board[0] for board in boards])

        self.children = [NodeWithPriorAndValue(board_and_prior[0], self, board_and_prior[1], value, self.nets) for
                         board_and_prior, value in zip(boards, values)]

        return self.children
