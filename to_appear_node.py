from indented_printer import IndentedPrinter
from node import Node
from rollout import rollout_from_appear
import to_play_node


class ToAppearNode(Node):
    def __init__(self, board, parent, prior_weight):
        super().__init__(board, parent, prior_weight)
        self.children = None
        self.nodeDict = dict()

    def print(self, depth):
        printer = IndentedPrinter(depth)
        printer.print("ToAppearNode")
        printer.print(self.board.to_string())
        printer.print("Games: %d, Score: %d")
        if self.children is not None:
            [c.print(depth + 1) for c in self.children]

    def rollout(self):
        return rollout_from_appear(self.board.copy())

    def getChildNodeToEvaluate(self):
        new_board = self.board.copy()
        if new_board.add_random():
            return self.getOrCreateNode(new_board)
        else:
            return None

    def getOrCreateNode(self, new_board):
        node = self.nodeDict.get(new_board, None)
        if node is None:
            node = to_play_node.ToPlayNode(new_board, self, self.prior_weight)
            self.nodeDict[new_board] = node
        return node