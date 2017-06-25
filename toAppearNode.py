from indented_printer import IndentedPrinter
from node import Node
from rollout import rolloutFromAppear
import toPlayNode


class ToAppearNode(Node):
    def __init__(self, board, parent):
        super().__init__(board, parent)
        self.children = None
        self.nodeDict = dict()

    def print(self, depth):
        printer = IndentedPrinter(depth)
        printer.print("ToAppearNode")
        printer.print(self.board.to_string())
        printer.print("Games: %d, Score: %d")
        if self.children is not None:
            [c.print(depth + 1) for c in self.children]

    def value(self):
        return (self.score + 1000) / (self.games + 1)

    def rollout(self):
        return rolloutFromAppear(self.board.copy())

    def getChildNodeToEvaluate(self):
        new_board = self.board.copy()
        if new_board.add_random():
            return self.getOrCreateNode(new_board)
        else:
            return None

    def getOrCreateNode(self, new_board):
        try:
            node = self.nodeDict.get(new_board, None)
        except ValueError:
            a = 0
        if node is None:
            node = toPlayNode.ToPlayNode(new_board, self)
            self.nodeDict[new_board] = node
        return node

