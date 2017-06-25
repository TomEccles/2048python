from indented_printer import IndentedPrinter
from node import Node
from rollout import *
from toAppearNode import ToAppearNode


class ToPlayNode(Node):
    def __init__(self, board, parent):
        super().__init__(board, parent)
        self.children = None

    def print(self, depth):
        printer = IndentedPrinter(depth)
        printer.print("ToPlayNode")
        printer.print(self.board.to_string())
        printer.print("Games: %d, Score: %d")
        if self.children is not None:
            [c.print(depth + 1) for c in self.children]

    def rollout(self):
        return rolloutFromMovePython(self.board.copy())

    def bestChild(self):
        options = self.get_possible_children()
        return None if not options else max(options, key=lambda a: a.games)

    def getChildNodeToEvaluate(self):
        options = self.get_possible_children()
        return None if not options else max(options, key=lambda a: a.value())

    def get_possible_children(self):
        if self.children is not None:
            return self.children

        b = self.board
        boards = [board for (board, change)
                  in [b.move_down_copy(), b.move_left_copy(), b.move_right_copy(), b.move_up_copy()]
                  if change]
        self.children = [ToAppearNode(board, self) for board in boards]

        return self.children
