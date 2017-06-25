from toPlayNode import ToPlayNode


class MovePlayer:
    def __init__(self, board):
        self.root = ToPlayNode(board, None)

    def evaluate(self):
        self.root.evaluate()

    def play(self, evaluations):
        for i in range(evaluations):
            self.evaluate()
        best = self.root.bestChild()
        return None if best is None else best.board
