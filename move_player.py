from to_play_node import ToPlayNode


class MovePlayer:
    def __init__(self, board, prior_weight):
        self.root = ToPlayNode(board, None, prior_weight=prior_weight)

    def evaluate(self):
        self.root.evaluate()

    def play(self, evaluations):
        for i in range(evaluations):
            self.evaluate()
        best = self.root.bestChild()
        return None if best is None else best.board
