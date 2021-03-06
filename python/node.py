from board import Board


class Node(object):
    def __init__(self, board: Board, parent, nets):
        self.board = board
        self.parent = parent
        self.score = 0
        self.games = 0
        self.children = None
        self.nets = nets

    def evaluate(self):
        if self.score == 0:
            value = self.rollout()
            self.registerScore(value)
        else:
            child = self.getChildNodeToEvaluate()
            if child is not None:
                child.evaluate()
            else:
                self.registerScore(0)

    def registerScore(self, value):
        self.score += value
        self.games += 1
        if self.parent is not None:
            self.parent.registerScore(value)

    def getChildNodeToEvaluate(self):
        raise NotImplementedError("getChildNodeToEvaluate should be implemented by inheritor of Node")

    def rollout(self):
        raise NotImplementedError("rollout should be implemented by inheritor of Node")
