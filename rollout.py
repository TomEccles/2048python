import roller
import random
import numpy as np

from board import Board, Move


def random_horizontal(b):
    if b.can_move_top_row_right() or random.randint(0, 1) == 0:
        return b.move(Move.left) or b.move(Move.right)
    return b.move(Move.right) or b.move(Move.left)


def move_successful(b):
    return b.move(Move.up) or random_horizontal(b) or b.move(Move.down)


def rolloutFromMovePython(b):
    m = 0
    while move_successful(b):
        b.add_random()
        m += 1
    return m


def rolloutFromMoveC(b):
    m = roller.roller(b.board.tolist())
    return m


def rolloutFromMove(b):
    return rolloutFromMoveC(b)


def rollout_from_appear(b):
    b.add_random()
    return rolloutFromMove(b)


if __name__ == "__main__":
    a = np.array([[1, 4, 7, 10], [1, 4, 7, 10], [1, 4, 7, 10], [1, 4, 7, 10]])
    s1 = 0
    s2 = 0
    b = Board(a)
    for i in range(1000):
        b = Board(a)
        m = rolloutFromMoveC(b)
        s1 += m
    for i in range(1000):
        b = Board(a)
        m = rolloutFromMovePython(b)
        s2 += m
    print(s1, s2)
