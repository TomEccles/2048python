import roller
import random
import numpy as np

from board import Board


def random_horizontal(b):
    if b.can_move_top_row_right() or random.randint(0, 1) == 0:
        return b.move_left() or b.move_right()
    return b.move_right() or b.move_left()


def move_successful(b):
    return b.move_up() or random_horizontal(b) or b.move_down()


def rolloutFromMovePython(b):
    m = 0
    while move_successful(b):
        b.add_random()
        m += 1
    return m

def rolloutFromMove(b):
    m = roller.roller(b.board.tolist())
    return m

def rolloutFromAppear(b):
    b.add_random()
    return rolloutFromMove((b))

if __name__ == "__main__":
    a = np.array([[1,0,0,0],[4,0,1,0],[7,0,0,0],[10,0,0,0]])
    s1 = 0
    s2 = 0
    for i in range(100):
        b = Board(a)
        m = rolloutFromMove(b)
        s1 += m
    for i in range(100):
        b = Board(a)
        m = rolloutFromMovePython(b)
        s2 += m
    print(s1, s2)