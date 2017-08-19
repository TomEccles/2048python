import numpy as np
import roller

from board import Board


def __rollout_from_move_c(b):
    m = roller.roller(b.board.tolist())
    return m


def rollout_from_move(b):
    return __rollout_from_move_c(b)


def rollout_from_appear(b):
    b = b.add_random()
    return rollout_from_move(b)


if __name__ == "__main__":
    a = np.array([[1, 4, 7, 10], [1, 4, 7, 10], [1, 4, 7, 10], [1, 4, 7, 10]])
    s1 = 0
    for i in range(1000):
        s1 += rollout_from_move(Board(a))
    print(s1)
