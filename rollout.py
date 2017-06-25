import random

from board import Board


def random_horizontal(b):
    if b.can_move_top_row_right() or random.randint(0, 1) == 0:
        return b.move_left() or b.move_right()
    return b.move_right() or b.move_left()


def move_successful(b):
    return b.move_up() or random_horizontal(b) or b.move_down()


def rolloutFromMove(b):
    if not move_successful(b):
        return 0
    return rollout(b)


def rollout(b):
    moves = 0
    while b.add_random():
        if not move_successful(b):
            break
        moves += 1
    return moves

if __name__ == "__main__":
    b = Board()
    print(rollout(b))
