import time

from board import Board
from move_player import MovePlayer
import cProfile


def play_game(evals):
    b = Board()
    moves = 0
    start = time.time()
    while b.add_random():
        p = MovePlayer(b)
        b = p.play(evals)
        if b is None:
            break
        moves += 1
    return moves, time.time() - start

cProfile.run("play_game(10)")
