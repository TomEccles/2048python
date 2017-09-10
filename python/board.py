import random
from functools import reduce

import numpy as np


def apply_to_rows_returning_or(array, fn):
    return reduce(lambda a, b: fn(b) or a, array, False)


def squish_uncached(array):
    """Returns a copy of the array, squished towards the start 2048 style. (1,1,0,1)->(2,1,0,0)"""
    new = np.zeros(shape=len(array))

    last = None
    new_index = 0

    zero_found = False
    zero_filled = False
    combined = False

    for i in array:
        if i == 0:
            zero_found = True
            continue
        if zero_found:
            zero_filled = True
        if i == last:
            new[new_index - 1] = i + 1
            last = None
            combined = True
        else:
            new[new_index] = i
            last = i
            new_index += 1
    return new, combined or zero_filled


class A(dict):
    def __missing__(self, key):
        self[key] = squish_uncached(key)
        return self[key]


squish_cache = A()


class Move:
    up = 0
    left = 1
    right = 2
    down = 3


all_moves = [Move.up, Move.left, Move.right, Move.down]


class Board:
    """2048 board, with mutation methods to play a game"""

    def __init__(self, arr=None):
        self.board = np.zeros(shape=(4, 4), dtype=int) if arr is None else np.copy(arr)

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash(self.board.tobytes())

    def copy(self):
        return Board(self.board)

    def nonzeroes(self):
        (first, second) = np.where(self.board != 0)
        return first.size

    def can_add_random(self):
        (first, second) = np.where(self.board == 0)
        return first.size > 0

    def add_random(self):
        """Add 1 (90%) or 2 (10%) to a random empty square"""
        (first, second) = np.where(self.board == 0)
        if first.size == 0:
            raise Exception("Can't add to full board")
        index = random.randint(0, len(first) - 1)
        new = 1 if random.random() < 0.9 else 2

        copy = self.copy()
        copy.board[first[index]][second[index]] = new
        return copy

    def print(self):
        print(self.to_string())

    def to_string(self):
        return np.array_str(self.board)

    def move(self, move):
        copy = self.copy()
        copy.__move_in_place(move)
        return copy

    def __move_in_place(self, move):
        if move == Move.up:
            return apply_to_rows_returning_or(self.board.T, self.squish_cached)
        elif move == Move.down:
            return apply_to_rows_returning_or(self.board.T, lambda i: self.squish_cached(i[::-1]))
        elif move == Move.left:
            return apply_to_rows_returning_or(self.board, self.squish_cached)
        elif move == Move.right:
            return apply_to_rows_returning_or(self.board, lambda i: self.squish_cached(i[::-1]))

    def can_move_top_row_right(self):
        """A cached squish"""
        top_row = self.board[0][::-1]
        return squish_cache[tuple(top_row)][1]

    @staticmethod
    def squish_cached(row):
        b = tuple(row)
        row[:], result = squish_cache[b]
        return result
