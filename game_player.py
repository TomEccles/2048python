import random
import time

import numpy
import os
import pickle

from board import Board, all_moves
from board_features import board_as_feature_array, move_as_one_hot_encoding
from file_utils import open_creating_dir_if_needed
from move_player import MovePlayer
import predict_moves


def play_game(evals, prior_weight=200):
    boards = []
    labels = []
    b = Board()
    moves = 0
    start = time.time()
    while b.add_random():
        updated_board = MovePlayer(b, prior_weight).play(evals)
        if updated_board is None:
            break

        for move in all_moves:
            copy = b.copy()
            copy.move(move)
            if copy == updated_board:
                boards.append(b.copy())
                labels.append(move)
                b.move(move)
                moves += 1
                break
        else:
            raise Exception("Move player has returned an illegal board!")
    return moves, time.time() - start, boards, labels


def get_data(n, results_file=None):
    d = []
    l = []
    r = []
    for _ in range(n):
        moves, t, boards, labels = play_game(100)
        d = d + boards
        l = l + labels
        r.append(moves)
        if results_file:
            with open_creating_dir_if_needed(results_file, 'a') as file:
                file.write("%d %f\n" % (moves, t))
    return d, l, r


def train_indefinitely(game_batch, passes, valid_batch, path = "."):
    count = 0
    while True:
        count += 1
        boards, labels, results = get_data(game_batch, path + "/output/results_iter_%i.txt" % count)
        b_val, l_val, r_val = get_data(valid_batch)
        predict_moves.feed_observations(numpy.array([board_as_feature_array(board) for board in boards]),
                                        numpy.array([move_as_one_hot_encoding(l) for l in labels]),
                                        passes)
        predict_moves.validate_observations(numpy.array([board_as_feature_array(board) for board in b_val]),
                                            numpy.array([move_as_one_hot_encoding(l) for l in l_val]))
        predict_moves.save(path + "/checkpoints/checkpoint_iter_%i.ckpt" % count)
        with open_creating_dir_if_needed(path + "/output/data_iter_%i.txt" % count, "wb") as file:
            pickle.dump([boards, labels, results],  file)


def pickle_data(n, filename):
    d, l, r = get_data(n, "./output/pickle_results.txt")
    with open(filename, "wb") as file:
        pickle.dump([d, l, r], file)


def pickle_load(filename):
    with open(filename, "rb") as file:
        d, l, r = pickle.load(file)
    return d, l, r


# file_100_vanilla = "./output/data_1"
# pickle_data(100, file_100_vanilla)
# d, l, r = pickle_load(file_100_vanilla)

predict_moves.load("./checkpoints/checkpoint_iter_7.ckpt")

path = "./runs/run%02d" % time.time()
print(path)

train_indefinitely(10, 10, 1, path)