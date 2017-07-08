import pickle
import time

import numpy

import predict_moves
import predict_values
from board import Board, all_moves
from board_features import board_as_feature_array, move_as_one_hot_encoding, board_as_feature_array_with_sum
from file_utils import open_creating_dir_if_needed
from move_player import MovePlayer
from nets import Nets
from zero_valuer import ZeroValuer


def play_game(evals, nets):
    to_move_boards = []
    to_appear_boards = []
    labels = []
    b = Board()
    moves = 0
    start = time.time()
    while b.can_add_random():
        b.add_random()
        updated_board = MovePlayer(b, nets).play(evals)
        if updated_board is None:
            break

        for move in all_moves:
            if b.move(move) == updated_board:
                to_move_boards.append(b)
                labels.append(move)
                b = b.move(move)
                to_appear_boards.append(b)
                moves += 1
                break
        else:
            raise Exception("Move player has returned an illegal board!")
    return moves, time.time() - start, to_move_boards, labels, to_appear_boards


def get_data(n, nets, results_file=None):
    to_move_boards = []
    to_appear_boards = []
    l = []
    r = []
    for _ in range(n):
        moves, t, boards, labels, a_boards = play_game(100, nets)
        to_move_boards = to_move_boards + boards
        to_appear_boards = to_appear_boards + a_boards
        l = l + labels
        r.append(moves)
        if results_file:
            with open_creating_dir_if_needed(results_file, 'a') as file:
                file.write("%d %f\n" % (moves, t))
    return to_move_boards, l, to_appear_boards, r


def train_indefinitely(predictor, game_batch, passes, valid_batch, path="."):
    count = 0
    while True:
        count += 1
        boards, labels, results = get_data(game_batch, predictor, path + "/output/results_iter_%i.txt" % count)
        b_val, l_val, r_val = get_data(valid_batch, predictor)
        predictor.feed_observations(numpy.array([board_as_feature_array(board) for board in boards]),
                                    numpy.array([move_as_one_hot_encoding(l) for l in labels]),
                                    passes)
        predictor.validate_observations(numpy.array([board_as_feature_array(board) for board in b_val]),
                                        numpy.array([move_as_one_hot_encoding(l) for l in l_val]))
        predictor.save(path + "/checkpoints/checkpoint_iter_%i.ckpt" % count)
        with open_creating_dir_if_needed(path + "/output/data_iter_%i.txt" % count, "wb") as file:
            pickle.dump([boards, labels, results], file)


def pickle_data(n, filename):
    move_boards, labels, results, appear_boards = get_data(n, ZeroValuer(), "./output/pickle_results_2.txt")
    with open(filename, "wb") as file:
        pickle.dump([move_boards, labels, results, appear_boards], file)


def pickle_load(filename):
    with open(filename, "rb") as file:
        d, l, r = pickle.load(file)
    return d, l, r


def load_vanilla_data_and_train():
    file_vanilla = "./runs/vanilla/output/data"
    # pickle_data(500, file_vanilla)
    d, l, r = pickle_load(file_vanilla)
    print(len(d), len(l), len(r), d[0], l[0], r[0])
    predictor = predict_moves.PriorNet()
    split = int(len(d) * 0.9)
    predictor.feed_observations(numpy.array([board_as_feature_array(board) for board in d[:split]]),
                                numpy.array([move_as_one_hot_encoding(l) for l in l[:split]]),
                                10)
    predictor.validate_observations(numpy.array([board_as_feature_array(board) for board in d[split:]]),
                                    numpy.array([move_as_one_hot_encoding(l) for l in l[split:]]))
    predictor.save("./runs/vanilla/checkpoints/priors.ckpt")


def main():
    predictor = predict_moves.PriorNet()
    predictor.load("./runs/vanilla/checkpoints/priors.ckpt")

    path = "./runs/run%02d" % time.time()
    print(path)

    train_indefinitely(predictor, 100, 4, 10, path)


def initial_valuer():
    valuer = predict_values.ValuerNet()
    file_vanilla = "./runs/vanilla/output/data"
    # pickle_data(500, file_vanilla)
    boards, l, r = pickle_load(file_vanilla)
    print(len(boards), len(l), len(r), boards[0], l[0], r[0])
    split = int(len(boards) * 0.9)

    ends = []
    for index, board in enumerate(boards):
        if index > 0 and len(numpy.nonzero(board.board)[0]) == 1:
            ends.append(index - 1)
    ends.append(len(boards) - 1)
    game = 0
    values = []
    for index, board in enumerate(boards):
        if index > ends[game]:
            game += 1
        values.append(ends[game] - index)
    for i in range(10):
        valuer.feed_observations(numpy.array([board_as_feature_array_with_sum(board) for board in boards[:split]]),
                                 numpy.array(values[:split]),
                                 1)
        valuer.validate_observations(numpy.array([board_as_feature_array_with_sum(board) for board in boards[split:]]),
                                     numpy.array(values[split:]))
    valuer.save("./runs/vanilla/checkpoints/valuer.ckpt")

file_vanilla = "./runs/vanilla/output/data_2"

pickle_data(500, file_vanilla)