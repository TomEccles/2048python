import pickle
import time

from board import Board, all_moves
from file_utils import open_creating_dir_if_needed
from move_player import MovePlayer
from nets import Nets

runs_root = "~/PycharmProjects/2048python-new/results/runs"
vanilla_data_path = runs_root + "/vanilla/output/data"

def play_game(evals, nets):
    to_move_boards = []
    to_appear_boards = []
    labels = []
    b = Board()
    moves = 0
    start = time.time()
    while b.can_add_random():
        b = b.add_random()
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


def get_data(n, nets, turns, results_file=None):
    to_move_boards = []
    to_appear_boards = []
    l = []
    r = []
    for _ in range(n):
        moves, t, boards, labels, a_boards = play_game(turns, nets)
        to_move_boards = to_move_boards + boards
        to_appear_boards = to_appear_boards + a_boards
        l = l + labels
        r.append(moves)
        if results_file:
            with open_creating_dir_if_needed(results_file, 'a') as file:
                file.write("%d %f\n" % (moves, t))
    return to_move_boards, l, to_appear_boards, r


def train(nets, game_batch, passes, turns, valid_batch, path, runs=-1):
    count = 0
    while count != runs:
        count += 1
        boards, labels, a_boards, results = get_data(game_batch, nets, turns,
                                                     path + "/output/results_iter_%i.txt" % count)
        b_val, l_val, a_val, r_val = get_data(valid_batch, nets, turns)
        nets.feed_observations(boards, labels, a_boards, passes)
        nets.validate_observations(b_val, l_val, a_val)
        nets.save(path + "/checkpoints/predict_iter_%i.ckpt" % count, path + "/checkpoints/value_iter_%i.ckpt" % count)
        with open_creating_dir_if_needed(path + "/output/data_iter_%i.txt" % count, "wb") as file:
            pickle.dump([boards, labels, a_boards, results], file)


def get_vanilla_data(turns, games):
    nets = Nets(0, 0)
    boards, labels, a_boards, results = get_data(games, nets, turns)
    with open_creating_dir_if_needed(vanilla_data_path, "w") as file:
        pickle.dump([boards, labels, a_boards, results], file)

def pickle_load(filename):
    with open(filename, "rb") as file:
        d, l, a, r = pickle.load(file)
    return d, l, a, r


def load_vanilla_data_and_train(file):
    d, l, a, r = pickle_load(file)
    print(len(d), len(l), len(a), len(r), d[0], l[0], a[0], r[0])
    nets = Nets(0, 0, None, None)
    feed_obs = 100000
    valid_obs = 10000
    end = feed_obs + valid_obs
    nets.feed_observations(d[:feed_obs], l[:feed_obs], a[:feed_obs], 10)
    nets.validate_observations(d[feed_obs:end], l[feed_obs:end], a[feed_obs:end])
    nets.save(runs_root + "/vanilla/checkpoints/predict.ckpt", runs_root + "/vanilla/checkpoints/value.ckpt")


def test_outputs():
    load_path = runs_root + "/run1499671533/checkpoints/"
    run_path = runs_root + "/test100/"
    turns = 100
    while True:
        nets = Nets(100, 10, load_path + "predict_iter_11.ckpt", load_path + "value_iter_11.ckpt")
        get_data(10, nets, turns, run_path + "last_iter.txt")

        nets = Nets(0, 0, runs_root + "/vanilla/checkpoints/predict.ckpt", runs_root + "/vanilla/checkpoints/value.ckpt")
        get_data(10, nets, turns, run_path + "control.txt")

        # nets = Nets(100, 10, "./runs/vanilla/checkpoints/predict.ckpt", "./runs/vanilla/checkpoints/value.ckpt")
        # get_data(10, nets, path + "first_iter.txt")

        # nets = Nets(100, 0, load_path + "predict_iter_11.ckpt", load_path +"value_iter_11.ckpt")
        # get_data(10, nets, path + "last_iter_no_value.txt")
        #
        # nets = Nets(0, 10, load_path + "predict_iter_11.ckpt", load_path +"value_iter_11.ckpt")
        # get_data(10, nets, path + "last_iter_no_prior.txt")


def train_from_vanilla():
    vanilla_data_path = runs_root + "/vanilla/output/data"
    load_vanilla_data_and_train(vanilla_data_path)
    run_path = runs_root + "/run%f/" % time.time()
    nets = Nets(100, 10)
    train(nets, game_batch=1, turns=100, passes=1, valid_batch=1, path=run_path, runs=1)


def run_one_game():
    path = "./runs/test/"
    nets = Nets(10, 10, runs_root + "/vanilla/checkpoints/predict.ckpt", runs_root + "/vanilla/checkpoints/value.ckpt")
    get_data(1, nets, 100, path + "control.txt")


# cProfile.run('run_one_game()')
train_from_vanilla()
