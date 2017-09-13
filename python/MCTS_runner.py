import pickle

import time

import numpy

import game_player
import file_utils
from board_features import board_as_feature_array
from move_player import MovePlayer
from nets import Nets, get_values

runs_root = "./results/runs"
vanilla_data_path = runs_root + "/vanilla/output/data"


def get_move_mcts(nets, evals):
    def move(b):
        return MovePlayer(b, nets).play(evals)

    return move


def get_data(nets, evals, games, path):
    """
    Gets some data with a specified value and policy network
    """
    return game_player.get_data(games, get_move_mcts(nets, evals), results_file=path)


def get_vanilla_data(evals, games, path):
    """
    Gets some data with a vanilla MCTS (no weight on the value or prior networks). Useful for getting some data to try
    training variations of networks on.
    """
    nets = Nets(0, 0)
    return game_player.get_data(games, get_move_mcts(nets, evals), results_file=path)


def train_value_only(nets, game_batch, passes, evals, valid_batch, path, runs=-1):
    """

    :param nets: The value and prior networks being used
    :param game_batch: The number of games to play between each training session
    :param passes: How many times to pass over the last bath of games when training
    :param evals: Evaluations in MCTS
    :param valid_batch: Number of games to validation after each training session
    :param path: Path to save results in
    :param runs: How many training sessions to do. -1 (default) is indefinite.
    :return:
    """
    count = 0
    while count != runs:
        count += 1
        train_batch = game_player.get_data(game_batch,
                                          get_move_mcts(nets, evals),
                                          results_file="%s%i.txt" % (path, count))

        valid_batch = game_player.get_data(valid_batch, get_move_mcts(nets, evals))

        nets.train_valuer(train_batch, passes)
        nets.validate_observations(valid_batch, train_batch)


def train(nets, game_batch, passes, evals, valid_batch, path, runs=-1):
    """

    :param nets: The value and prior networks being used
    :param game_batch: The number of games to play between each training session
    :param passes: How many times to pass over the last bath of games when training
    :param evals: Evaluations in MCTS
    :param valid_batch: Number of games to validation after each training session
    :param path: Path to save results in
    :param runs: How many training sessions to do. -1 (default) is indefinite.
    :return:
    """
    count = 0
    while count != runs:
        count += 1
        train_batch = game_player.get_data(game_batch,
                                            get_move_mcts(nets, evals),
                                            path + "/output/results_iter_%i.txt" % count)
        valid_batch = game_player.get_data(valid_batch, get_move_mcts(nets, evals))
        nets.feed_observations(train_batch, passes)
        nets.validate_observations(valid_batch, train_batch)
        nets.save(path + "/checkpoints/predict_iter_%i.ckpt" % count, path + "/checkpoints/value_iter_%i.ckpt" % count)
        with file_utils.open_creating_dir_if_needed(path + "/output/data_iter_%i.txt" % count, "wb") as file:
            pickle.dump([train_batch], file)


def train_from_vanilla(game_batch, evals, passes, valid_batch):
    run_path = runs_root + "/run%.0f" % time.time()
    nets = Nets(100, 10)
    train(nets, game_batch, passes, evals, valid_batch, run_path)


def test_outputs():
    run_path = runs_root + "/weight_test/"
    supervised_pred = runs_root + "/run1504073134/checkpoints/predict_iter_11.ckpt"
    value_from_sup = runs_root + "/run1504649146/checkpoints/value_iter_1.ckpt"
    evals = 100

    def play_10(prior_w, value_w):
        nets = Nets(prior_w, value_w, supervised_pred, value_from_sup)
        game_player.get_data(10, get_move_mcts(nets, evals),
                             results_file=run_path + "p-%i_v-%i.txt" % (prior_w, value_w))

    while True:
        play_10(100, 10)
        play_10(200, 10)
        play_10(100, 20)


def pickle_load(filename):
    with open(filename, "rb") as file:
        d, l, a, r = pickle.load(file)
    return d, l, a, r


#test_outputs()
