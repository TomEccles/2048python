import pickle

import time

import game_player
import file_utils
from move_player import MovePlayer
from nets import Nets


runs_root = "./results/runs"
vanilla_data_path = runs_root + "/vanilla/output/data"


def get_move_mcts(nets, evals):
    def move(b):
        return MovePlayer(b, nets).play(evals)
    return move


def get_vanilla_data(evals, games):
    """
    Gets some data with a vanilla MCTS (no weight on the value or prior networks). Useful for getting some data to try
    training variations of networks on.
    """
    nets = Nets(0, 0)

    boards, labels, a_boards, results, _ = game_player.get_data(games, get_move_mcts(nets, evals), runs_root + "/vanilla/output/results.txt")
    with file_utils.open_creating_dir_if_needed(vanilla_data_path, "wb") as file:
        pickle.dump([boards, labels, a_boards, results], file)


def load_vanilla_data_and_train():
    """
    Loads the vanilla data created by get_vanilla_data, and trains a network on it. Useful when we want to tweak
    networks and see how well they train.
    """
    d, l, a, r = pickle_load(vanilla_data_path)
    print(len(d), len(l), len(a), len(r), d[0], l[0], a[0], r[0])
    nets = Nets(0, 0, None, None)
    feed_obs = 100000
    valid_obs = 20000

    end = feed_obs + valid_obs

    if end > min(len(d), len(l), len(a)):
        raise ValueError("Not enough data")

    while True:
        nets.feed_observations(d[:feed_obs], l[:feed_obs], a[:feed_obs], 10)
        nets.validate_observations(d[feed_obs:end], l[feed_obs:end], a[feed_obs:end])



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
        boards, labels, a_boards, results, _ = game_player.get_data(game_batch,
                                                     get_move_mcts(nets, evals),
                                                     path + "/output/results_iter_%i.txt" % count)
        b_val, l_val, a_val, r_val, _ = game_player.get_data(valid_batch, get_move_mcts(nets, evals))
        nets.feed_observations(boards, labels, a_boards, passes)
        nets.validate_observations(b_val, l_val, a_val)
        nets.save(path + "/checkpoints/predict_iter_%i.ckpt" % count, path + "/checkpoints/value_iter_%i.ckpt" % count)
        with file_utils.open_creating_dir_if_needed(path + "/output/data_iter_%i.txt" % count, "wb") as file:
            pickle.dump([boards, labels, a_boards, results], file)


def train_from_vanilla(game_batch, evals, passes, valid_batch):
    run_path = runs_root + "/run%.0f" % time.time()
    nets = Nets(100, 10)
    train(nets, game_batch, passes, evals, valid_batch, run_path)


def test_outputs():
    load_path = runs_root + "/run1499671533/checkpoints/"
    run_path = runs_root + "/test/"
    evals = 100
    while True:
        nets = Nets(100, 10, load_path + "predict_iter_11.ckpt", load_path + "value_iter_11.ckpt")
        game_player.get_data(10, get_move_mcts(nets, evals), run_path + "last_iter.txt")


def pickle_load(filename):
    with open(filename, "rb") as file:
        d, l, a, r = pickle.load(file)
    return d, l, a, r


train_from_vanilla(100, 100, 10, 10)