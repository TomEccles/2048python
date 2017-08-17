import cProfile
import pickle
import time

from board import Board, all_moves
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


def get_data(n, nets, results_file=None):
    to_move_boards = []
    to_appear_boards = []
    l = []
    r = []
    for _ in range(n):
        moves, t, boards, labels, a_boards = play_game(1000, nets)
        to_move_boards = to_move_boards + boards
        to_appear_boards = to_appear_boards + a_boards
        l = l + labels
        r.append(moves)
        if results_file:
            with open_creating_dir_if_needed(results_file, 'a') as file:
                file.write("%d %f\n" % (moves, t))
    return to_move_boards, l, to_appear_boards, r


def train(nets, game_batch, passes, valid_batch, path=".", runs=-1):
    count = 0
    while count != runs:
        count += 1
        boards, labels, a_boards, results = get_data(game_batch, nets, path + "/output/results_iter_%i.txt" % count)
        b_val, l_val, a_val, r_val = get_data(valid_batch, nets)
        nets.feed_observations(boards, labels, a_boards, passes)
        nets.validate_observations(b_val, l_val, a_val)
        nets.save(path + "/checkpoints/predict_iter_%i.ckpt" % count, path + "/checkpoints/value_iter_%i.ckpt" % count)
        with open_creating_dir_if_needed(path + "/output/data_iter_%i.txt" % count, "wb") as file:
            pickle.dump([boards, labels, a_boards, results], file)


def pickle_data(n, filename):
    move_boards, labels, appear_boards, results = get_data(n, ZeroValuer(), "./output/pickle_results_2.txt")
    with open(filename, "wb") as file:
        pickle.dump([move_boards, labels, appear_boards, results], file)


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
    nets.save("./runs/vanilla/checkpoints/predict.ckpt", "./runs/vanilla/checkpoints/value.ckpt")


def main():
    path = "./runs/test2/"
    while True:
        nets = Nets(100, 10, "./runs/run1499671533/checkpoints/predict_iter_11.ckpt",
                    "./runs/run1499671533/checkpoints/value_iter_11.ckpt")
        get_data(10, nets, path + "last_iter.txt")

        nets = Nets(0, 0, "./runs/vanilla/checkpoints/predict.ckpt", "./runs/vanilla/checkpoints/value.ckpt")
        get_data(10, nets, path + "control.txt")

        # nets = Nets(100, 10, "./runs/vanilla/checkpoints/predict.ckpt", "./runs/vanilla/checkpoints/value.ckpt")
        # get_data(10, nets, path + "first_iter.txt")

        # nets = Nets(100, 0, "./runs/run1499671533/checkpoints/predict_iter_11.ckpt", "./runs/run1499671533/checkpoints/value_iter_11.ckpt")
        # get_data(10, nets, path + "last_iter_no_value.txt")
        #
        # nets = Nets(0, 10, "./runs/run1499671533/checkpoints/predict_iter_11.ckpt", "./runs/run1499671533/checkpoints/value_iter_11.ckpt")
        # get_data(10, nets, path + "last_iter_no_prior.txt")


def run_one_game():
    path = "./runs/test/"
    nets = Nets(10, 10, "./runs/vanilla/checkpoints/predict.ckpt", "./runs/vanilla/checkpoints/value.ckpt")
    get_data(1, nets, path + "control.txt")


# cProfile.run('run_one_game()')
main()
