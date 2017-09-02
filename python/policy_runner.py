import os

from policy_player import PolicyPlayer
import game_player


runs_root = "./results/runs"

def values(moves):
    r = 0.99
    return [1-r**(moves - i) for i in range(moves)]


def normalise(res):
    m = sum(res) / len(res)
    std = (sum([(r-m)**2 for r in res])/len(res)) ** 0.5
    return [(r-m) / std for r in res]


def train_policy(game_batch, passes, runs):
    load_path = "./results/runs/run1504073134/checkpoints/predict_iter_7.ckpt"

    trainer = PolicyPlayer(load_path)
    def get_move_policy(b):
        return b.move(trainer.get_move(b))
    r = 0
    max_average = 0
    while r != runs:
        to_move_boards, l, _, scores, res = game_player.get_data(game_batch, get_move_policy, values, runs_root + "/policy_test_3/results.txt")
        r += 1
        res = normalise(res)
        trainer.feed_observations(to_move_boards, l, res, passes)
        average = sum(scores) / len(scores)
        if average < max_average - 3:
            break
        else:
            max_average = max(max_average, average)
            trainer.save(runs_root + "/policy_test_4/checkpoint/checkpoint_%i.ckpt" % r)


train_policy(game_batch=1000, passes=10, runs=-1)