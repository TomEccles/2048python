import os

from policy_player import PolicyPlayer
import game_player
from predict_moves import PolicyNetwork

runs_root = "./results/runs"

def values(moves):
    r = 0.99
    return [1-r**(moves - i) for i in range(moves)]


def normalise(res):
    m = sum(res) / len(res)
    std = (sum([(r-m)**2 for r in res])/len(res)) ** 0.5
    return [(r-m) / std for r in res]


def train_policy(player, game_batch, passes, runs, result_path):
    def get_move_policy(b):
        return b.move(player.get_move(b))
    r = 0
    max_average = 0
    while r != runs:
        batch_result = game_player.get_data(game_batch, get_move_policy, values, "%s%i.txt" % (result_path, r))
        r += 1
        normalised_action_vals = normalise(batch_result.action_values)
        player.feed_observations(batch_result.to_move_boards, batch_result.moves, normalised_action_vals, passes)
        average = sum(batch_result.game_scores) / len(batch_result.game_scores)
        if average < max_average - 3:
            break
        else:
            max_average = max(max_average, average)


# load_path = "./results/runs/run1504073134/checkpoints/predict_iter_7.ckpt"
# net = PolicyNetwork()
# net.load(load_path)
# trainer = PolicyPlayer(net)
# train_policy(trainer, game_batch=1000, passes=10, runs=-1)