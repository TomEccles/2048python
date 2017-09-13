import random

import policy_runner
from file_utils import open_creating_dir_if_needed
from nets import Nets
import MCTS_runner
from policy_player import PolicyPlayer
from predict_moves import PolicyNetwork

# Cross cutting parameters
run_dir = "./results/runs/full_run"
evals = 100
value_weight = 10
prior_weight = 200

# Stage 1 parameters
s1_games = 1000
s1_training_passes = 10
s1_result_path = run_dir + "/output/s1_results.txt"
s1_policy_path = run_dir + "/checkpoint/s1_policy.ckpt"

# Stage 1 - play some fully vanilla games with MCTS, and train a policy network to emulate that
def run_stage_1():
    game_batch = MCTS_runner.get_vanilla_data(evals, s1_games, path=s1_result_path)
    nets = Nets(0, 0)
    nets.train_policy_net(game_batch, s1_training_passes)
    nets.policy_net.save(s1_policy_path)

# Stage 2 parameters
s2_batch_size = 1000
s2_training_passes = 10
s2_result_path=run_dir + "/output/s2_results_"
s2_policy_path = run_dir + "/checkpoint/s2_policy.ckpt"

# Stage 2 - do reinforcement learning to improve the policy network
def run_stage_2():
    policy_network = PolicyNetwork()
    policy_network.load(s1_policy_path)
    player = PolicyPlayer(policy_network)
    policy_runner.train_policy(player, s2_batch_size, s2_training_passes, -1, s2_result_path)
    policy_network.save(s2_policy_path)

# Stage 3 parameters
s3_games = 200
s3_valid_games = 10
s3_training_passes = 10
s3_results_path = run_dir + "/output/s3_results_"
s3_value_path = run_dir + "/checkpoint/s3_value.ckpt"

# Stage 3 - supervised learning of a value network
# The performance seems surprisingly dependent on having exactly the right network. Training against the stage 1 games
# gives much worse results
def run_stage_3():
    nets = Nets(prior_weight, 0, s2_policy_path)
    MCTS_runner.train_value_only(nets, s3_games, s3_training_passes, evals, s3_valid_games, s3_results_path, 1)
    nets.value_net.save(s3_value_path)

# Stage 4 - run it!
s4_results_path = run_dir + "/output/s4_results.txt"
def run_stage_4():
    nets = Nets(prior_weight, value_weight, policy_to_load=s2_policy_path, valuer_to_load=s3_value_path)
    while True:
        MCTS_runner.get_data(nets, evals, 1, s4_results_path)

search_results = run_dir + "/output/search_results.txt"
def search_weights():
    base_prior_weight = 100
    base_value_weight = 10
    max_prior_weight = 300
    max_value_weight = 30
    steps = 10
    nets = Nets(0, 0, policy_to_load=s2_policy_path, valuer_to_load=s3_value_path)
    while True:
        rand_prior_weight = random.randint(0, steps) * max_prior_weight/steps
        nets.value_weight = base_value_weight
        nets.prior_weight = rand_prior_weight
        result = MCTS_runner.get_data(nets, evals, 1, None)
        with open_creating_dir_if_needed(search_results, 'a') as file:
            file.write("%d %d %d\n" % (nets.prior_weight, nets.value_weight, result.game_scores[0]))

        rand_value_weight = random.randint(0, steps) * max_value_weight/steps
        nets.value_weight = rand_value_weight
        nets.prior_weight = base_prior_weight
        result = MCTS_runner.get_data(nets, evals, 1, None)
        with open_creating_dir_if_needed(search_results, 'a') as file:
            file.write("%d %d %d\n" % (nets.prior_weight, nets.value_weight, result.game_scores[0]))

search_weights()
