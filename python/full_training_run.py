import policy_runner
from nets import Nets
import MCTS_runner

# Cross cutting parameters
from policy_player import PolicyPlayer
from predict_moves import PolicyNetwork

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
def s2_result_path(iter):
    return run_dir + "/output/s2_results_%i.txt" % iter
s2_policy_path = run_dir + "/checkpoint/s2_policy.ckpt"

# Stage 2 - do reinforcement learning to improve the policy network
def run_stage_2():
    policy_network = PolicyNetwork()
    policy_network.load(s1_policy_path)
    player = PolicyPlayer(policy_network)
    policy_runner.train_policy(player, s2_batch_size, s2_training_passes, -1, s2_result_path)
    policy_network.save(s2_policy_path)

# Stage 3 parameters
s3_games = 100
s3_valid_games = 10
s3_training_passes = 10
s3_results_path = run_dir + "/output/s3_results.txt"
s3_value_path = run_dir + "/checkpoint/s3_value.ckpt"

# Stage 3 - supervised learning of a value network
# The performance seems surprisingly dependent on having exactly the right network. Training against the stage 1 games
# gives much worse results
def run_stage_3():
    nets = Nets(prior_weight, 0, s2_policy_path)
    result_batch = MCTS_runner.get_data(nets, evals, s3_games, path=s3_results_path)
    valid_bath = MCTS_runner.get_data(nets, evals, s3_valid_games, path=s3_results_path)
    nets.train_valuer(result_batch, s3_training_passes)
    nets.validate_observations(valid_bath, result_batch)
    nets.value_net.save(s3_value_path)

# Stage 4 - run it!
s4_results_path = run_dir + "/output/s4_results.txt"
def run_stage_4():
    nets = Nets(prior_weight, value_weight, policy_to_load=s2_policy_path, valuer_to_load=s3_value_path)
    while True:
        MCTS_runner.get_data(nets, evals, 1, s4_results_path)

run_stage_3()
run_stage_4()