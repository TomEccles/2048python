import time

from board import Board, all_moves
from file_utils import open_creating_dir_if_needed


class GameBatchResult(object):
    def __init__(self, to_move_boards, moves, to_appear_boards, game_scores, action_values):
        self.to_move_boards = to_move_boards
        self.moves = moves
        self.to_appear_boards = to_appear_boards
        self.game_scores = game_scores
        self.action_values = action_values


def play_game(get_move):
    to_move_boards = []
    to_appear_boards = []
    labels = []
    b = Board()
    moves = 0
    start = time.time()
    while b.can_add_random():
        b = b.add_random()
        updated_board = get_move(b)
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


def no_values(moves):
    return [0.0 for i in range(moves)]


def get_data(games, get_move, values=no_values, results_file=None):
    """
    Plays some games of 2048, and returns everything that happened
    """
    to_move_boards = []
    to_appear_boards = []
    labels = []
    results = []
    move_scores = []
    for _ in range(games):
        score, t, boards, moves, a_boards = play_game(get_move)
        to_move_boards = to_move_boards + boards
        to_appear_boards = to_appear_boards + a_boards
        labels += moves
        results.append(score)
        move_scores += values(score)
        if results_file:
            with open_creating_dir_if_needed(results_file, 'a') as file:
                file.write("%d %f\n" % (score, t))
    return GameBatchResult(to_move_boards, labels, to_appear_boards, results, move_scores)
