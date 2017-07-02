import numpy

from board import all_moves, Board, Move


def compare(array):
    return numpy.array([(1 if array[i] > array[j] > 0 else 0) for i in range(16) for j in range(16)])


def non_zero(array):
    return numpy.array([(0 if i == 0 else 1) for i in array])


def can_move(board, move):
    copy = board.copy()
    copy.move(move)
    return copy != board


def possible_moves(board):
    return numpy.array([(1 if can_move(board, m) else 0) for m in all_moves])


def board_as_feature_array(board):
    board_array = numpy.reshape(board.board, 16)
    return numpy.concatenate(
        (numpy.copy(board_array) / 10.0, non_zero(board_array), compare(board_array), possible_moves(board)))


def move_as_one_hot_encoding(move):
    return numpy.array([(1 if move == i else 0) for i in range(4)])


def main():
    board = Board()
    board.add_random()
    board.add_random()
    board.move(Move.up)
    board.print()
    print(board_as_feature_array(board))
    for move in all_moves:
        print(move_as_one_hot_encoding(move))


if __name__ == '__main__':
    main()
