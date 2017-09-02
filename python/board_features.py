import numpy

from board import all_moves, Board, Move

sum_normalisation = 1000

def compare(array):
    return numpy.array([(1 if array[i] > array[j] > 0 else 0) for i in range(16) for j in range(16)])


def non_zero(array):
    return numpy.array([(0 if i == 0 else 1) for i in array])


def can_move(board, move):
    return board.move(move) != board


def possible_moves(board):
    return numpy.array([(1 if can_move(board, m) else 0) for m in all_moves])


def board_as_feature_array(board):
    board_array = numpy.reshape(board.board, 16)
    return numpy.concatenate(
        (numpy.copy(board_array) / 10.0, non_zero(board_array), compare(board_array), possible_moves(board),
         numpy.array([numpy.sum(board_array) / sum_normalisation])))


def move_as_one_hot_encoding(move, board_array, m=0):
    gates = board_array[-4:]
    n = numpy.sum(gates)
    return numpy.array([(1-(n-1)*m if move == i else m*gates[i]) for i in range(4)])


def main():
    board = Board()
    board = board.add_random()
    board = board.add_random()
    board = board.move(Move.up)
    board.print()
    print(board_as_feature_array(board))
    print(board_as_feature_array(board))
    for move in all_moves:
        print(move_as_one_hot_encoding(move))


if __name__ == '__main__':
    main()
