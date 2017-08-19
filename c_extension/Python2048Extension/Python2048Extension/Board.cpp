#include "stdafx.h"
#include "Board.h"
#include <iostream>
#include <string>

typedef std::array<int, side * side> numbers;

static int squashed[15][15][15][15][4] = {};
static bool done[15][15][15][15] = {};
static bool change[15][15][15][15] = {};

Board::Board()
{
    board = {};
}

Board::Board(long inNumbers[side][side])
{
    for (int i = 0; i < side; i++) {
        for (int j = 0; j < side; j++) {
            board[side*i + j] = inNumbers[i][j];
        }
    }
}

Board::~Board()
{
}

std::string Board::moveIndicatorString()
{
    std::string s = "";
    for (Move move : allMoves)
    {
        s.append(std::to_string(canMove(move) ? 1 : 0));
        s.append(",");
    }
    return s;
}

bool Board::move(Move move)
{
    switch (move)
    {
    case Move::UP:
        return moveUp();
    case Move::DOWN:
        return moveDown();
    case Move::LEFT:
        return moveLeft();
    case Move::RIGHT:
        return moveRight();
    default:
        return false;
    }
}

bool Board::topRowCanMoveRight()
{
    bool seenZero = false;
    bool seenNonZero = false;
    int last = 0;
    for (int i = 3; i >= 0; i--) {
        int next = board[i];
        if (next != 0) {
            if (seenZero) return true;
            if (seenNonZero && next == last) return true;
            seenNonZero = true;
            last = next;
        }
        else seenZero = true;
    }
    return false;
}

// This must be called on squashed arrays - it fails if there are zeros in the middle
bool Board::consolidateInPlace(int start, int gap)
{
    int writeMoves = 0;
    int readMoves = 0;
    bool change = false;
    while (readMoves < side)
    {
        int readPosition = start + readMoves * gap;
        int value = board[readPosition];
        if (value == 0) break;

        int writePosition = start + writeMoves * gap;

        if (readMoves + 1 < side && value == board[readPosition + gap])
        {
            board[writePosition] = value + 1;
            readMoves += 2;
            writeMoves += 1;
            change = true;
        }
        else
        {
            board[writePosition] = value;
            readMoves += 1;
            writeMoves += 1;
        }
    }

    for (; writeMoves < side; writeMoves++)
    {
        board[start + writeMoves * gap] = 0;
    }
    return change;
}

bool Board::removeZerosInPlace(int start, int gap)
{
    int writeMoves = 0;
    bool zero = false;
    bool change = false;

    for (int i = 0; i < side; i++)
    {
        int value = board[start + gap * i];
        if (value != 0)
        {
            board[start + writeMoves * gap] = value;
            writeMoves += 1;
            if (zero) change = true;
        }
        else zero = true;
    }
    for (; writeMoves < side; writeMoves++)
    {
        board[start + writeMoves * gap] = 0;
    }
    return change;
}

// TODO: this is an utter mess. It's currently built to be both a quick in place evaluator, and do caching.
// Probably easier to not bother doing the evaluation in place (as speed isn't an issue with the cache).
bool Board::squashInPlace(int start, int gap)
{
    int *row[4];
    for (int i = 0; i < 4; i++) row[i] = &board[start + gap*i];
    if (done[*row[0]][*row[1]][*row[2]][*row[3]]) {
        int oldValues[4];
        for (int i = 0; i < 4; i++)
            oldValues[i] = *row[i];
        for (int i = 0; i < 4; i++)
            *row[i] = squashed[oldValues[0]][oldValues[1]][oldValues[2]][oldValues[3]][i];
        return change[oldValues[0]][oldValues[1]][oldValues[2]][oldValues[3]];
    }
    else {
        int o[4];
        for (int i = 0; i < 4; i++) o[i] = *row[i];
        
        bool c = false;
        c |= removeZerosInPlace(start, gap);
        c |= consolidateInPlace(start, gap);

        done[o[0]][o[1]][o[2]][o[3]] = true;
        change[o[0]][o[1]][o[2]][o[3]] = c;
        for (int i = 0; i < 4; i++)
            squashed[o[0]][o[1]][o[2]][o[3]][i] = *row[i];
        return c;
    }
}

bool Board::moveLeft()
{
    bool change = false;
    for (int i = 0; i < side; i++)
    {
        change |= squashInPlace(4*i, 1);
    }
    return change;
}

bool Board::moveRight()
{
    bool change = false;
    for (int i = 0; i < side; i++)
    {
        change |= squashInPlace(4 * i + 3, -1);
    }
    return change;
}

bool Board::moveUp()
{
    bool change = false;
    for (int i = 0; i < side; i++)
    {
        change |= squashInPlace(i, 4);
    }
    return change;
}

bool Board::moveDown()
{
    bool change = false;
    for (int i = 0; i < side; i++)
    {
        change |= squashInPlace(i + 12, -4);
    }
    return change;
}

bool Board::canMove(Move move)
{
    Board newBoard = *this;
    return newBoard.move(move);
}

int zeroes(const std::array<int, side * side> &arr) {
    int count = 0;
    for (int i : arr)
    {
        if (i == 0) count++;
    }
    return count;
}

void modifyNthZero(std::array<int, side * side> &arr, int index, int number)
{
    int count = 0;
    for (int &i : arr)
    {
        if (i == 0)
        {
            if (index == count)
            {
                i = number;
                break;
            }
            count++;
        }
    }
}

bool Board::addRandom()
{
    int count = zeroes(board);
    if (count == 0) return false;
    int index = rand() % count;
    int number = (rand() % 10 == 0) ? 2 : 1;
    modifyNthZero(board, index, number);
    return true;
}

int Board::elementAt(int i, int j)
{
    return board[4*i+j];
}

void Board::print(int indent)
{
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < indent; j++) std::cout << "--";
        for (int j = 0; j < side; j++)
        {
            std::cout << board[i * side + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

std::string Board::toString()
{
    std::string s = "";
    for (int i = 0; i < side * side; i++)
    {
        s.append(std::to_string(board[i]));
        s.push_back(',');
    }
    return s;
}

std::string Board::normalisedString()
{
    std::string s = "";
    for (int i = 0; i < side * side; i++)
    {
        int norm = board[i] > 0 ? 1 : 0;
        s.append(std::to_string(norm));
        s.push_back(',');
    }
    return s;
}

//TODO: 36 is a magic number
std::array<int, 36> Board::calcArray()
{
    std::array<int, 36> arr;
    for (int i = 0; i < 16; i++) {
        arr[i] = board[i];
        arr[i + 16] = board[i] > 0 ? 1 : 0;
    }

    int index = 0;
    for (Move move : allMoves)
    {
        arr[32 + index] = canMove(move) ? 1 : 0;
        index++;
    }
    return arr;
}

//TODO: 37 is a magic number
std::array<int, 37> Board::calcArrayWithSum()
{
    std::array<int, 37> arr;
    for (int i = 0; i < 16; i++) {
        arr[i] = board[i];
        arr[i + 16] = board[i] > 0 ? 1 : 0;
    }

    int index = 0;
    for (Move move : allMoves)
    {
        arr[32 + index] = canMove(move) ? 1 : 0;
        index++;
    }

    arr[36] = 0;
    for (int i : board)
    {
       if (i > 0) arr[36] += 1 << i;
    }

    return arr;
}


bool Board::equals(Board other)
{
    return board == other.board;
}
