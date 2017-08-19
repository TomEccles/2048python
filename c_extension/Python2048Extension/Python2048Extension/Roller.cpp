#include "stdafx.h"
#include "Roller.h"
#include "Board.h"
#include <iostream>

bool Roller::greedyMove(Board &rolloutBoard)
{
    
    if (rolloutBoard.topRowCanMoveRight() || (rand() % 2) == 0) {
        return
            rolloutBoard.move(Move::UP) ||
            rolloutBoard.move(Move::LEFT) ||
            rolloutBoard.move(Move::RIGHT) ||
            rolloutBoard.move(Move::DOWN);
    }
    
    return
        rolloutBoard.move(Move::UP) ||
        rolloutBoard.move(Move::RIGHT) ||
        rolloutBoard.move(Move::LEFT) ||
        rolloutBoard.move(Move::DOWN);
}

int Roller::rolloutFromMove(Board rolloutBoard)
{
    int turns = 0;
    greedyMove(rolloutBoard);
    while (rolloutBoard.addRandom())
    {
        turns += 1;
        greedyMove(rolloutBoard);
    }
    return turns;
}
