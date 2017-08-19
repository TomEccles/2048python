#pragma once
#include "Board.h"
#include <vector>

class Roller
{
    public:
        int rolloutFromMove(Board b);
        bool greedyMove(Board &b);
};

