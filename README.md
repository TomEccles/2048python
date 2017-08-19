# 2048python
2048 bot. To determine the next move:
- Does a Monte Carlo tree search, guided by a neural network which has learned to predict the bot's moves
- Values leaves in the tree using rollouts and another neural network
The rollouts are in C++ for speed, everything else is Python with Tensorflow.
