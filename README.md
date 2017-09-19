# 2048python
2048 bot. To determine the next move:
- Does a Monte Carlo tree search, guided by a neural network which has learned to predict the bot's moves
- Values leaves in the tree using rollouts and another neural network
The rollouts are in C++ for speed, everything else is Python with Tensorflow.

The rollout policy is not random - it tries to gather things in the top left:
- Move up if possible
- Else move left or right randomly - but prefer left if moving right would affect the top row
- Else move down
This produces better results than random or deterministic up>left>right>down

Approximate setup instructions
- Prerequisites are python 3.5, visual studio 2015
- Update Python2048Extension.vcxproj to have your python include path rather than mine (yes, sorry)
- Run build_extension.sh

full_training_run.py is the main file. This has a training process with four steps:
- Use MCTS with no neural networks to get a lot of data
- Use that to train the policy network with supervised learning
- Improve the policy network with policy gradients
- Use the resulting network to run some more games
- Use those to train a value network