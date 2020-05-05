# Tic_tac_toe_Reinforcement_Learning

## Overview
This is a repo that allows you to train, and play tic-tac-toe with an Reinforcement Learning AI. The algorithm uses a State value based Upper-confidence bound Reinforcement Learning, based on a Markov decision process. 

There are two modes in which this can be run: Training and Playing. In the training mode, the AI plays with itself, for any number of rounds (typically several tens of thousands), and goes on learning. This can be launched with **Play_and_train.py**, and creates an excel file that contains the state values of the AI. This can be used for the _Playing mode_. In the playing mode, a human can play with the AI, which occurs without any learning on the part of the AI. This can be launched with **Play_game.py**. These parts are completed and can be used.

## Features in progress
There are a few new components being added to the program. Stand-by for changes in a few weeks to these parts:
* GUI: Currently the game is played via formatted prints in the command-line. A GUI is planned to be added soon. 
* Hyper-parameter tuning: The trained state values have a large margin of improvements. The training epochs, and learning rates are key parameters.
* Arbitrary dimensionality of the game board: Currently it is 3 by 3. Want to extend it to nxn limited only by computational power. 
* Deep Q Network: Currently the AI uses UCB (Upper Confidence Bound) as part of the Markov process. Using a Deep Q-learning would be quite useful, especially for higher dimensions. 

## Future applications
This was intended as a ramp-up project to familiarize with the concepts and applications of RL. The ultimate vision for this project is to use this on a _Control Problem_, especially for _chaotic systems_. Stay tuned for more updates!
