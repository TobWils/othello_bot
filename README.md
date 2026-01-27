this is a repo for my othello bot to store all the relavent files and have a virtual environment with the dependancies clearly listed

use the .new_venv rather than the regular .venv as .venv is old, isent up to date with the code, has the wrong libraries and more

requirements file should be good now


--ISTRUCTIONS ON HOW TO PLAY THE BOT--

if you want to play the bot use the play_bot.py python file it should work and start a game

in othello black goes first not white and the bot playes as white by defult

if you want the bot to play as black then set line 3 in play_bot.py to be: Bot = bot(-1)
black is represented by X on the board

if you want the bot to play as white then set line 3 in play_bot.py to be: Bot = bot(1)
white is represented by O on the board


--ISTRUCTIONS FOR TESTING AND TRAINING THE BOT--

the test_and_train_bot.py file is the main python file for this.
it has various variables that are intended to be changed to quickly get the bot to do diferent things while testing and training.

the first 3 of these you will encounter are on lines 10, 11 and 12.
they are:

10 - train_bot # whether or not to train the bot, if set to false it will read the weights into the bot by defult

11 - retrain_bot # whether to generate new random parameters, seed values are set in most programes for repeatability however

12 - train_MCTS_mode # this one is complicated but it will basicaly run an MCTS for a while to get decent evals of board positions and then train the net on those results, problem is that when you do this the resulting networks dont actualy play very well at the moment so is set to False for now


there are 2 further main parameters that are used for training and testing:

25 - train_iters # how many trainning games the bot should train on, if set to 0 there will be a divide by 0 error so istead use train_bot = False

41 - test_iters # how many testing games the bot should be evaluated on, at about 10,000 it takes about 75 seconds ish


there are more parameters in the code that dont have names but can be changed to get diferent results depending on what your looking at or for like on line 56 theres an "if 1 == 0:" statement that exists to give an easy choice betwene kernel modes for smoothing the test results so they are actualy readable and on line 8 where ive set the bot to play as white by defult

the funny looking thing at the bottom of the code is to test the MCTS (Monty Carlo Tree Search) algorithm on diferent "modes" either using random games or the neural net in its simulation step and evaluating how fast it goes and giving you the evaluations on the first few board states of a game, this by defult wont run