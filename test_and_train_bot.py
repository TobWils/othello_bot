from bot import bot
import numpy as np
import time as t
import matplotlib.pyplot as plt

np.random.seed(600)

test = bot(1) # initalise bot

train_bot = True # wether to train the bot or use precalculated weights and biases to speed up neural net testing in future
retrain_bot = False
train_MCTS_mode = False
if train_bot: # probably add something to cut out files that aren't needed
    if not retrain_bot:
        test.read_brains()

    if train_MCTS_mode:
        start = t.time()
        test.MCTS_train(256*256*384, "play_random_game", True)
        end = t.time()
        print(end - start)
        print()

    else:
        train_iters = 300 # at 30,000 should take about 1 hour so let it run in the background for a bit
        start = t.time()
        for i in range(int(train_iters/2)): # trains bot
            test.play_training_game(np.random.randint(1,55))
            test.play_training_game(60)
        end = t.time()
        print(end - start) # avereages ~0.2126666021347046 seconds per game in training (100 games in 21.26666021347046 sec) with a 2 layer bot with 16 neurons in each layer
        print(test.train_w_wins/(test.train_w_wins+test.train_b_wins))
        print(test.train_w_wins)
        print(test.train_b_wins)
        print()

    #test.save_brains()
else:
    test.read_brains()

    if 0 == 1:
        scaling_test_iters = 15
        scaling_test_start = 0
        scaling_test_stop = 15000
        scaling_test_lengths = np.linspace(scaling_test_start,scaling_test_stop,scaling_test_iters, dtype=np.int64)
        scaling_test_times = np.zeros(scaling_test_iters)

        for i in range(scaling_test_iters):
            scaling_test_bot = bot(1)
            start = t.time()
            for ii in range(scaling_test_lengths[i]):
                scaling_test_bot.play_training_game(1)
            end = t.time()
            scaling_test_times[i] = end - start
            print(f"completed scaling test itr:{i} | it lasted for {scaling_test_lengths[i]} iterations")
        
        plt.plot(scaling_test_times)
        plt.show()
        print()

# testing
test_iters = 500
test_modes = ["neural_net", "MCTS_neural_net", "MCTS_random"]
start = t.time()
for i in range(test_iters):
    test.play_testing_game(test_modes[0], test_modes[0],15)
end = t.time()
test_acuracy = test.test_correct_predict/test.test_num_predict
total_acuracy = np.sum(test.test_correct_predict)/np.sum(test.test_num_predict)

print(end - start) # avereages ~0.2126666021347046 seconds per game in training (100 games in 21.26666021347046 sec) with a 2 layer bot with 16 neurons in each layer
print(f"average testing acuracy was: {str(np.round(total_acuracy,4)*100)[:5]}% \naverage test speed was: {(end-start)/test_iters} sec/itr")
print()

if train_bot:
    fig, (ax1, ax2)  = plt.subplots(1,2)
    ax1.set_ylim(0,1)
    ax1.plot(0.5*np.ones(60))
    ax1.plot(test_acuracy)
    ax2.plot(test.brain.loss)
else:
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_ylim(0,1)
    ax1.plot(0.5*np.ones(60))
    ax1.plot(test_acuracy)
plt.show()

if 1 == 0:
    sim_modes = ["neural_net", "play_random_game"]

    start = t.time()
    network = test.MCTS(test.create_board(), -1, 4*384, sim_modes[0]) # player should always be -1 as black goes first this is not to say the player input should be removed though as this is a secial case where we are starting at the same position each time
    end = t.time()
    # the MCTS runs in about 0.58 - 0.61 seconds using a 4 layer [16,12,4] neural net to simulate game outcomes and covers 9346 board states with 4*384 iterations
    # the MCTS runs in about 10.25 - 10.5 seconds using random decision making to simulate game outcomes and covers 9308 board states with 4*384 iterations

    for i in range(min(len(network),32)):
        print(network[i].node_board)
        print(network[i].node_player)
        print(f"estimated (W win | B win): {network[i].total_val/network[i].number_visits}")
        print(network[i].total_val)
        print(network[i].number_visits)
        print()

    print(len(network))
    print(end - start)
