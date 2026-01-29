from bot import bot
import numpy as np
import time as t
import matplotlib.pyplot as plt

np.random.seed(500)

test = bot(1) # initalise bot

train_bot = True # wether to train the bot or use precalculated weights and biases to speed up neural net testing in future
retrain_bot = True
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
        train_iters = 30000 # at 30,000 should take about 1 hour so let it run in the background for a bit
        start = t.time()
        for i in range(train_iters): # trains bot
            test.play_training_game(np.random.randint(1,46))
        end = t.time()
        print(end - start) # avereages ~0.2126666021347046 seconds per game in training (100 games in 21.26666021347046 sec) with a 2 layer bot with 16 neurons in each layer
        print(test.train_w_wins/(test.train_w_wins+test.train_b_wins))
        print(test.train_w_wins)
        print(test.train_b_wins)
        print()

    test.save_brains()
else:
    test.read_brains()

# testing, managed to get an over all 68.4% acuracy but its varying betwene that and about 65%
test_iters = 10000
test.test_acuracy = np.zeros(test_iters)
prediction_dist = np.zeros(test_iters)
start = t.time()
for i in range(test_iters):
    test.play_testing_game(i, int(59*(i/test_iters) + 1))
    prediction_dist[i] = int(59*(i/test_iters) + 1)
end = t.time()

print(end - start) # avereages ~0.2126666021347046 seconds per game in training (100 games in 21.26666021347046 sec) with a 2 layer bot with 16 neurons in each layer
print(f"average testing acuracy was: {str(np.round(np.average(test.test_acuracy),4)*100)[:5]}% \naverage test speed was: {(end-start)/test_iters} sec/itr")
print()


kernel_size = int(test_iters/20)
if 1 == 0:
    smoothing_kernel = np.arange(kernel_size)
    for i in range(int(kernel_size/2),kernel_size):
        smoothing_kernel[i] = kernel_size - i

    smoothing_kernel = smoothing_kernel/np.sum(smoothing_kernel)
else:
    smoothing_kernel = np.ones(kernel_size)/kernel_size

smoothed_acuracy = np.convolve(test.test_acuracy,smoothing_kernel,'valid')
smoothed_distances = np.convolve(prediction_dist,smoothing_kernel,'valid')

averaged_acuracy = np.zeros(59)
for i in range(test_iters):
    averaged_acuracy[int(prediction_dist[i]-1)] += test.test_acuracy[i]

averaged_acuracy = averaged_acuracy*59/test_iters

if train_bot:
    plt.subplot(221)
    plt.plot(smoothed_distances,smoothed_acuracy)
    plt.subplot(223)
    plt.plot(averaged_acuracy)
    plt.subplot(122)
    plt.plot(test.brain.loss)
else:
    plt.subplot(211)
    plt.plot(smoothed_distances,smoothed_acuracy)
    plt.subplot(212)
    plt.plot(averaged_acuracy)
plt.show()

if 0 == 1:
    sim_modes = ["neural_net", "play_random_game"]

    start = t.time()
    network = test.MCTS(test.create_board(), -1, 4*384, sim_modes[0]) # player should always be -1 as black goes first this is not to say the player input should be removed though as this is a secial case where we are starting at the same position each time
    end = t.time()
    # the MCTS runs in about 0.58 - 0.61 seconds using a 4 layer [16,12,4] neural net to simulate game outcomes and covers 9346 board states with 4*384 iterations
    # the MCTS runs in about 10.25 - 10.5 seconds using random decision making to simulate game outcomes and covers 9308 board states with 4*384 iterations

    for i in range(min(len(network),5)):
        print(network[i].node_board)
        print(network[i].node_player)
        print(f"estimated (W win | B win): {network[i].total_val/network[i].number_visits}")
        print(network[i].total_val)
        print(network[i].number_visits)
        print()

    print(len(network))
    print(end - start)
