import numpy as np
from bot import bot
import matplotlib.pyplot as plt
import time as t

np.random.seed(100)

test_bot = bot(1)
test_bot.read_brains()

if 1 == 0:
    test_modes = ["neural_net", "play_random_game"]
    network = test_bot.MCTS(test_bot.create_board(),-1,100000,test_modes[0])

    visits_mask = np.array([(node.number_visits>10) for node in network])

    xdata = np.array([node.node_board.flatten() for node in network])
    xdata = xdata + 0.05*(np.random.rand(np.shape(xdata)[0],np.shape(xdata)[1]) - 0.5) # to make the matrixes solvable
    xdata = xdata[visits_mask]

    ydata = np.array([np.array(node.total_val)/node.number_visits for node in network])
    ydata = ydata[visits_mask]
else:
    xdata = np.zeros((1,65))
    ydata = np.zeros((1,2))


def play_sample_game(xdata, ydata, sample_depth): # runs a random search for the MCTS
    board = test_bot.create_board()
    current_player = -1
    #xdata = np.concat((xdata,[np.append(board,[current_player])]), axis = 0)
    n = 0

    while True:
        # check if moves can be made
        moves = test_bot.valid_moves(board, current_player)

        if not moves:
            if not test_bot.valid_moves(board, -current_player): # stop game if no moves can be made by anyone
                break
            current_player = -current_player
            continue

        # make random moves to see how the game finnishes
        move = moves[np.random.randint(0,len(moves))]
        test_bot.make_move(board, current_player,move[0],move[1])

        # change player
        current_player = -current_player

        if n >= sample_depth:
            xdata = np.concat((xdata,[np.append(board,[current_player])]), axis = 0)
        n += 1

    if n > sample_depth:
        white_win_prob = (np.sum(board.flatten()) > 0)*1
        y_to_concat = np.ones((n - sample_depth,2))
        interpols = np.linspace(0.5,0.999,n)[sample_depth:]
        interpols = interpols[:, None]
        y_to_concat[:, 0] *= white_win_prob
        y_to_concat[:, 1] *= 1-white_win_prob  

        y_to_concat = y_to_concat*interpols + 0.5*np.ones((n - sample_depth,2))*(1-interpols)

        ydata = np.concat((ydata, y_to_concat), axis = 0)

    return xdata, ydata

start = t.time()
num_sample_games = 10000
for _ in range(num_sample_games):
    xdata, ydata = play_sample_game(xdata,ydata,58)
xdata = xdata[1:]
ydata = ydata[1:]
end = t.time()
print(f"training set took {end-start} seconds\n")

M = np.shape(xdata)[0]
N = np.shape(xdata)[1]
K = np.shape(ydata)[1]
print(np.shape(xdata))
print(np.shape(ydata))
print(M,N,K)

class data():
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x_vals = x
        self.y_vals = y

sample = data(xdata,ydata)

A = np.ones((M,N+1))
A[:, 1:] = sample.x_vals

At = np.transpose(A)
B = np.dot(At,A)

v = np.dot(At,np.log(sample.y_vals)*10) # the *10 acounts for the temperature of the softmax being set to 0.1
#v = np.dot(At,sample.y_vals*1000)

sol = np.array([np.linalg.solve(B,v[:,i]) for i in range(K)])
sol = np.transpose(sol)


# testing
test_xdata = np.zeros((1,65))
test_ydata = np.zeros((1,2))
start = t.time()
num_sample_games = 1000
for _ in range(num_sample_games):
    test_xdata, test_ydata = play_sample_game(test_xdata,test_ydata,58)
test_xdata = test_xdata[1:]
test_ydata = test_ydata[1:]
end = t.time()
print(f"testing set took {end-start} seconds\n")

predicts = np.dot(test_xdata,sol[1:])
predicts = predicts + sol[0]
predicts = np.array([test_bot.brain.softmax(predicts[i]) for i in range(len(predicts))])

errors = (test_ydata - predicts)**2

total_error = np.array([np.sum(errors[:,0]),np.sum(errors[:,1])])
mean_error = np.sqrt(total_error/len(errors[:,0]))

print()
print(mean_error)

corect_predicts = 0
for i in range(len(predicts)):
    if np.argmax(predicts[i]) == np.argmax(test_ydata[i]):
        corect_predicts += 1

print(f"acuracy rate was overall: {100*corect_predicts/len(predicts)}%")

fig, (ax1, ax2)  = plt.subplots(2,1)
ax1.plot(errors)
ax2.plot(test_ydata[:, 0])
ax2.plot(predicts[:, 0])

plt.show()