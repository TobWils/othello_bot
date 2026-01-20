import numpy as np
from MLP import MLP
import time as t

class bot():
    def __init__(self, bot_player_idx):
        self.brain = MLP(64,2,"n",8, [64,64,64,64,64,64,64])
        self.BOARD_SIZE = 8
        self.train_w_wins = 0
        self.train_b_wins = 0
        self.is_player = bot_player_idx # 1 coresponds to being white player, change to -1 for black
        self.exploration_const = 3 # is used for MCTS algorithm, specificaly the UCB1 function. larger values seem to make the tree search slower but more acurate it seems
    
    def save_brains(self):
        for i in range(self.brain.layers):
            bias_location = "bot brains/bias_" + str(i + 1)
            weight_location = "bot brains/weight_" + str(i + 1)
            self.brain.wright_bias(i, bias_location)
            self.brain.wright_matrix(i, weight_location)
    
    def read_brains(self):
        for i in range(self.brain.layers):
            bias_location = "bot brains/bias_" + str(i + 1)
            weight_location = "bot brains/weight_" + str(i + 1)
            self.brain.read_bias(i, bias_location)
            self.brain.read_matrix(i, weight_location)

    def evaluate_board(self,board):
        return self.brain.propigate(board.flatten())

    def evaluate_moves(self, board, player, eval_mode:str):
        if eval_mode == "neural_net":
            """Return all valid moves for the player."""
            opponent = -player
            directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
            moves = np.array([[0,0]],dtype=np.ndarray)

            for r in range(self.BOARD_SIZE):
                for c in range(self.BOARD_SIZE):
                    if board[r][c] != 0:
                        continue
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        found_opponent = False
                        while 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == opponent:
                            nr += dr
                            nc += dc
                            found_opponent = True
                        if found_opponent and 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == player:
                            moves = np.concat((moves,np.array([[r,c]])),axis=0)
                            break
            moves = moves[1:]
            
            scores = np.array(moves,dtype=np.ndarray) # probably a better way of doing this like incorporating it into the main loop
            
            for i in range(len(moves)):
                temp_board = np.array(board)
                self.make_move(temp_board, player, moves[i][0], moves[i][1]) # fixed the probelm of all the temp_boards being the same, the net eval has some other issue here > problem was with matrix decleration in the MLP has been fixed now
                #scores[i] = np.sum(temp_board.flatten())
                scores[i] = self.brain.propigate_withought_softmax(temp_board.flatten())

            return moves[np.argmax(scores.transpose() * player)] # the *player bit is to acount for the negative values coresponding to black
    
        elif eval_mode == "MCTS_neural_net":
            tree = self.MCTS(board, player, 64*384, "neural_net")

            #move_nodes = [tree[tree[0].child_node_idxs[i]] for i in range(len(tree[0].child_node_idxs))]
            move_nodes = [tree[idx] for idx in tree[0].child_node_idxs]

            if player == 1:
                #scores = [move_nodes[i].total_val[0]/move_nodes[i].number_visits for i in range(len(move_nodes))]
                scores = [idx.total_val[0]/idx.number_visits for idx in move_nodes]
            else:
                #scores = [move_nodes[i].total_val[1]/move_nodes[i].number_visits for i in range(len(move_nodes))]
                scores = [idx.total_val[1]/idx.number_visits for idx in move_nodes]

            return move_nodes[np.argmax(scores)].node_board
        
        elif eval_mode == "MCTS_random":
            tree = self.MCTS(board, player, 192, "play_random_game")

            move_nodes = [tree[tree[0].child_node_idxs[i]] for i in range(len(tree[0].child_node_idxs))]

            if player == 1:
                scores = [move_nodes[i].total_val/move_nodes[i].number_visits for i in range(len(move_nodes))]
            else:
                scores = [1 - move_nodes[i].total_val/move_nodes[i].number_visits for i in range(len(move_nodes))]

            return move_nodes[np.argmax(scores)].node_board

    def play_training_game(self): # does not currently work for some reason
        #board = self.create_board()
        player_idx = -1
        #game = [np.array(board)]
        game = np.zeros((61,8,8), dtype=np.float64)
        game[0] = self.create_board()
        i = 0

        while True:
            # check if moves can be made
            moves = self.valid_moves(game[i], player_idx)

            if not moves:
                if not self.valid_moves(game[i], -player_idx): # stop game if no moves can be made by anyone
                    break
                player_idx = -player_idx
                continue

            # make moves
            if True:
                if np.random.randint(0,5) < 5: # the partialy random action is to alow the bot to see more game posibilities in training
                    move = moves[np.random.randint(0,len(moves))]
                else:
                    move = self.evaluate_moves(np.copy(game[i]), player_idx, "neural_net")

                board = np.copy(game[i])
                self.make_move(board, player_idx,move[0],move[1])
                game[i+1] = np.copy(board)
                
            else:
                game[i+1] = self.evaluate_moves(game[i], player_idx, "MCTS_neural_net")

            # change player
            player_idx = -player_idx

            #game.append(np.array(board))
            i += 1
        
        #game = game[3:]
        winner = max(min(np.sum(np.copy(game[i]).flatten()),0.5),-0.5) + 0.5 # must change later to be probability of white winning, see MCTS play_tree_game() function

        self.train_w_wins += winner
        self.train_b_wins += 1-winner
        #for turn in game:
        for n in range(i):
            self.brain.back_propigate_once_cross_entropy_Adam(np.copy(game[n]).flatten(),[winner, 1-winner])

    # monte carlo tree search

    def play_tree_game(self, Board, current_player): # runs a random search for the MCTS
        board = np.copy(Board)

        while True:
            # check if moves can be made
            moves = self.valid_moves(board, current_player)

            if not moves:
                if not self.valid_moves(board, -current_player): # stop game if no moves can be made by anyone
                    break
                current_player = -current_player
                continue

            # make random moves to see how the game finnishes
            move = moves[np.random.randint(0,len(moves))]
            self.make_move(board, current_player,move[0],move[1])

            # change player
            current_player = -current_player

        white_win_prob = max(min(np.sum(board.flatten()),0.5),-0.5) + 0.5

        return np.array([white_win_prob, 1-white_win_prob]) # using probability of white winning as output metric

    def UCB1(self, total_val, number_visits, number_parent_visits): # is used in the MCTS algorithm for cell evaluation
        return (total_val/number_visits) + (self.exploration_const*np.sqrt(np.log(number_parent_visits)/number_visits))

    class node(): # probably should replace later with parralel arrays but use this for now to get it working
        def __init__(self, total_val:np.ndarray, number_visits:np.int64, parent_node_idx:np.int64, node_idx:np.int64, board:np.ndarray, player:np.int64):
            self.total_val = total_val
            self.number_visits = number_visits
            self.parent_node_idx = parent_node_idx
            self.node_idx = node_idx # this is probably unnesasery but ill remove it later if its an issue
            self.child_node_idxs = []
            self.is_terminal_node = False # if this turns out to be true the value should be adjusted in the simulation step, dont know if this is standard but its what im doing
            self.node_board = board
            self.node_player = player

    def MCTS(self, board, player, iters, simulation_mode: str):
        #initalisation
        nodes = [self.node([0,0], 0, -1, 0, board, player)]

        for i in range(iters):
            #selection
            keep_selecting = True
            selection_idx = 0 # could change the selecion index as moves get made to be able to cut parts of the search tree out
            while keep_selecting:
                childeren = nodes[selection_idx].child_node_idxs
                parent_visits = nodes[selection_idx].number_visits
                if not childeren:
                    keep_selecting = False
                else:
                    child_vals = np.array([0]*len(childeren))
                    for i in range(len(childeren)):
                        child = nodes[childeren[i]]
                        if child.number_visits == 0: # exists to prevent division by 0 in the UCB1 value calcs as if a node hasent been viseted it must be prioratised over nodes that have
                            selection_idx = child.node_idx # this method doesnt randomise the selection for unvisited nodes so could be improved in future
                            keep_selecting = False
                            break
                        elif not child.is_terminal_node:
                            if nodes[selection_idx].node_player == -1:
                                child_vals[i] = self.UCB1(child.total_val[1]/child.number_visits, child.number_visits, parent_visits) # child.total_val/child.number_visits is the probability that white wins
                            else:
                                child_vals[i] = self.UCB1(child.total_val[0]/child.number_visits, child.number_visits, parent_visits)
                    
                    if keep_selecting:
                        selection_idx = childeren[np.argmax(child_vals)]

            #expansion
            if not nodes[selection_idx].is_terminal_node:
                new_moves = self.valid_moves(nodes[selection_idx].node_board, nodes[selection_idx].node_player)
                if not new_moves:
                    if not self.valid_moves(nodes[selection_idx].node_board, -nodes[selection_idx].node_player): # deals with game finnishes
                        nodes[selection_idx].is_terminal_node = True

                        if nodes[selection_idx].number_visits == 0:
                            #skip to backpropigation
                            value = (nodes[selection_idx].node_player + 1)/2
                            value = np.array([value, 1-value])
                            nodes[selection_idx].total_val += value
                            nodes[selection_idx].number_visits += 1
                            back_prop_idx = nodes[selection_idx].parent_node_idx
                            while back_prop_idx != -1:
                                nodes[back_prop_idx].total_val += value
                                nodes[back_prop_idx].number_visits += 1
                                back_prop_idx = nodes[back_prop_idx].parent_node_idx
                        
                        continue

                    else: # deals with positions you cant move in
                        current_player = -nodes[selection_idx].node_player
                        init_board = np.copy(nodes[selection_idx].node_board)
                        sim_node_idx = len(nodes)
                        nodes.append(self.node(0, 0, selection_idx, sim_node_idx, init_board, current_player))
                        nodes[selection_idx].child_node_idxs.append(sim_node_idx)

                else:
                    current_player = -nodes[selection_idx].node_player
                    for i in range(len(new_moves)):
                        init_board = np.copy(nodes[selection_idx].node_board)
                        self.make_move(init_board, -current_player, new_moves[i][0], new_moves[i][1])
                        nodes.append(self.node([0,0], 0, selection_idx, len(nodes), init_board, current_player))
                        nodes[selection_idx].child_node_idxs.append(len(nodes)-1)
                    
                    sim_node_idx = np.random.randint(len(nodes) - len(new_moves), len(nodes)) # this will be the node that we simulae the oucome of in the next step
                    
                #simulation
                if simulation_mode == "neural_net":
                    nodes[sim_node_idx].total_val = self.evaluate_board(nodes[sim_node_idx].node_board) # evaluation using neural nets to be quick, a more expensive sim could be done maybey repurpouse the play game tree function for this
                elif simulation_mode == "play_random_game":
                    nodes[sim_node_idx].total_val = self.play_tree_game(nodes[sim_node_idx].node_board, current_player) # evaluation using random game, should be relativly fast and is more acurate
                
                #backpropigation
                nodes[sim_node_idx].number_visits = 1 # shold work as the value previously should have been 0 so 0+1=1
                back_prop_idx = nodes[sim_node_idx].parent_node_idx
                while back_prop_idx != -1:
                    nodes[back_prop_idx].total_val += nodes[sim_node_idx].total_val
                    nodes[back_prop_idx].number_visits += 1
                    back_prop_idx = nodes[back_prop_idx].parent_node_idx

        return nodes
    
    def MCTS_train(self, tree_iters:int, tree_mode:str, save_as_go:bool):
        tree = self.MCTS(self.create_board(), -1, tree_iters, tree_mode) # only plays as black
        np.random.shuffle(tree)

        t1 = t.time()

        for Node in tree:
            if Node.number_visits > 5:
                #print(f"prob:{Node.total_val/Node.number_visits}|vals:{Node.total_val,Node.number_visits}")
                #for _ in range(Node.number_visits): # max(int(100*Node.number_visits/tree_iters),1)
                self.brain.back_propigate_once_cross_entropy_Adam(Node.node_board.flatten(), Node.total_val/Node.number_visits)
            
            t2 = t.time()
            if t2 - t1 > 1800 and save_as_go: # saves progress every 30 min roughly (1800 sec)
                self.save_brains()
                t1 = t.time()

    # Othello (Reversi) - Console Version
    # Two-player version (Black = X, White = O) Black = -1 White = 1, black gets index 0 white gets index 1

    def create_board(self):
        board: np.ndarray = np.zeros((self.BOARD_SIZE,self.BOARD_SIZE))#np.array([[ 0 for _ in range(self.BOARD_SIZE)]for _ in range(self.BOARD_SIZE)])
        mid = self.BOARD_SIZE // 2
        board[mid - 1][mid - 1] = 1
        board[mid][mid] = 1
        board[mid - 1][mid] = -1
        board[mid][mid - 1] = -1
        return board

    def print_board(self,board):
        """Display the board."""
        board = np.array(np.astype(board, np.int8), dtype= str)
        board = np.char.replace(board, "0", " ")
        board = np.char.replace(board, "-1", "X")
        board = np.char.replace(board, "1", "O")
        print("    " + "   ".join(str(i) for i in range(self.BOARD_SIZE)))
        for i, row in enumerate(board):
            print(i, str(row))

    def valid_moves(self,board, player):
        """Return all valid moves for the player."""
        opponent = -player
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        moves = []

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r][c] != 0:
                    continue
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == opponent:
                        nr += dr
                        nc += dc
                        found_opponent = True
                    if found_opponent and 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == player:
                        moves.append([r, c])
                        break
        return moves

    def make_move(self, board:np.ndarray, player, row, col):
        """Place a piece and flip opponent pieces."""
        opponent = 1 if player == -1 else -1
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        board[row][col] = player

        for dr, dc in directions:
            tiles_to_flip = []
            nr, nc = row + dr, col + dc
            while 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == opponent:
                tiles_to_flip.append((nr, nc))
                nr += dr
                nc += dc
            if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == player:
                for rr, cc in tiles_to_flip:
                    board[rr][cc] = player

    def score(self,board):
        """Return the count of X and O."""
        board = np.array(board, dtype= str)
        board = np.char.replace(board, "0", " ")
        board = np.char.replace(board, "-1", "X")
        board = np.char.replace(board, "1", "O")
        x = np.sum(np.strings.count(board,"X"))
        o = np.sum(np.strings.count(board,"O"))
        return x, o

    def main(self):
        board = self.create_board()
        #current_player = 'X'
        player_idx = -1 # black starts in othello

        while True:
            moves = self.valid_moves(board, player_idx)

            if not moves:
                if not self.valid_moves(board, -player_idx):
                        break
                print("No valid moves, skipping turn.\n")
                #current_player = 'O' if current_player == 'X' else 'X'
                player_idx = -player_idx
                continue

            if player_idx != self.is_player:
                self.print_board(board)
                x_count, o_count = self.score(board)
                print(f"Score → O: {o_count}, X: {x_count}")
                print(f"{ 'X' if player_idx == -1 else 'O' }'s turn| prediction: {self.evaluate_board(board)}")


                print("Valid moves:", moves)
                try:
                    row, col = map(int, input("Enter row and column (e.g., 2 3): ").split())
                    if [row, col] not in moves:
                        print("Invalid move. Try again.\n")
                        continue
                except:
                    print("Invalid input. Use two numbers like '2 3'.\n")
                    continue

                self.make_move(board, player_idx, row, col)

            elif player_idx == self.is_player:
                board = self.evaluate_moves(board, self.is_player, "MCTS_neural_net")

            #current_player = 'O' if current_player == 'X' else 'X'
            player_idx = -player_idx

        self.print_board(board)
        x_count, o_count = self.score(board)
        print(f"Final Score → X: {x_count}, O: {o_count}")
        if x_count > o_count:
            print("X wins!")
        elif o_count > x_count:
            print("O wins!")
        else:
            print("It's a tie!")

np.random.seed(1000)

test = bot(1) # initalise bot

train_bot = True # wether to train the bot or use precalculated weights and biases to speed up neural net testing in future
retrain_bot = True
train_MCTS_mode = True
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
        start = t.time()
        for i in range(300): # trains bot
            test.play_training_game()
        end = t.time()
        print(end - start) # avereages ~0.2126666021347046 seconds per game in training (100 games in 21.26666021347046 sec) with a 2 layer bot with 16 neurons in each layer
        print(test.train_w_wins/(test.train_w_wins+test.train_b_wins))
        print(test.train_w_wins)
        print(test.train_b_wins)
        print()

    test.save_brains()
else:
    test.read_brains()

if 1 == 1:
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

test.main()