# this is an implementation of the alpha zero algorithm based on a tutorial i found implementing it with tick tack toe and conect 4
# quite a lot of the code from there is unchanged as of now and im still finishing adapting it
# the link to the tutorial is: https://www.youtube.com/watch?v=wuSQpLinRB4

import numpy as np
import matplotlib.pyplot as plt
import time as t

import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.manual_seed(0)


class othello():
    def __init__(self):
        self.BOARD_SIZE = 8
        self.ACTION_SIZE = self.BOARD_SIZE*self.BOARD_SIZE

    # Othello (Reversi) - Console Version
    # Two-player version (Black = X, White = O) Black = 1 White = -1

    def create_board(self):
        board: np.ndarray = np.zeros((self.BOARD_SIZE,self.BOARD_SIZE))#np.array([[ 0 for _ in range(self.BOARD_SIZE)]for _ in range(self.BOARD_SIZE)])
        mid = self.BOARD_SIZE // 2
        board[mid - 1][mid - 1] = -1
        board[mid][mid] = -1
        board[mid - 1][mid] = 1
        board[mid][mid - 1] = 1
        return board

    def print_board(self,board):
        """Display the board."""
        board = np.array(np.astype(board, np.int8), dtype= str)
        board = np.char.replace(board, "0", " ")
        board = np.char.replace(board, "-1", "O")
        board = np.char.replace(board, "1", "X")
        print("    " + "   ".join(str(i) for i in range(self.BOARD_SIZE)))
        for i, row in enumerate(board):
            print(i, str(row))

    def valid_moves(self,board, player):
        opponent = -player
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        moves = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))

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
                        moves[r][c] = 1
                        break
        return moves.flatten()

    def make_move(self, board:np.ndarray, player, move):
        row = move // self.BOARD_SIZE
        col = move % self.BOARD_SIZE
        if self.valid_moves(board,player)[move] == 1:

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
        
            return board
        
        else:
            print("--------")
            print(self.valid_moves(board,player).reshape((8,8)), row, col, player)
            self.print_board(board)
            print("--------")
            raise Exception(f'Invalid move {move}')
            #try:
            #    fhjdshjds
            #except ArithmeticError as e:
            #    raise Exception

    def score(self,board):
        """Return the count of X and O."""
        board = np.array(board, dtype= str)
        board = np.char.replace(board, "0", " ")
        board = np.char.replace(board, "-1", "O")
        board = np.char.replace(board, "1", "X")
        x = np.sum(np.strings.count(board,"X"))
        o = np.sum(np.strings.count(board,"O"))
        return x, o

    def check_win(self, board, player_idx):
        moves = self.valid_moves(board, -player_idx)
        if np.sum(moves) == 0:
            moves = self.valid_moves(board, player_idx)
            if np.sum(moves) == 0:
                score = player_idx*np.sum(board)
                return score > 0
        return False

    def get_value_and_terminated(self, board, player):
        if self.check_win(board, player):
            return 1, True
        if np.sum(self.valid_moves(board, -player)) == 0:
            if np.sum(self.valid_moves(board, player)) == 0:
                return 0, True
        return 0, False

    def change_perspective(self, board, player):
        return board*player

    def get_encoded_board(self, board):
        encoded_state = np.stack(
            (board == -1, board == 0, board == 1)
        ).astype(np.float32)
        return encoded_state

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()

        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, game: othello, num_res_blocks, num_hidden, device):
        super().__init__()

        self.device = device

        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*game.ACTION_SIZE, game.ACTION_SIZE)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*game.ACTION_SIZE, 1),
            nn.Tanh()
        )

        self.to(device)
    
    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.backbone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class Node():
    def __init__(self, game: othello, args, board, player, parent=None, move=None, prior=0, number_visits=0):
        self.game = game
        self.args = args
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.prior = prior

        self.childeren = []

        self.number_visits = number_visits
        self.total_val = 0
    
    def is_fully_expanded(self):
        return len(self.childeren) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.childeren:
            ucb = self.UCB(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def UCB(self, child):
        if child.number_visits == 0:
            q_value = 0
        else:
            q_value = 1 - (child.total_val/child.number_visits + 1)/2 # want to be a number in range 0 to 1 and to be low so that the win prob is minimised for oponent
        return q_value + child.prior*self.args["C"]*np.sqrt(self.number_visits/(child.number_visits + 1))
    
    def expand(self, policy):
        if np.sum(policy) != 0:
            for move, prob in enumerate(policy):
                if prob > 0:
                    child_board = self.board.copy()
                    child_board = self.game.make_move(child_board, 1, move)
                    child_board = self.game.change_perspective(child_board, player = -1) # switch perspectives so that freindly peices are always 1, this is done before moves as the available moves depends on the curent player
                    child_player = -self.player

                    child = Node(self.game, self.args, child_board, child_player, self, move, prob)
                    #print(child.move)
                    if child.move == None:
                        raise Exception(f"invalid None type move {child.parent}\n{np.reshape(policy,(8,8))}")
                    self.childeren.append(child)
        #else:
        #    child = Node(self.game, self.args, self.board, -self.player, parent=self, prior=1)
        #    self.childeren.append(child)
    
    def backpropigate(self, value):
        self.total_val += value
        self.number_visits += 1

        value = -value
        if self.parent is not None:
            self.parent.backpropigate(value)

class MCTS():
    def __init__(self, game: othello, args, model: ResNet):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, board, player):
        board = self.game.change_perspective(board,player)
        root = Node(self.game, self.args, board, player, number_visits=1)

        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_board(board),
                device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"])*policy + self.args["dirichlet_epsilon"]*np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.ACTION_SIZE)
        
        valid_moves = self.game.valid_moves(board, 1)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)


        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select()
            
            if node.move == None:
                self.game.print_board(node.parent.board)
                print(self.game.get_value_and_terminated(node.parent.board, node.parent.player))
                raise Exception(f"invalid None type move {node.parent.childeren}")
            
            if self.game.valid_moves(node.parent.board,1)[node.move] == 1:

                value, is_terminal = self.game.get_value_and_terminated(node.parent.board, 1)
                value = -value # as the move stored in a node is the move made by the oponent so the value should be inverted

                if not is_terminal:
                    policy, value = self.model(
                        torch.tensor(
                            self.game.get_encoded_board(node.board),
                            device=self.model.device
                        ).unsqueeze(0)
                    )
                    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                    valid_moves = self.game.valid_moves(node.board,1)
                    policy *= valid_moves
                    if np.sum(policy) != 0:
                        policy /= np.sum(policy)

                    value = value.item()

                    node.expand(policy)
                
                node.backpropigate(value)

        action_probs = np.zeros(self.game.ACTION_SIZE)
        for child in root.childeren:
            action_probs[child.move] = child.number_visits
        action_probs /= np.sum(action_probs)

        return action_probs

class AlphaZero():
    def __init__(self, model: ResNet, optimiser: torch.optim, game: othello, args):
        self.model = model
        self.optimiser = optimiser
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        self.train_loss = []
    
    def self_play(self):
        memory = []
        player = 1
        board = self.game.create_board()
        countr = 0
        tot_time = 0

        while True:
            #print("v player v")
            #print(player)
            if np.sum(self.game.valid_moves(board, player)) !=0:
                neutral_board = self.game.change_perspective(board, player)
                start = t.time()
                action_probs = self.mcts.search(neutral_board, 1)
                end = t.time()
                countr += 1
                tot_time += end - start

                memory.append((neutral_board, action_probs, player))
                #self.game.print_board(board)

                temperature_action_probs = action_probs ** (1/self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)
                move = np.random.choice(self.game.ACTION_SIZE, p=temperature_action_probs)

                board = self.game.make_move(board, player, move)

                value, is_terminal = self.game.get_value_and_terminated(board, player)
                #try:
                #except:
                #    raise Exception(f"Invalid move {np.reshape(temperature_action_probs,(8,8))}\n{np.reshape(action_probs,(8,8))}")

                if is_terminal:
                    return_memory = []
                    winner = 1 if self.game.check_win(board,1) else ( -1 if self.game.check_win(board,-1) else 0)
                    for hist_neutral_board, hist_action_probs, hist_player in memory:
                        hist_outcome = winner if hist_player == winner else -winner
                        return_memory.append((
                            torch.tensor(
                                self.game.get_encoded_board(hist_neutral_board)
                            ),
                            hist_action_probs,
                            hist_outcome
                        ))
                    
                    print(f"avg time: {tot_time/countr}| count: {countr}| total time: {tot_time}\nvalue: {winner}")
                    print(self.game.score(board)[0] - self.game.score(board)[1])
                    return return_memory
            
            player = -player

    def train(self, memory):
        np.random.shuffle(memory)
        running_loss = 0
        num_evals = 0
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx : min(len(memory)-1,batchIdx + self.args["batch_size"])]
            if len(sample) != 0:
                boards, policy_targets, value_targets = zip(*sample)

                boards, policy_targets, value_targets = np.array(boards), np.array(policy_targets), np.array(value_targets).reshape(-1,1)

                boards = torch.tensor(boards, dtype=torch.float32, device=self.model.device)
                policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
                value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

                out_policy, out_value = self.model(boards)

                policy_loss = F.cross_entropy(out_policy,policy_targets)
                value_loss = F.mse_loss(out_value, value_targets)

                loss = policy_loss + value_loss
                running_loss += loss.item()
                num_evals += 1

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
        
        running_loss = running_loss/num_evals
        self.train_loss.append(running_loss)

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            print(f"\n----{iteration}----\n- self play -")
            memory = []

            self.model.eval()
            start = t.time()
            for self_play_iteration in range(self.args["num_self_play_iterations"]):
                memory += self.self_play()
            end = t.time()
            print(f"time taken was: {end - start} seconds")
            
            print("\n- train -")
            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                self.train(memory)
            print(self.train_loss[-1])
            
            torch.save(self.model.state_dict(), f"torch_bot_save_file/model_{iteration}.pt")
            torch.save(self.optimiser.state_dict(), f"torch_bot_save_file/optimiser_{iteration}.pt")


def test_model():
    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")

    Othello = othello()

    args = {
        "C": 2,
        "num_searches": 1000,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    board = Othello.create_board()
    player = 1
    for i in range(30):
        moves = Othello.valid_moves(board, player)
        if np.sum(moves) != 0:
            moves = [i for i in range(len(moves)) if moves[i] == 1]
            rand_move = moves[np.random.randint(0,len(moves))]
            board = Othello.make_move(board,player,rand_move)
        player = -player
    
    while np.sum(Othello.valid_moves(board, player)) < 2:
        moves = Othello.valid_moves(board, player)
        if np.sum(moves) != 0:
            moves = [i for i in range(len(moves)) if moves[i] == 1]
            rand_move = moves[np.random.randint(0,len(moves))]
            board = Othello.make_move(board,player,rand_move)
        player = -player

    Othello.print_board(board)
    print(f"player is: {player}")
    print()

    board = Othello.change_perspective(board, player)

    encoded_board = Othello.get_encoded_board(board)

    print(encoded_board)

    tensor_board = torch.tensor(encoded_board).unsqueeze(0).to(device)

    model = ResNet(Othello, 4, 64, device)
    model.load_state_dict(torch.load("torch_bot_save_file/model_2.pt"))
    model.eval()
    mcts = MCTS(Othello, args, model)

    policy, value = model(tensor_board)
    value = value.item()
    policy = torch.softmax(policy, axis = 1).squeeze(0).detach().cpu().numpy()
    policy = policy*Othello.valid_moves(board,1)# + min(policy)*0.5
    policy = policy.reshape((8,8))

    print(f"model valuation: {value}")
    print("the player is represented by the lighter colours in the image on the right")
    print("the image on the top left is the log10 of the model policy (after making with valid moves)")
    print("the image on the bottom left is the log10 of the mcts policy")

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(224)

    board_background = np.array([((i%2 - 0.5) * (i//8%2 - 0.5))*0.4*(board[i//8][i%8] == 0) for i in range(64)]).reshape((8,8))
    img_board = board + board_background
    ax1.imshow(img_board)
    ax2.imshow(np.log10(policy))

    mcts_policy = mcts.search(board, 1)
    mcts_policy = mcts_policy.reshape((8,8))
    print(f"{mcts_policy}")
    ax3.imshow(np.log10(mcts_policy))

    plt.show()

def display_model_evaluation(board, player, model: ResNet, mcts: MCTS, Othello: othello, fig: plt.Figure, board_plot, eval_plot1, eval_plot2, device = "cpu"):
    board = Othello.change_perspective(board, player)

    encoded_board = Othello.get_encoded_board(board)

    tensor_board = torch.tensor(encoded_board).unsqueeze(0).to(device)

    policy, value = model(tensor_board)
    value = value.item()
    print(f"model evaluation is: {value}")
    policy = torch.softmax(policy, axis = 1).squeeze(0).detach().cpu().numpy()
    policy = policy*Othello.valid_moves(board,1)
    policy = policy.reshape((8,8))

    board_background = np.array([((i%2 - 0.5) * (i//8%2 - 0.5))*0.4*(board[i//8][i%8] == 0) for i in range(64)]).reshape((8,8))
    img_board = board + board_background
    board_plot.set_data(img_board)
    eval_plot1.set_data(np.log10(policy))

    mcts_policy = mcts.search(board, 1)
    mcts_policy = mcts_policy.reshape((8,8))
    eval_plot2.set_data(np.log10(mcts_policy))

    fig.canvas.draw()
    fig.canvas.flush_events()

def display_init_eval(board, player, model: ResNet, mcts: MCTS, Othello: othello, ax1,ax2,ax3, device = "cpu"):
    board = Othello.change_perspective(board, player)

    encoded_board = Othello.get_encoded_board(board)

    tensor_board = torch.tensor(encoded_board).unsqueeze(0).to(device)

    policy, value = model(tensor_board)
    value = value.item()
    print(f"model evaluation is: {value}")
    policy = torch.softmax(policy, axis = 1).squeeze(0).detach().cpu().numpy()
    policy = policy*Othello.valid_moves(board,1)
    policy = policy.reshape((8,8))

    board_background = np.array([((i%2 - 0.5) * (i//8%2 - 0.5))*0.4*(board[i//8][i%8] == 0) for i in range(64)]).reshape((8,8))
    img_board = board + board_background
    board_plot = ax1.imshow(img_board)
    eval_plot1 = ax2.imshow(np.log10(policy))

    mcts_policy = mcts.search(board, 1)
    mcts_policy = mcts_policy.reshape((8,8))
    eval_plot2 = ax3.imshow(np.log10(mcts_policy))

    plt.show(block = False)
    return board_plot, eval_plot1, eval_plot2

def main():
    game = othello()
    player = 1
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)


    args = {
        "C": 2,
        "num_searches": 10000,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    model = ResNet(game, 4, 64, "cpu")
    model.load_state_dict(torch.load("torch_bot_save_file/model_2.pt"))
    model.eval()

    mcts = MCTS(game, args, model)

    board = game.create_board()

    board_plt, evl_plt1, evl_plt2 = display_init_eval(board, player, model, mcts, game, ax1,ax2,ax3)

    while True:
        game.print_board(board)

        if player == 1:
            display_model_evaluation(board, player, model, mcts, game, fig, board_plt, evl_plt1, evl_plt2)
            valid_moves = game.valid_moves(board, player)
            print(f"valid moves: {[[i//game.BOARD_SIZE, i%game.BOARD_SIZE] for i in range(game.ACTION_SIZE) if valid_moves[i] == 1]}")
            row = int(input(f"enter {player}'s move row: "))
            col = int(input(f"enter {player}'s move col: "))

            move = row*game.BOARD_SIZE + col
            if valid_moves[move] == 0:
                print("move not valid")
                continue
        else:
            neutral_board = game.change_perspective(board, player)
            mcts_probs = mcts.search(neutral_board, 1)
            move = np.argmax(mcts_probs)

        board = game.make_move(board, player, move)

        value, terminated = game.get_value_and_terminated(board, player)

        if terminated:
            game.print_board(board)
            if value == 1:
                print(f"player {player} won")
                break
            else:
                print("draw")
        
        player = -player

def test_alphazero():
    Othello = othello()

    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")

    args = {
        "C": 4,
        "num_searches": 60,
        "num_iterations": 9,
        "num_self_play_iterations": 10,
        "num_epochs": 32,
        "batch_size": 8,
        "temperature": 3.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    origonal_args = {
        "C": 2,
        "num_searches": 60,
        "num_iterations": 3,
        "num_self_play_iterations": 10,
        "num_epochs": 4,
        "batch_size": 2,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    model = ResNet(Othello, 4, 64, device)
    #model.load_state_dict(torch.load(f"torch_bot_save_file/model_{args['num_iterations']-1}.pt"))

    optimiser = torch.optim.Adam(model.parameters(), lr = 10**(-3), weight_decay=10**(-4))
    #optimiser.load_state_dict(torch.load(f"torch_bot_save_file/optimiser_{args['num_iterations']-1}.pt"))

    alphaZero = AlphaZero(model, optimiser, Othello, args)

    start = t.time()
    alphaZero.learn()
    end = t.time()

    print(f"time taken was: {end - start}")
    plt.plot(alphaZero.train_loss)
    plt.show()

test_alphazero()
#test_model()
#main()