# this is an implementation of the alpha zero algorithm based on a tutorial i found implementing it with tick tack toe and conect 4
# quite a lot of the code from there is unchanged as of now and im still finishing adapting it
# the link to the tutorial is: https://www.youtube.com/watch?v=wuSQpLinRB4

import numpy as np
import matplotlib.pyplot as plt
import time as t

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


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

    def score(self,board):
        """Return the count of X and O."""
        board = np.array(board, dtype= str)
        board = np.char.replace(board, "0", " ")
        board = np.char.replace(board, "1", "X")
        board = np.char.replace(board, "-1", "O")
        x = np.sum(np.strings.count(board,"X"))
        o = np.sum(np.strings.count(board,"O"))
        return x, o

    def check_win(self, board, player_idx, move):
        if move == None:
            return False
        board = self.make_move(board, player_idx, move)
        moves = self.valid_moves(board, -player_idx)
        return np.sum(moves) == 0 and np.sum(self.valid_moves(board,player_idx)) == 0

    def get_value_and_terminated(self, board, player, move):
        if self.check_win(board, player, move):
            return 1, True
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
                    child_board = self.game.change_perspective(child_board, player = -1) # switch perspectives so that freindly peices are always 1, this is done before moves as the available moves depends on the curent player
                    child_board = self.game.make_move(child_board, 1, move)
                    child_player = -self.player

                    child = Node(self.game, self.args, child_board, child_player, self, move, prob)
                    self.childeren.append(child)
        else:
            child = Node(self.game, self.args, self.board, -self.player, self)
            self.childeren.append(child)
    
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
        root = Node(self.game, self.args, board, player, number_visits=1)

        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_board(board),
                device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"])*policy + self.args["dirichlet_epsilon"]*np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.ACTION_SIZE)
        
        valid_moves = self.game.valid_moves(board, player)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)


        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.board, node.player, node.move)
            value = -value # as the move stored in a node is the move made by the oponent so the value should be inverted

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_board(node.board),
                        device=self.model.device
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.valid_moves(node.board,node.player)
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
    
    def self_play(self):
        memory = []
        player = 1
        board = self.game.create_board()

        while True:
            if np.sum(self.game.valid_moves(board, player)) !=0:
                neutral_board = self.game.change_perspective(board, player)
                action_probs = self.mcts.search(neutral_board, 1)

                memory.append((neutral_board, action_probs, player))

                temperature_action_probs = action_probs ** (1/self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)
                move = np.random.choice(self.game.ACTION_SIZE, p=temperature_action_probs)

                board = self.game.make_move(board, player, move)

                value, is_terminal = self.game.get_value_and_terminated(board, player, move)

                if is_terminal:
                    return_memory = []
                    for hist_neutral_board, hist_action_probs, hist_player in memory:
                        hist_outcome = value if hist_player == player else -value
                        return_memory.append((
                            torch.tensor(
                                self.game.get_encoded_board(hist_neutral_board)
                            ),
                            hist_action_probs,
                            hist_outcome
                        ))
                    
                    return return_memory
            
            player = -player

    def train(self, memory):
        np.random.shuffle(memory)
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

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval()
            for self_play_iteration in range(self.args["num_self_play_iterations"]):
                memory += self.self_play()
            
            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"torch_bot_save_file/model_{iteration}.pt")
            torch.save(self.optimiser.state_dict(), f"torch_bot_save_file/optimiser_{iteration}.pt")


def test_model():
    device = torch.device(torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")

    Othello = othello()

    board = Othello.create_board()

    Othello.print_board(board)

    encoded_board = Othello.get_encoded_board(board)

    print(encoded_board)

    tensor_board = torch.tensor(encoded_board).unsqueeze(0).to(device)

    model = ResNet(Othello, 4, 64, device)
    model.load_state_dict(torch.load("torch_bot_save_file/model_2.pt"))
    model.eval()

    policy, value = model(tensor_board)
    value = value.item()
    policy = torch.softmax(policy, axis = 1).squeeze(0).detach().cpu().numpy()

    print(value, policy)

    plt.bar(range(Othello.ACTION_SIZE), policy)
    plt.show()

def main():
    game = othello()
    player = 1

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

    while True:
        game.print_board(board)

        if player == 1:
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

        value, terminated = game.get_value_and_terminated(board, player, move)

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

    model = ResNet(Othello, 4, 64, device)
    model.load_state_dict(torch.load("torch_bot_save_file/model_2.pt"))

    optimiser = torch.optim.Adam(model.parameters(), lr = 10**(-3), weight_decay=10**(-4))
    optimiser.load_state_dict(torch.load("torch_bot_save_file/optimiser_2.pt"))

    args = {
        "C": 2,
        "num_searches": 60,
        "num_iterations": 3,
        "num_self_play_iterations": 60,
        "num_epochs": 4,
        "batch_size": 12,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    alphaZero = AlphaZero(model, optimiser, Othello, args)

    start = t.time()
    alphaZero.learn()
    end = t.time()

    print(f"time taken was: {end - start}")

test_alphazero()
#test_model()
#main()