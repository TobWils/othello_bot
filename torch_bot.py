import numpy as np
import matplotlib.pyplot as plt
import time as t

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, game: othello, num_res_blocks, num_hidden):
        super().__init__()

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
    
    def forward(self, x):
        x = self.start_block(x)
        for res_block in self.backbone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def test_model():
    Othello = othello()

    board = Othello.create_board()

    Othello.print_board(board)

    encoded_board = Othello.get_encoded_board(board)

    print(encoded_board)

    tensor_board = torch.tensor(encoded_board).unsqueeze(0)

    model = ResNet(Othello, 4, 64)

    policy, value = model(tensor_board)
    value = value.item()
    policy = torch.softmax(policy, axis = 1).squeeze(0).detach().cpu().numpy()

    print(value, policy)

    plt.bar(range(Othello.ACTION_SIZE), policy)
    plt.show()

class Node():
    def __init__(self, game: othello, args, board, player, parent=None, move=None):
        self.game = game
        self.args = args
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move

        self.childeren = []
        self.expandable_moves = game.valid_moves(board, player)

        self.number_visits = 0
        self.total_val = 0
    
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.childeren) > 0
    
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
        q_value = 1 - (child.total_val/child.number_visits + 1)/2 # want to be a number in range 0 to 1 and to be low so that the win prob is minimised for oponent
        return q_value + self.args["C"]*np.sqrt(np.log(self.number_visits)/child.number_visits)
    
    def expand(self):
        if np.sum(self.expandable_moves) != 0:
            move = np.random.choice(np.where(self.expandable_moves == 1)[0])
            self.expandable_moves[move] = 0

            child_board = self.board.copy()
            child_board = self.game.change_perspective(child_board, player = -1) # switch perspectives so that freindly peices are always 1, this is done before moves as the available moves depends on the curent player
            child_board = self.game.make_move(child_board, 1, move)
            child_player = -self.player

            child = Node(self.game, self.args, child_board, child_player, self, move)
            self.childeren.append(child)

            return child
        else:
            child = Node(self.game, self.args, self.board, -self.player, self)
            self.childeren.append(child)

            return child
    
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.board, self.player, self.move)
        value = -value

        if is_terminal:
            return value
        
        rollout_board = self.board.copy()
        rollout_board = self.game.change_perspective(rollout_board, player = -1)
        rollout_player = 1 # in this implementation we only ever show the bot a board that has 1's for its peices and -1's for oponent peices hence we set the rolout player to be 1
        while True:
            valid_moves = self.game.valid_moves(rollout_board, rollout_player)
            if np.sum(valid_moves) != 0:
                move = np.random.choice(np.where(valid_moves == 1)[0])
                rollout_board = self.game.make_move(rollout_board, rollout_player, move)
                value, is_terminal = self.game.get_value_and_terminated(rollout_board, self.player, move)
                if is_terminal:
                    if rollout_player == -1:
                        value = -value
                    return value
            
            rollout_player = -rollout_player
    
    def backpropigate(self, value):
        self.total_val += value
        self.number_visits += 1

        value = -value
        if self.parent is not None:
            self.parent.backpropigate(value)

class MCTS():
    def __init__(self, game: othello, args):
        self.game = game
        self.args = args

    def search(self, board, player):
        root = Node(self.game, self.args, board, player)

        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.board, node.player, node.move)
            value = -value # as the move stored in a node is the move made by the oponent so the value should be inverted

            if not is_terminal:
                node = node.expand()
                value = node.simulate()
            
            node.backpropigate(value)

        action_probs = np.zeros(self.game.ACTION_SIZE)
        for child in root.childeren:
            action_probs[child.move] = child.number_visits
        action_probs /= np.sum(action_probs)

        return action_probs


def main():
    game = othello()
    player = 1

    args = {
        "C": 1.41,
        "num_searches": 1000
    }

    mcts = MCTS(game, args)

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
            mcts_probs = mcts.search(neutral_board, -player)
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

test_model()