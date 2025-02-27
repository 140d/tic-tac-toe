import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# TicTacToe Class - Game Environment
class TicTacToe:
    def __init__(self):
        self.board = [[" "]*3 for _ in range(3)]
        self.current_player = "X"

    def print_board(self):
        print("\n".join(["|" + "|".join(row) + "|" for row in self.board]))
        print("-" * 9)

    def reset(self):
        """Resets the game board."""
        self.board = [[" "]*3 for _ in range(3)]
        self.current_player = "X"

    def is_valid_move(self, row, col):
        return self.board[row][col] == " "

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def available_moves(self):
        """Returns a list of available moves as (row, col) tuples."""
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == " "]

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != " " or \
               self.board[0][i] == self.board[1][i] == self.board[2][i] != " ":
                return self.board[i][i]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " " or \
           self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[1][1]

        if all(cell != " " for row in self.board for cell in row):
            return "Draw"
        return None

    def get_state(self):
        """Returns the current state of the board as a 1D array."""
        return np.array([1 if cell == "X" else -1 if cell == "O" else 0 for row in self.board for cell in row])

# QAgent Class - RL Agent
class QAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.model = self.build_model()  # Q-network model for approximating Q-values
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self):
        """Build a simple neural network model to approximate Q-values."""
        model = nn.Sequential(
            nn.Linear(9, 128),  # 9 input features (3x3 board)
            nn.ReLU(),
            nn.Linear(128, 9)  # 9 output actions (one for each cell)
        )
        return model

    def choose_action(self, game):
        """Choose an action using epsilon-greedy policy."""
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if random.random() < self.epsilon:  # Explore
            available_moves = game.available_moves()
            return random.choice(available_moves)
        else:  # Exploit
            q_values = self.model(state_tensor)
            q_values = q_values.detach().numpy().flatten()
            available_moves = game.available_moves()
            move_values = [q_values[r * 3 + c] for r, c in available_moves]
            best_move = available_moves[np.argmax(move_values)]
            return best_move

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using the Q-learning update rule."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        target = reward + self.gamma * torch.max(next_q_values) * (1 - int(done))
        loss = nn.MSELoss()(q_values[0, action], target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training the Agent
def train_agent(agent, num_episodes=1000):
    for episode in range(num_episodes):
        game = TicTacToe()
        done = False
        state = game.get_state()

        while not done:
            action = agent.choose_action(game)
            row, col = action
            game.make_move(row, col)

            next_state = game.get_state()
            winner = game.check_winner()

            if winner == "X":
                reward = 1
                done = True
            elif winner == "O":
                reward = -1
                done = True
            elif winner == "Draw":
                reward = 0
                done = True
            else:
                reward = 0
                done = False

            agent.learn(state, row * 3 + col, reward, next_state, done)
            state = next_state

# Saving the trained model
def save_model(agent, filename="tic_tac_toe_model.pth"):
    torch.save(agent.model.state_dict(), filename)

# Main Script to Train and Save Model
if __name__ == "__main__":
    agent = QAgent(epsilon=0.1, alpha=0.5, gamma=0.9)

    # Train the agent to play against itself
    print("Training the agent. This may take some time...")
    train_agent(agent, num_episodes=5000)

    # Save the trained model
    print("Saving the trained model...")
    save_model(agent)
    print("Model saved successfully!")

# TicTacToe Class - Game Environment
class TicTacToe:
    def __init__(self):
        self.board = [[" "]*3 for _ in range(3)]
        self.current_player = "X"

    def print_board(self):
        print("\n".join(["|" + "|".join(row) + "|" for row in self.board]))
        print("-" * 9)

    def reset(self):
        """Resets the game board."""
        self.board = [[" "]*3 for _ in range(3)]
        self.current_player = "X"

    def is_valid_move(self, row, col):
        return self.board[row][col] == " "

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def available_moves(self):
        """Returns a list of available moves as (row, col) tuples."""
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == " "]

    def check_winner(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != " " or \
               self.board[0][i] == self.board[1][i] == self.board[2][i] != " ":
                return self.board[i][i]

        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " " or \
           self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[1][1]

        if all(cell != " " for row in self.board for cell in row):
            return "Draw"
        return None

    def get_state(self):
        """Returns the current state of the board as a 1D array."""
        return np.array([1 if cell == "X" else -1 if cell == "O" else 0 for row in self.board for cell in row])

# QAgent Class - RL Agent
class QAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.model = self.build_model()  # Q-network model for approximating Q-values
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def build_model(self):
        """Build a simple neural network model to approximate Q-values."""
        model = nn.Sequential(
            nn.Linear(9, 128),  # 9 input features (3x3 board)
            nn.ReLU(),
            nn.Linear(128, 9)  # 9 output actions (one for each cell)
        )
        return model

    def load_model(self, filename="tic_tac_toe_model.pth"):
        """Load a trained model from file."""
        self.model.load_state_dict(torch.load(filename))

    def choose_action(self, game):
        """Choose an action using epsilon-greedy policy."""
        state = game.get_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if random.random() < self.epsilon:  # Explore
            available_moves = game.available_moves()
            return random.choice(available_moves)
        else:  # Exploit
            q_values = self.model(state_tensor)
            q_values = q_values.detach().numpy().flatten()
            available_moves = game.available_moves()
            move_values = [q_values[r * 3 + c] for r, c in available_moves]
            best_move = available_moves[np.argmax(move_values)]
            return best_move

# Function to play with the trained agent
def play_with_agent(agent):
    game = TicTacToe()
    human_player = input("Do you want to play as 'X' or 'O'? ").upper()

    # Validate human player input
    if human_player not in ["X", "O"]:
        print("Invalid input, defaulting to 'X'.")
        human_player = "X"

    print("\nYou can now play against the trained agent!")

    # Main game loop
    while True:
        game.print_board()
        winner = game.check_winner()

        if winner:
            if winner == "Draw":
                print("It's a draw!")
            else:
                print(f"Player {winner} wins!")
            break

        # Player's turn
        if game.current_player == human_player:
            print(f"Your turn (Player {human_player})!")
            try:
                row, col = map(int, input("Enter row and column (0 1 2): ").split())
                if not game.make_move(row, col):
                    print("Invalid move. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter row and column as two integers (0 1 2).")
                continue
        else:
            # Agent's turn
            print(f"Agent's turn (Player {game.current_player})!")
            row, col = agent.choose_action(game)
            game.make_move(row, col)

# Main Script to Load and Play
if __name__ == "__main__":
    agent = QAgent(epsilon=0.1, alpha=0.5, gamma=0.9)

    # Load the trained model
    print("Loading the trained model...")
    agent.load_model("tic_tac_toe_model.pth")

    # Play with the trained agent
    play_with_agent(agent)

