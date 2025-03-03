import random

class TicTacToe:
    def __init__(self):
        self.board = [[" "] * 3 for _ in range(3)] # Initialize the game board as a 3x3 grid with empty spaces
        self.current_player = "X" # Start with player "X"

    def print_board(self):
        print("  0 1 2") # Print the column numbers at the top
        # Loop through each row and print the row number followed by the row content
        for i, row in enumerate(self.board):
            print(f"{i} " + "|".join(row))

    def is_valid_move(self, row, col):
        return self.board[row][col] == " " # Check if the specified cell is empty

    def make_move(self, row, col):
        # If the move is valid, place the current player's symbol on the board
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            # Switch the current player to the other player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False # If the move is invalid, return False

    def check_winner(self):
        # Check rows and columns for a winner
        for i in range(3):
            if (self.board[i][0] == self.board[i][1] == self.board[i][2] != " " or
                self.board[0][i] == self.board[1][i] == self.board[2][i] != " "):
                # Return the winning player's symbol
                return self.board[i][i]

        # Check diagonals for a winner
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " " or \
           self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[1][1]

        # Check if the board is full (draw)
        if all(cell != " " for row in self.board for cell in row): return "Draw"

        return None # If no winner and no draw, return None

    def get_winning_move(self, player):
        # Check if the specified player can win in the next move
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    self.board[row][col] = player # Simulate the move
                    # Check if this move results in a win
                    if self.check_winner() == player:
                        # Undo the move and return the winning position
                        self.board[row][col] = " "
                        return (row, col)
                    # Undo the move if it doesn't result in a win
                    self.board[row][col] = " "
        return None # If no winning move is found, return None

    def count_available_lines(self, row, col):
        count = 0 # Count how many lines (rows, columns, diagonals) are still open for the given cell

        # Check the row
        if self.board[row][0] == self.board[row][1] == self.board[row][2] == " ":
            count += 1

        # Check the column
        if self.board[0][col] == self.board[1][col] == self.board[2][col] == " ":
            count += 1

        # Check the main diagonal (if the cell is on it)
        if row == col and self.board[0][0] == self.board[1][1] == self.board[2][2] == " ":
            count += 1

        # Check the anti-diagonal (if the cell is on it)
        if row + col == 2 and self.board[0][2] == self.board[1][1] == self.board[2][0] == " ":
            count += 1

        return count

    def get_bot_move(self):
        # First, check if the bot can win in the next move
        winning_move = self.get_winning_move("O")
        if winning_move: return winning_move

        # Then, check if the opponent can win in the next move and block them
        opponent_winning_move = self.get_winning_move("X")
        if opponent_winning_move: return opponent_winning_move

        # If no immediate win or block, find all available moves
        available_moves = [(r, c) for r in range(3) for c in range(3) if self.is_valid_move(r, c)]

        # Score each available move based on how many lines it opens
        move_scores = []
        for move in available_moves:
            row, col = move
            score = self.count_available_lines(row, col)
            move_scores.append((move, score))

        # Find the move(s) with the highest score
        max_score = max(move_scores, key=lambda x: x[1])[1]
        best_moves = [move for move, score in move_scores if score == max_score]

        # Randomly choose one of the best moves
        return random.choice(best_moves) if best_moves else None

    def play(self):
        print("Welcome to Tic Tac Toe!")
        while True:
            # Print the current state of the board
            self.print_board()
            print(f"Player {self.current_player}'s turn.")

            if self.current_player == "X":
                # Human player's turn
                try:
                    # Get the row and column from the player
                    row, col = map(int, input("Enter row and column (0 1 2): ").split())
                    # Make the move if it's valid, otherwise prompt again
                    if not self.make_move(row, col):
                        print("Invalid move, try again.")
                        continue
                except (ValueError, IndexError):
                    # Handle invalid input
                    print("Invalid input! Please enter valid row and column.")
                    continue
            else:
                print("Bot's turn...")
                row, col = self.get_bot_move()
                self.make_move(row, col)
                print(f"Bot chooses position ({row}, {col})")

            # Check if the game has ended (win or draw)
            winner = self.check_winner()
            if winner:
                self.print_board()
                print(f"{'Draw' if winner == 'Draw' else f'Player {winner} wins!'}")
                break

if __name__ == "__main__":
    TicTacToe().play()
