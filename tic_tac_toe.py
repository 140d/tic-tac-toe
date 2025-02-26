class TicTacToe:
    def __init__(self):
        self.board = [[" "]*3 for _ in range(3)]
        self.current_player = "X"

    def print_board(self):
        print("\n".join(["|" + "|".join(row) + "|" for row in self.board]))
        print("-" * 9)

    def is_valid_move(self, row, col):
        return self.board[row][col] == " "

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

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

    def play(self):
        print("Welcome to Tic Tac Toe!")
        while True:
            self.print_board()
            print(f"Player {self.current_player}'s turn.")
            try:
                row, col = map(int, input("Enter row and column (0 1 2): ").split())
                if row not in range(3) or col not in range(3) or not self.make_move(row, col):
                    print("Invalid move, try again.")
                    continue
            except (ValueError, IndexError):
                print("Invalid input! Please enter valid row and column.")
                continue

            winner = self.check_winner()
            if winner:
                self.print_board()
                print("It's a draw!" if winner == "Draw" else f"Player {winner} wins!")
                break


if __name__ == "__main__":
    TicTacToe().play()
