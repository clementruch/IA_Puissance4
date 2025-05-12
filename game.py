from typing import List, Optional
from minimax import minimax
from evaluation import evaluate

Board = List[List[Optional[str]]]


def create_board() -> Board:
    return [[None] * 7 for _ in range(6)]

# Affiche le plateau dans la console
def print_board(board: Board) -> None:
    print("\nPlateau :")
    for row in board:
        print("| "+" | ".join(cell if cell is not None else ' ' for cell in row)+" |")
    print("  " + "   ".join(str(i) for i in range(7)))
    print()


def get_valid_moves(board: Board) -> List[int]:
    return [c for c in range(7) if board[0][c] is None]


def make_move(board: Board, col: int, player: str) -> bool:
    for r in range(5, -1, -1):
        if board[r][col] is None:
            board[r][col] = player
            return True
    return False


def is_full(board: Board) -> bool:
    return all(cell is not None for cell in board[0])


def check_win(board: Board, player: str) -> bool:
    rows, cols = 6, 7
    # horizontales
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r][c+i] == player for i in range(4)):
                return True
    # verticales
    for c in range(cols):
        for r in range(rows - 3):
            if all(board[r+i][c] == player for i in range(4)):
                return True
    # diag ↘
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    # diag ↗
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
    return False


def choose_move_ai(board: Board, player: str, depth: int) -> int:
    score, move = minimax(board, depth, maximizing_player=(player=='X'))
    if move is None:
        raise ValueError("AI n'a pas trouvé de coup valide")
    return move


def play_game():
    board = create_board()
    current = 'X'

    # Configuration : humain ou IA
    modes = {}
    for p in ['X','O']:
        mode = ''
        while mode not in ['H','A']:
            mode = input(f"Joueur {p} : (H)umain ou (A)I ? ").strip().upper()
        modes[p] = mode
    depth = None
    if 'A' in modes.values():
        d = input("Profondeur Minimax pour l'IA (ex 4): ").strip()
        depth = int(d)

    while True:
        print_board(board)
        if modes[current] == 'H':
            valid = get_valid_moves(board)
            move = -1
            while move not in valid:
                try:
                    move = int(input(f"Colonne à jouer {current} (0-6): "))
                except ValueError:
                    continue
        else:
            print(f"IA {current} réfléchit... (profondeur {depth})")
            move = choose_move_ai(board, current, depth)
            print(f"IA {current} joue en colonne {move}")

        make_move(board, move, current)

        # Vérification de fin
        if check_win(board, current):
            print_board(board)
            print(f"Joueur {current} a gagné !")
            break
        if is_full(board):
            print_board(board)
            print("Match nul !")
            break

        current = 'O' if current == 'X' else 'X'


if __name__ == '__main__':
    play_game()