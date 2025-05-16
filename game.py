from typing import List, Optional
import minimax
import alphabeta
from evaluation import evaluate as good_evaluate
from evaluationBad import simple_evaluate as bad_evaluate

Board = List[List[Optional[str]]]


def create_board() -> Board:
    return [[None] * 7 for _ in range(6)]


def print_board(board: Board) -> None:
    print("\nPlateau :")
    for row in board:
        print("| " + " | ".join(cell if cell is not None else ' ' for cell in row) + " |")
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
    """Détecte si le joueur a 4 en ligne."""
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
    # diag bas
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True
    # diag haut
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True
    return False



def choose_move_ai(board: Board, player: str, depth: int, algo: str) -> int:
    if algo == 'alphabeta':
        score, move = alphabeta.alphabeta(board, depth, float('-inf'), float('inf'), maximizing_player=(player == 'X'))
    else:
        score, move = minimax.minimax(board, depth, maximizing_player=(player == 'X'))
    if move is None:
        raise ValueError("AI n'a pas trouvé de coup valide")
    return move


def play_game():
    modes = {}
    algos = {}
    depths = {}

    # Joueur X
    mode = ''
    while mode not in ['H', 'A']:
        mode = input("Joueur X : (H)umain ou (A)I ? ").strip().upper()
    modes['X'] = mode
    if mode == 'A':
        algo = ''
        while algo not in ['1', '2']:
            algo = input("Algorithme IA pour X ? 1) Minimax 2) Alpha-Beta : ").strip()
        algos['X'] = 'minimax' if algo == '1' else 'alphabeta'
        depth = ''
        while not (depth.isdigit() and 1 <= int(depth) <= 8):
            depth = input("Profondeur IA pour X (1-8) : ").strip()
        depths['X'] = int(depth)
    else:
        algos['X'] = None
        depths['X'] = None

    # Joueur O
    mode = ''
    while mode not in ['H', 'A']:
        mode = input("Joueur O : (H)umain ou (A)I ? ").strip().upper()
    modes['O'] = mode
    if mode == 'A':
        algo = ''
        while algo not in ['1', '2']:
            algo = input("Algorithme IA pour O ? 1) Minimax 2) Alpha-Beta : ").strip()
        algos['O'] = 'minimax' if algo == '1' else 'alphabeta'
        depth = ''
        while not (depth.isdigit() and 1 <= int(depth) <= 8):
            depth = input("Profondeur IA pour O (1-8) : ").strip()
        depths['O'] = int(depth)
    else:
        algos['O'] = None
        depths['O'] = None

    board = create_board()
    current = 'X'

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
            print(f"IA {current} réfléchit... (algo: {algos[current]}, profondeur {depths[current]})")
            move = choose_move_ai(board, current, depths[current], algos[current])
            print(f"IA {current} joue en colonne {move}")

        make_move(board, move, current)

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
