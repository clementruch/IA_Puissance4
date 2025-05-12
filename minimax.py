from typing import List, Optional, Tuple
from evaluation import evaluate

Board = List[List[Optional[str]]]


def get_valid_moves(board: Board) -> List[int]:
    return [c for c in range(7) if board[0][c] is None]


def make_move(board: Board, col: int, player: str) -> Board:
    new_board = [row.copy() for row in board]
    for r in range(5, -1, -1):
        if new_board[r][col] is None:
            new_board[r][col] = player
            break
    return new_board


def is_terminal(board: Board) -> bool:
    # Victoire détectée par l'évaluation (±1000)
    if abs(evaluate(board)) >= 1000:
        return True
    # Plus aucun coup possible => match nul
    return len(get_valid_moves(board)) == 0


def minimax(board: Board, depth: int, maximizing_player: bool) -> Tuple[int, Optional[int]]:
    if depth == 0 or is_terminal(board):
        return evaluate(board), None

    player = 'X' if maximizing_player else 'O'
    best_score = float('-inf') if maximizing_player else float('inf')
    best_move: Optional[int] = None

    for move in get_valid_moves(board):
        child = make_move(board, move, player)
        score, _ = minimax(child, depth - 1, not maximizing_player)
        if maximizing_player:
            if score > best_score:
                best_score, best_move = score, move
        else:
            if score < best_score:
                best_score, best_move = score, move

    return best_score, best_move


# if __name__ == "__main__":
#     empty_board: Board = [[None] * 7 for _ in range(6)]
#     score, move = minimax(empty_board, depth=4, maximizing_player=True)
#     print(f"Minimax depth=4 → score={{score}}, move={{move}}")
