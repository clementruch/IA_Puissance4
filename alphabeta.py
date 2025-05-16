from typing import List, Optional, Tuple
from evaluation import evaluate

Board = List[List[Optional[str]]]

def alphabeta(board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[int, Optional[int]]:
    if depth == 0 or is_terminal(board):
        return evaluate(board), None

    player = 'X' if maximizing_player else 'O'
    best_move: Optional[int] = None

    if maximizing_player:
        max_eval = float('-inf')
        for move in get_valid_moves(board):
            child = make_move(board, move, player)
            eval_score, _ = alphabeta(child, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in get_valid_moves(board):
            child = make_move(board, move, player)
            eval_score, _ = alphabeta(child, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # coupure alpha
        return min_eval, best_move

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