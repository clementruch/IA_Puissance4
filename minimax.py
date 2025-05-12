import math
from game import get_valid_locations, drop_piece, get_next_open_row, is_winning_move, copy_board
from evaluation import evaluate_board

PLAYER = 1
AI = -1
EMPTY = 0

def minimax(board, depth, maximizingPlayer):
    valid_locations = get_valid_locations(board)

    is_terminal = is_winning_move(board, PLAYER) or is_winning_move(board, AI) or len(valid_locations) == 0
    if depth == 0 or is_terminal:
        if is_terminal:
            if is_winning_move(board, AI):
                return (None, 100000)
            elif is_winning_move(board, PLAYER):
                return (None, -100000)
            else:  # Match nul
                return (None, 0)
        else:  # Feuille
            return (None, evaluate_board(board, AI))

    if maximizingPlayer:
        value = -math.inf
        best_col = None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = copy_board(board)
            drop_piece(temp_board, row, col, AI)
            _, new_score = minimax(temp_board, depth - 1, False)
            if new_score > value:
                value = new_score
                best_col = col
        return best_col, value

    else:  # minimizingPlayer
        value = math.inf
        best_col = None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = copy_board(board)
            drop_piece(temp_board, row, col, PLAYER)
            _, new_score = minimax(temp_board, depth - 1, True)
            if new_score < value:
                value = new_score
                best_col = col
        return best_col, value
