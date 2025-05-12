import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 1
AI = -1

def evaluate_window(window, piece):
    """Attribue un score à une fenêtre de 4 cases."""
    score = 0
    opp_piece = PLAYER if piece == AI else AI

    count_piece = window.count(piece)
    count_empty = window.count(EMPTY)
    count_opp   = window.count(opp_piece)

    if count_piece == 4:
        score += 1000
    elif count_piece == 3 and count_empty == 1:
        score += 50
    elif count_piece == 2 and count_empty == 2:
        score += 5
    elif count_piece == 1 and count_empty == 3:
        score += 1

    if count_opp == 3 and count_empty == 1:
        score -= 80  # priorité à bloquer l'adversaire

    return score

def evaluate_board(board, piece):
    """Retourne un score global pour `piece` – plus il est élevé, mieux c'est."""
    score = 0
    # Convertir en liste de listes (si besoin)
    grid = board

    # Score centre
    center_array = [row[COLS//2] for row in grid]
    center_count = center_array.count(piece)
    score += center_count * 6

    # Horizontal
    for r in range(ROWS):
        row_array = [grid[r][c] for c in range(COLS)]
        for c in range(COLS - 3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    # Vertical
    for c in range(COLS):
        col_array = [grid[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    # Diagonale positive (/)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [grid[r+i][c+3-i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Diagonale négative (\)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [grid[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score
