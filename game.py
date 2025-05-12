ROWS = 6
COLS = 7
EMPTY = 0
PLAYER = 1
AI = -1

def is_winning_move(board, piece):
    """Retourne True si `piece` a aligné 4 pions."""
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == piece for i in range(4)):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True

    # Diagonale \
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True

    # Diagonale /
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r+i][c-i] == piece for i in range(4)):
                return True

    return False
