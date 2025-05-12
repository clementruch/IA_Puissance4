from typing import List, Tuple, Optional

# On représente le plateau par une liste de listes 6×7, avec 'X', 'O' ou None.
Board = List[List[Optional[str]]]

def all_lines(board: Board) -> List[List[Tuple[int, int]]]:
    rows, cols = 6, 7
    lines: List[List[Tuple[int, int]]] = []
    # horizontales
    for r in range(rows):
        for c in range(cols - 3):
            lines.append([(r, c + i) for i in range(4)])
    # verticales
    for c in range(cols):
        for r in range(rows - 3):
            lines.append([(r + i, c) for i in range(4)])
    # diagonales montantes (↗)
    for r in range(3, rows):
        for c in range(cols - 3):
            lines.append([(r - i, c + i) for i in range(4)])
    # diagonales descendantes (↘)
    for r in range(rows - 3):
        for c in range(cols - 3):
            lines.append([(r + i, c + i) for i in range(4)])
    return lines

def score_line(cells: List[Optional[str]]) -> int:
    nX = cells.count('X')
    nO = cells.count('O')
    nv = cells.count(None)
    # Victoire
    if nX == 4:
        return 1000
    if nO == 4:
        return -1000
    # Trois pions + un espace
    if nX == 3 and nv == 1:
        return 50
    if nO == 3 and nv == 1:
        return -50
    # Deux pions + deux espaces
    if nX == 2 and nv == 2:
        return 5
    if nO == 2 and nv == 2:
        return -5
    # Un pion + trois espaces
    if nX == 1 and nv == 3:
        return 1
    if nO == 1 and nv == 3:
        return -1
    return 0

def evaluate(board: Board) -> int:
    total = 0
    for line in all_lines(board):
        cells = [board[r][c] for r, c in line]
        total += score_line(cells)
    return total

# # Exemple rapide
# if __name__ == "__main__":
#     vide = [[None]*7 for _ in range(6)]
#     test = [row.copy() for row in vide]
#     test[5][3] = 'X'
#     test[4][3] = 'X'
#     test[3][3] = 'O'
#     test[2][3] = 'X'
#     print("Score =", evaluate(test))
