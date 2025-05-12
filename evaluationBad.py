from typing import List, Tuple, Optional

Board = List[List[Optional[str]]]

def simple_lines(board: Board) -> List[List[Tuple[int, int]]]:
    rows, cols = 6, 7
    lines: List[List[Tuple[int,int]]] = []
    # horizontales
    for r in range(rows):
        for c in range(cols - 3):
            lines.append([(r, c + i) for i in range(4)])
    # verticales
    for c in range(cols):
        for r in range(rows - 3):
            lines.append([(r + i, c) for i in range(4)])
    return lines

def simple_score_line(cells: List[Optional[str]]) -> int:
    nX = cells.count('X')
    nO = cells.count('O')
    nv = cells.count(None)

    if nX == 4:
        return 1000
    if nO == 4:
        return -1000
    # deux pions + deux vides
    if nX == 2 and nv == 2:
        return 5
    if nO == 2 and nv == 2:
        return -5
    return 0

def simple_evaluate(board: Board) -> int:
    total = 0
    for line in simple_lines(board):
        cells = [board[r][c] for r,c in line]
        total += simple_score_line(cells)
    return total

# Exemple de test
if __name__ == "__main__":
    vide = [[None]*7 for _ in range(6)]
    test = [row.copy() for row in vide]
    test[5][2], test[5][3] = 'X', 'X'
    print("Score simple =", simple_evaluate(test))  # =5

    test[5][4] = 'X'
    print("Score simple =", simple_evaluate(test))  # toujours =5, car on ignore les 3 en ligne
