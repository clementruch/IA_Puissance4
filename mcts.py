import math
import random
from typing import List, Optional

from evaluation import evaluate
from minimax import get_valid_moves, make_move, is_terminal

Board = List[List[Optional[str]]]

class Node:
    def __init__(self, board: Board, player: Optional[str], parent: Optional['Node']=None, move: Optional[int]=None):
        self.board = board  # plateau du jeu
        self.player = player  # joueur ayant joué pour arriver ici ('X' ou 'O')
        self.parent = parent
        self.move = move
        self.children: List['Node'] = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = get_valid_moves(board).copy()

    def uct_select_child(self, C: float) -> 'Node':
        # Upper Confidence Bound applied to Trees (UCT)
        return max(
            self.children,
            key=lambda c: (c.wins / c.visits) + C * math.sqrt(2 * math.log(self.visits) / c.visits)
        )

    def add_child(self, move: int, board: Board, player: str) -> 'Node':
        child = Node(board, player, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result_winner: Optional[str]) -> None:
        self.visits += 1
        if result_winner is not None and self.player == result_winner:
            self.wins += 1


def simulate_random_playout(board: Board, current_player: str) -> Optional[str]:
    # Simule une partie aléatoire à partir du plateau courant
    b = [row.copy() for row in board]
    player = current_player
    while not is_terminal(b):
        move = random.choice(get_valid_moves(b))
        b = make_move(b, move, player)
        player = 'O' if player == 'X' else 'X'
    score = evaluate(b)
    if score >= 1000:
        return 'X'
    if score <= -1000:
        return 'O'
    return None


def mcts(root_board: Board, root_player: str, iter_max: int = 1000, C: float = 1.4) -> int:
    # Monte Carlo Tree Search principal
    # root_player: joueur à qui c'est le tour ('X' ou 'O')
    # iter_max: nombre d'itérations de simulation
    # C: coefficient d'exploration
    rootnode = Node(root_board, player=('O' if root_player == 'X' else 'X'))

    for _ in range(iter_max):
        node = rootnode
        board = [row.copy() for row in root_board]

        # Sélection
        while not node.untried_moves and node.children:
            node = node.uct_select_child(C)
            board = make_move(board, node.move, node.player)

        # Expansion
        if node.untried_moves:
            m = random.choice(node.untried_moves)
            next_player = 'O' if node.player == 'X' else 'X'
            board = make_move(board, m, next_player)
            node = node.add_child(m, board, next_player)

        # Simulation
        sim_player = 'O' if node.player == 'X' else 'X'
        winner = simulate_random_playout(board, sim_player)

        # Rétropropagation
        while node:
            node.update(winner)
            node = node.parent

    # Retourne le coup du fils le plus visité
    best_child = max(rootnode.children, key=lambda c: c.visits)
    return best_child.move
