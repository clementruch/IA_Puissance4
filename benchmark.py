import argparse
import time
import statistics
from typing import List, Optional
import minimax
import alphabeta
from evaluation import evaluate as good_evaluate
from evaluationBad import simple_evaluate as bad_evaluate

# ----------- Logique du jeu -----------
Board = List[List[Optional[str]]]
ROWS, COLS = 6, 7

def create_board() -> Board:
    return [[None]*COLS for _ in range(ROWS)]

def get_valid_moves(board: Board) -> List[int]:
    return [c for c in range(COLS) if board[0][c] is None]

def make_move(board: Board, col: int, player: str) -> None:
    for r in range(ROWS-1, -1, -1):
        if board[r][col] is None:
            board[r][col] = player
            return

def is_full(board: Board) -> bool:
    return all(cell is not None for cell in board[0])

def check_win(board: Board, player: str) -> bool:
    # horizontales
    for r in range(ROWS):
        for c in range(COLS-3):
            if all(board[r][c+i]==player for i in range(4)): return True
    # verticales
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c]==player for i in range(4)): return True
    # diagonales bas-gauche à haut-droite
    for r in range(ROWS-3):
        for c in range(COLS-3):
            if all(board[r+i][c+i]==player for i in range(4)): return True
    # diagonales haut-gauche à bas-droite
    for r in range(3, ROWS):
        for c in range(COLS-3):
            if all(board[r-i][c+i]==player for i in range(4)): return True
    return False

# ----------- Wrappers d'instrumentation -----------
class MinimaxWrapper:
    def __init__(self, func):
        self.func = func
        self.nodes = 0
    def __call__(self, board, depth, maximizing_player):
        self.nodes += 1
        return self.func(board, depth, maximizing_player)

class AlphaBetaWrapper:
    def __init__(self, func):
        self.func = func
        self.nodes = 0
    def __call__(self, board, depth, alpha, beta, maximizing_player):
        self.nodes += 1
        return self.func(board, depth, alpha, beta, maximizing_player)

# ----------- Simulation d'une partie -----------
def simulate_game(depth: int,
                  eval_x, eval_o,
                  algo_x: str, algo_o: str) -> dict:
    # Choix de l'évaluation
    minimax.evaluate = eval_x
    alphabeta.evaluate = eval_x

    # Wrappers
    mm_wrapper = MinimaxWrapper(minimax.minimax)
    ab_wrapper = AlphaBetaWrapper(alphabeta.alphabeta)

    board = create_board()
    current = 'X'
    metrics = {'nodes_per_move': [], 'time_per_move': []}

    while True:
        # Sélection de l'algorithme
        algo = mm_wrapper if (current=='X' and algo_x=='minimax') or (current=='O' and algo_o=='minimax') else ab_wrapper
        algo.nodes = 0
        start = time.perf_counter()
        if isinstance(algo, MinimaxWrapper):
            _, move = algo(board, depth, maximizing_player=(current=='X'))
        else:
            _, move = algo(board, depth, float('-inf'), float('inf'), maximizing_player=(current=='X'))
        end = time.perf_counter()
        metrics['time_per_move'].append(end-start)
        metrics['nodes_per_move'].append(algo.nodes)
        make_move(board, move, current)
        if check_win(board, current):
            metrics['winner'] = current
            break
        if is_full(board):
            metrics['winner'] = None
            break
        current = 'O' if current=='X' else 'X'

    metrics['total_time'] = sum(metrics['time_per_move'])
    metrics['total_nodes'] = sum(metrics['nodes_per_move'])
    metrics['moves'] = len(metrics['time_per_move'])
    return metrics

# ----------- Fonction principale & rapport -----------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Minimax vs Alpha-Beta")
    parser.add_argument('--algo_x', choices=['minimax','alphabeta'], default='minimax')
    parser.add_argument('--algo_o', choices=['minimax','alphabeta'], default='alphabeta')
    parser.add_argument('--eval', choices=['good','bad'], default='good')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--games', type=int, default=10)
    args = parser.parse_args()

    # Affichage des paramètres en français
    print("\n=== Paramètres du test ===")
    print(f"Algorithme X       : {args.algo_x}")
    print(f"Algorithme O       : {args.algo_o}")
    print(f"Heuristique        : {'classique' if args.eval=='good' else 'simple'}")
    print(f"Profondeur IA      : {args.depth}")
    print(f"Nombre de parties  : {args.games}")
    print("==========================\n")

    # Choix de la fonction d'évaluation
    eval_func = good_evaluate if args.eval=='good' else bad_evaluate

    results = []
    for i in range(args.games):
        print(f"Partie {i+1}/{args.games} en cours...")
        metrics = simulate_game(args.depth, eval_func, eval_func,
                                 args.algo_x, args.algo_o)
        results.append(metrics)

    # Agrégation des résultats
    wins = {'X':0,'O':0,'draw':0}
    times = []
    nodes = []
    for m in results:
        if m['winner']=='X': wins['X']+=1
        elif m['winner']=='O': wins['O']+=1
        else: wins['draw']+=1
        times.append(m['total_time'])
        nodes.append(m['total_nodes'])

    print("\n=== Résumé ===")
    print(f"Victoires X: {wins['X']} | Victoires O: {wins['O']} | Matchs nuls: {wins['draw']}")
    print(f"Temps moyen par partie : {statistics.mean(times):.3f}s (écart-type {statistics.stdev(times):.3f})")
    print(f"Nombre moyen de nœuds  : {statistics.mean(nodes):.0f} (écart-type {statistics.stdev(nodes):.0f})")

if __name__ == '__main__':
    main()
