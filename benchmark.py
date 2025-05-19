import os
import csv
import time
from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import pandas as pd
import minimax
import alphabeta
from evaluation import evaluate as good_evaluate
from evaluationBad import simple_evaluate as bad_evaluate

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
    for r in range(ROWS):
        for c in range(COLS-3):
            if all(board[r][c+i]==player for i in range(4)): return True
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c]==player for i in range(4)): return True
    for r in range(ROWS-3):
        for c in range(COLS-3):
            if all(board[r+i][c+i]==player for i in range(4)): return True
    for r in range(3, ROWS):
        for c in range(COLS-3):
            if all(board[r-i][c+i]==player for i in range(4)): return True
    return False

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

def simulate_game(depth_x, depth_o, algo_x, algo_o, eval_func) -> Dict:
    minimax.evaluate = eval_func
    alphabeta.evaluate = eval_func

    mm = MinimaxWrapper(minimax.minimax)
    ab = AlphaBetaWrapper(alphabeta.alphabeta)

    board = create_board()
    current = 'X'
    metrics = {
        'nodes_per_move': [],
        'time_per_move': [],
        'winner': None,
        'moves': 0
    }

    while True:
        if current == 'X':
            algo = mm if algo_x == 'minimax' else ab
            depth = depth_x
        else:
            algo = mm if algo_o == 'minimax' else ab
            depth = depth_o

        algo.nodes = 0
        start = time.perf_counter()
        if isinstance(algo, MinimaxWrapper):
            _, move = algo(board, depth, maximizing_player=(current == 'X'))
        else:
            _, move = algo(board, depth, float('-inf'), float('inf'), maximizing_player=(current == 'X'))
        end = time.perf_counter()

        metrics['time_per_move'].append(end - start)
        metrics['nodes_per_move'].append(algo.nodes)
        make_move(board, move, current)
        metrics['moves'] += 1

        if check_win(board, current):
            metrics['winner'] = current
            break
        if is_full(board):
            metrics['winner'] = None
            break
        current = 'O' if current == 'X' else 'X'

    metrics['total_time'] = sum(metrics['time_per_move'])
    metrics['total_nodes'] = sum(metrics['nodes_per_move'])
    return metrics

def main():
    eval_funcs = [
        ("Bonne", good_evaluate),
        ("Mauvaise", bad_evaluate)
    ]
    algos = ["minimax", "alphabeta"]
    results = []
    depths = [2, 3, 4]  # élargis si ta machine est rapide

    for eval_name, eval_func in eval_funcs:
        for algo_x in algos:
            for algo_o in algos:
                for dx in depths:
                    for dy in depths:
                        m = simulate_game(dx, dy, algo_x, algo_o, eval_func)
                        m.update({
                            'eval_func': eval_name,
                            'algo_x': algo_x,
                            'algo_o': algo_o,
                            'depth_x': dx,
                            'depth_o': dy,
                        })
                        results.append(m)
                        print(f"Eval: {eval_name}, {algo_x}({dx}) vs {algo_o}({dy}) -> "
                              f"Gagnant: {m['winner'] or 'Nul'}, Temps: {m['total_time']:.3f}s, "
                              f"Nœuds: {m['total_nodes']}")

    # Export CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/benchmark_cross_depth.csv'
    keys = ['eval_func', 'algo_x', 'algo_o', 'depth_x', 'depth_o', 'winner', 'total_time', 'total_nodes', 'moves']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in results:
            w.writerow({k: row[k] for k in keys})
    print(f"\nRésultats exportés → {csv_path}")

    # -------- Courbes ---------
    df = pd.DataFrame(results)
    for eval_name in df.eval_func.unique():
        for (algo_x, algo_o), group in df[df.eval_func==eval_name].groupby(['algo_x','algo_o']):
            # 1. Temps en fonction de depth_x, courbes pour chaque depth_o
            plt.figure()
            for dy in sorted(group.depth_o.unique()):
                sub = group[group.depth_o == dy].sort_values("depth_x")
                plt.plot(sub["depth_x"], sub["total_time"], marker='o', label=f"Prof O={dy}")
            plt.title(f"Temps total (X:{algo_x} vs O:{algo_o}, {eval_name})")
            plt.xlabel("Profondeur X")
            plt.ylabel("Temps total (s)")
            plt.legend(title="Profondeur O")
            plt.grid()
            plt.savefig(f"results/courbe_temps_X_{eval_name}_{algo_x}_vs_{algo_o}.png")
            plt.close()

            # 2. Temps en fonction de depth_o, courbes pour chaque depth_x
            plt.figure()
            for dx in sorted(group.depth_x.unique()):
                sub = group[group.depth_x == dx].sort_values("depth_o")
                plt.plot(sub["depth_o"], sub["total_time"], marker='o', label=f"Prof X={dx}")
            plt.title(f"Temps total (O:{algo_o} vs X:{algo_x}, {eval_name})")
            plt.xlabel("Profondeur O")
            plt.ylabel("Temps total (s)")
            plt.legend(title="Profondeur X")
            plt.grid()
            plt.savefig(f"results/courbe_temps_O_{eval_name}_{algo_x}_vs_{algo_o}.png")
            plt.close()

            # 3. Nœuds explorés en fonction de depth_x
            plt.figure()
            for dy in sorted(group.depth_o.unique()):
                sub = group[group.depth_o == dy].sort_values("depth_x")
                plt.plot(sub["depth_x"], sub["total_nodes"], marker='o', label=f"Prof O={dy}")
            plt.title(f"Nœuds explorés (X:{algo_x} vs O:{algo_o}, {eval_name})")
            plt.xlabel("Profondeur X")
            plt.ylabel("Nombre de nœuds")
            plt.legend(title="Profondeur O")
            plt.grid()
            plt.savefig(f"results/courbe_noeuds_X_{eval_name}_{algo_x}_vs_{algo_o}.png")
            plt.close()

            # 4. Victoire X en fonction de depth_x
            plt.figure()
            for dy in sorted(group.depth_o.unique()):
                sub = group[group.depth_o == dy].sort_values("depth_x")
                win_x = (sub['winner'] == 'X').astype(float)*100
                plt.plot(sub["depth_x"], win_x, marker='o', label=f"Prof O={dy}")
            plt.title(f"Taux victoire X (X:{algo_x} vs O:{algo_o}, {eval_name})")
            plt.xlabel("Profondeur X")
            plt.ylabel("Victoire X (%)")
            plt.ylim(-5, 105)
            plt.legend(title="Profondeur O")
            plt.grid()
            plt.savefig(f"results/courbe_winrate_X_{eval_name}_{algo_x}_vs_{algo_o}.png")
            plt.close()

if __name__ == "__main__":
    main()
