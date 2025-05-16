import os
import time
import statistics
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import csv
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
            if all(board[r][c+i]==player for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c]==player for i in range(4)):
                return True
    for r in range(ROWS-3):
        for c in range(COLS-3):
            if all(board[r+i][c+i]==player for i in range(4)):
                return True
    for r in range(3, ROWS):
        for c in range(COLS-3):
            if all(board[r-i][c+i]==player for i in range(4)):
                return True
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


def simulate_game(depth_x: int, depth_o: int, algo_x: str, algo_o: str,
                  eval_func) -> dict:
    minimax.evaluate = eval_func
    alphabeta.evaluate = eval_func

    mm_wrapper = MinimaxWrapper(minimax.minimax)
    ab_wrapper = AlphaBetaWrapper(alphabeta.alphabeta)

    board = create_board()
    current = 'X'
    metrics = {'nodes_per_move': [], 'time_per_move': []}

    while True:
        if current == 'X':
            algo = mm_wrapper if algo_x == 'minimax' else ab_wrapper
            depth = depth_x
        else:
            algo = mm_wrapper if algo_o == 'minimax' else ab_wrapper
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

        if check_win(board, current):
            metrics['winner'] = current
            break
        if is_full(board):
            metrics['winner'] = None
            break
        current = 'O' if current == 'X' else 'X'

    metrics['total_time'] = sum(metrics['time_per_move'])
    metrics['total_nodes'] = sum(metrics['nodes_per_move'])
    metrics['moves'] = len(metrics['time_per_move'])
    return metrics


def main():
    eval_func = good_evaluate

    scenarios = [
        ('minimax', 'minimax'),
        ('alphabeta', 'alphabeta'),
        ('minimax', 'alphabeta'),
    ]

    results = []

    print(f"{'Scénario':15} | {'Prof X':6} | {'Prof O':6} | {'Gagnant':7} | {'Temps total (s)':15} | {'Nœuds totaux':12}")
    print("-"*80)

    for algo_x, algo_o in scenarios:
        for depth_x in range(2, 6):
            for depth_o in range(2, 6):
                metrics = simulate_game(depth_x, depth_o, algo_x, algo_o, eval_func)
                winner = metrics['winner'] if metrics['winner'] else 'Nul'
                print(f"{algo_x + ' vs ' + algo_o:15} | {depth_x:6} | {depth_o:6} | {winner:7} | {metrics['total_time']:<15.4f} | {metrics['total_nodes']:<12}")
                metrics.update({
                    'scenario': f'{algo_x} vs {algo_o}',
                    'depth_x': depth_x,
                    'depth_o': depth_o
                })
                results.append(metrics)

    os.makedirs('results', exist_ok=True)
    csv_file = 'results/benchmark_results.csv'
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'scenario', 'depth_x', 'depth_o', 'winner',
            'total_time', 'total_nodes', 'moves'
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({
                'scenario': row['scenario'],
                'depth_x': row['depth_x'],
                'depth_o': row['depth_o'],
                'winner': row['winner'] if row['winner'] else 'Nul',
                'total_time': row['total_time'],
                'total_nodes': row['total_nodes'],
                'moves': row['moves']
            })

    print(f"\nRésultats exportés dans {csv_file}")


if __name__ == '__main__':
    main()
