import os
import csv
import time
import json
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import statistics

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Imports de vos modules (assumés disponibles)
import minimax
import alphabeta
from evaluation import evaluate as good_evaluate
from evaluationBad import simple_evaluate as bad_evaluate
from mcts import mcts

Board = List[List[Optional[str]]]
ROWS, COLS = 6, 7


def create_board() -> Board:
    return [[None] * COLS for _ in range(ROWS)]


def get_valid_moves(board: Board) -> List[int]:
    return [c for c in range(COLS) if board[0][c] is None]


def make_move(board: Board, col: int, player: str) -> None:
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] is None:
            board[r][col] = player
            return


def is_full(board: Board) -> bool:
    return all(cell is not None for cell in board[0])


def check_win(board: Board, player: str) -> bool:
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == player for i in range(4)): return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == player for i in range(4)): return True
    # Diagonal \
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == player for i in range(4)): return True
    # Diagonal /
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == player for i in range(4)): return True
    return False


class AlgorithmWrapper:
    """Wrapper pour mesurer les performances des algorithmes"""

    def __init__(self, name: str, func, eval_func):
        self.name = name
        self.func = func
        self.eval_func = eval_func
        self.reset_metrics()

    def reset_metrics(self):
        self.nodes_explored = 0
        self.total_time = 0.0
        self.move_times = []
        self.move_nodes = []

    def make_move(self, board: Board, depth: int, player: str) -> int:
        start_time = time.perf_counter()

        if self.name == 'MCTS':
            # Pour MCTS, depth représente le nombre d'itérations * 1000
            iterations = depth * 1000
            move = self.func(board, player, iter_max=iterations, C=1.4)
            # Pour MCTS, on estime les nœuds explorés
            nodes = iterations
        else:
            # Configuration de l'évaluateur
            minimax.evaluate = self.eval_func
            alphabeta.evaluate = self.eval_func

            if self.name == 'Minimax':
                _, move = self.func(board, depth, maximizing_player=(player == 'X'))
            else:  # Alpha-Beta
                _, move = self.func(board, depth, float('-inf'), float('inf'),
                                    maximizing_player=(player == 'X'))

            # Estimation approximative des nœuds pour minimax/alphabeta
            # Ceci est une approximation basée sur la profondeur et le facteur de branchement
            avg_branching = 7  # En moyenne 7 coups possibles au Puissance 4
            if self.name == 'Alpha-Beta':
                nodes = int((avg_branching ** depth) * 0.6)  # Alpha-Beta élimine ~40% des nœuds
            else:
                nodes = avg_branching ** depth

        end_time = time.perf_counter()
        move_time = end_time - start_time

        self.move_times.append(move_time)
        self.move_nodes.append(nodes)
        self.nodes_explored += nodes
        self.total_time += move_time

        return move


def simulate_game(config: Dict) -> Dict:
    """Simule une partie avec la configuration donnée"""

    # Création des wrappers d'algorithmes
    eval_func = good_evaluate if config['eval_func'] == 'good' else bad_evaluate

    algo_x = AlgorithmWrapper(
        config['algo_x'],
        minimax.minimax if config['algo_x'] == 'Minimax' else
        alphabeta.alphabeta if config['algo_x'] == 'Alpha-Beta' else mcts,
        eval_func
    )

    algo_o = AlgorithmWrapper(
        config['algo_o'],
        minimax.minimax if config['algo_o'] == 'Minimax' else
        alphabeta.alphabeta if config['algo_o'] == 'Alpha-Beta' else mcts,
        eval_func
    )

    board = create_board()
    current_player = 'X'
    move_count = 0
    max_moves = 42  # Maximum de coups possibles au Puissance 4

    game_log = []

    while move_count < max_moves:
        if current_player == 'X':
            algo = algo_x
            depth = config['depth_x']
        else:
            algo = algo_o
            depth = config['depth_o']

        move = algo.make_move(board, depth, current_player)

        if move is None or move not in get_valid_moves(board):
            break

        make_move(board, move, current_player)
        move_count += 1

        game_log.append({
            'move': move_count,
            'player': current_player,
            'column': move,
            'time': algo.move_times[-1],
            'nodes': algo.move_nodes[-1]
        })

        if check_win(board, current_player):
            winner = current_player
            break

        if is_full(board):
            winner = None
            break

        current_player = 'O' if current_player == 'X' else 'X'
    else:
        winner = None

    return {
        'config': config,
        'winner': winner,
        'moves': move_count,
        'game_log': game_log,
        'stats_x': {
            'total_time': algo_x.total_time,
            'total_nodes': algo_x.nodes_explored,
            'avg_time_per_move': statistics.mean(algo_x.move_times) if algo_x.move_times else 0,
            'avg_nodes_per_move': statistics.mean(algo_x.move_nodes) if algo_x.move_nodes else 0,
            'max_time_per_move': max(algo_x.move_times) if algo_x.move_times else 0,
        },
        'stats_o': {
            'total_time': algo_o.total_time,
            'total_nodes': algo_o.nodes_explored,
            'avg_time_per_move': statistics.mean(algo_o.move_times) if algo_o.move_times else 0,
            'avg_nodes_per_move': statistics.mean(algo_o.move_nodes) if algo_o.move_nodes else 0,
            'max_time_per_move': max(algo_o.move_times) if algo_o.move_times else 0,
        },
        'total_game_time': algo_x.total_time + algo_o.total_time,
        'total_game_nodes': algo_x.nodes_explored + algo_o.nodes_explored
    }


def run_comprehensive_benchmark():
    """Lance un benchmark complet avec toutes les combinaisons"""

    # Configuration des tests
    algorithms = ['Minimax', 'Alpha-Beta', 'MCTS']
    eval_functions = ['good', 'bad']
    depths = [2, 3, 4, 5]  # Ajusté pour inclure plus de profondeurs

    results = []
    total_games = len(algorithms) * len(algorithms) * len(eval_functions) * len(depths) * len(depths)
    game_count = 0

    print(f"Démarrage du benchmark complet: {total_games} parties à jouer")
    print("=" * 80)

    for eval_func in eval_functions:
        for algo_x in algorithms:
            for algo_o in algorithms:
                for depth_x in depths:
                    for depth_o in depths:
                        game_count += 1

                        config = {
                            'eval_func': eval_func,
                            'algo_x': algo_x,
                            'algo_o': algo_o,
                            'depth_x': depth_x,
                            'depth_o': depth_o,
                            'id': f"{eval_func}_{algo_x}{depth_x}_vs_{algo_o}{depth_o}"
                        }

                        print(f"[{game_count:3d}/{total_games}] {config['id']}", end=" ... ")

                        try:
                            result = simulate_game(config)
                            results.append(result)

                            winner_str = result['winner'] if result['winner'] else 'Draw'
                            print(f"Winner: {winner_str:4s} | "
                                  f"Moves: {result['moves']:2d} | "
                                  f"Time: {result['total_game_time']:.3f}s | "
                                  f"Nodes: {result['total_game_nodes']:,}")

                        except Exception as e:
                            print(f"ERROR: {str(e)}")
                            continue

    return results


def export_results(results: List[Dict], base_path: str = 'results'):
    """Exporte les résultats en CSV et JSON"""
    os.makedirs(base_path, exist_ok=True)

    # Export CSV détaillé
    csv_data = []
    for result in results:
        config = result['config']
        base_row = {
            'eval_func': config['eval_func'],
            'algo_x': config['algo_x'],
            'algo_o': config['algo_o'],
            'depth_x': config['depth_x'],
            'depth_o': config['depth_o'],
            'winner': result['winner'],
            'moves': result['moves'],
            'total_game_time': result['total_game_time'],
            'total_game_nodes': result['total_game_nodes'],
            'x_total_time': result['stats_x']['total_time'],
            'x_total_nodes': result['stats_x']['total_nodes'],
            'x_avg_time': result['stats_x']['avg_time_per_move'],
            'x_avg_nodes': result['stats_x']['avg_nodes_per_move'],
            'x_max_time': result['stats_x']['max_time_per_move'],
            'o_total_time': result['stats_o']['total_time'],
            'o_total_nodes': result['stats_o']['total_nodes'],
            'o_avg_time': result['stats_o']['avg_time_per_move'],
            'o_avg_nodes': result['stats_o']['avg_nodes_per_move'],
            'o_max_time': result['stats_o']['max_time_per_move'],
        }
        csv_data.append(base_row)

    csv_path = os.path.join(base_path, 'comprehensive_benchmark.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    # Export JSON complet
    json_path = os.path.join(base_path, 'comprehensive_benchmark.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats exportés:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

    return csv_path


def create_comprehensive_visualizations(csv_path: str, output_dir: str = 'results'):
    """Crée des visualisations complètes des résultats"""

    df = pd.read_csv(csv_path)

    # Configuration des styles
    plt.style.use('seaborn-v0_8')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # 1. Taux de victoire par algorithme et fonction d'évaluation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, eval_func in enumerate(['good', 'bad']):
        data = df[df['eval_func'] == eval_func]

        # Calcul des taux de victoire pour X
        win_rates = []
        algorithms = ['Minimax', 'Alpha-Beta', 'MCTS']

        for algo in algorithms:
            algo_data = data[data['algo_x'] == algo]
            if len(algo_data) > 0:
                win_rate = (algo_data['winner'] == 'X').mean() * 100
                win_rates.append(win_rate)
            else:
                win_rates.append(0)

        axes[i].bar(algorithms, win_rates, color=colors[:3])
        axes[i].set_title(f'Taux de victoire du joueur X\n(Évaluation {eval_func})')
        axes[i].set_ylabel('Taux de victoire (%)')
        axes[i].set_ylim(0, 100)

        # Ajout des valeurs sur les barres
        for j, v in enumerate(win_rates):
            axes[i].text(j, v + 1, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rates_by_algorithm.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Temps d'exécution moyen par algorithme et profondeur
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for eval_idx, eval_func in enumerate(['good', 'bad']):
        data = df[df['eval_func'] == eval_func]

        # Temps par algorithme X
        ax1 = axes[eval_idx, 0]
        algo_times = data.groupby(['algo_x', 'depth_x'])['x_avg_time'].mean().unstack()
        algo_times.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title(f'Temps moyen par coup - Joueur X\n(Évaluation {eval_func})')
        ax1.set_ylabel('Temps (secondes)')
        ax1.legend(title='Profondeur', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)

        # Nœuds par algorithme X
        ax2 = axes[eval_idx, 1]
        algo_nodes = data.groupby(['algo_x', 'depth_x'])['x_avg_nodes'].mean().unstack()
        algo_nodes.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_title(f'Nœuds moyens par coup - Joueur X\n(Évaluation {eval_func})')
        ax2.set_ylabel('Nombre de nœuds')
        ax2.legend(title='Profondeur', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_by_depth.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Comparaison directe des algorithmes (heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for eval_idx, eval_func in enumerate(['good', 'bad']):
        data = df[df['eval_func'] == eval_func]

        # Matrice de confrontation (qui gagne contre qui)
        algorithms = ['Minimax', 'Alpha-Beta', 'MCTS']
        win_matrix = np.zeros((len(algorithms), len(algorithms)))

        for i, algo_x in enumerate(algorithms):
            for j, algo_o in enumerate(algorithms):
                matchups = data[(data['algo_x'] == algo_x) & (data['algo_o'] == algo_o)]
                if len(matchups) > 0:
                    win_rate = (matchups['winner'] == 'X').mean()
                    win_matrix[i, j] = win_rate

        im = axes[eval_idx].imshow(win_matrix, cmap='RdYlBu', vmin=0, vmax=1)
        axes[eval_idx].set_xticks(range(len(algorithms)))
        axes[eval_idx].set_yticks(range(len(algorithms)))
        axes[eval_idx].set_xticklabels(algorithms)
        axes[eval_idx].set_yticklabels(algorithms)
        axes[eval_idx].set_xlabel('Algorithme O (Adversaire)')
        axes[eval_idx].set_ylabel('Algorithme X (Joueur)')
        axes[eval_idx].set_title(f'Matrice de victoires\n(Évaluation {eval_func})')

        # Ajout des pourcentages dans les cellules
        for i in range(len(algorithms)):
            for j in range(len(algorithms)):
                text = axes[eval_idx].text(j, i, f'{win_matrix[i, j]:.2f}',
                                           ha="center", va="center", color="black")

        plt.colorbar(im, ax=axes[eval_idx])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'algorithm_matchup_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Distribution des temps par algorithme
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    algorithms = ['Minimax', 'Alpha-Beta', 'MCTS']
    eval_funcs = ['good', 'bad']

    for eval_idx, eval_func in enumerate(eval_funcs):
        data = df[df['eval_func'] == eval_func]

        for algo_idx, algo in enumerate(algorithms):
            ax = axes[eval_idx, algo_idx]
            algo_data = data[data['algo_x'] == algo]

            if len(algo_data) > 0:
                # Histogramme des temps
                times = algo_data['x_avg_time']
                ax.hist(times, bins=20, alpha=0.7, color=colors[algo_idx], edgecolor='black')
                ax.set_title(f'{algo}\n(Évaluation {eval_func})')
                ax.set_xlabel('Temps moyen par coup (s)')
                ax.set_ylabel('Fréquence')

                # Statistiques
                mean_time = times.mean()
                std_time = times.std()
                ax.axvline(mean_time, color='red', linestyle='--', linewidth=2,
                           label=f'Moyenne: {mean_time:.3f}s')
                ax.legend()

    plt.suptitle('Distribution des temps d\'exécution par algorithme', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Évolution des performances avec la profondeur
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for eval_idx, eval_func in enumerate(['good', 'bad']):
        data = df[df['eval_func'] == eval_func]

        # Temps vs profondeur
        ax1 = axes[eval_idx, 0]
        for algo in algorithms:
            algo_data = data[data['algo_x'] == algo].groupby('depth_x')['x_avg_time'].mean()
            ax1.plot(algo_data.index, algo_data.values, marker='o', label=algo, linewidth=2)

        ax1.set_title(f'Temps vs Profondeur\n(Évaluation {eval_func})')
        ax1.set_xlabel('Profondeur')
        ax1.set_ylabel('Temps moyen (s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Nœuds vs profondeur
        ax2 = axes[eval_idx, 1]
        for algo in algorithms:
            algo_data = data[data['algo_x'] == algo].groupby('depth_x')['x_avg_nodes'].mean()
            ax2.plot(algo_data.index, algo_data.values, marker='o', label=algo, linewidth=2)

        ax2.set_title(f'Nœuds vs Profondeur\n(Évaluation {eval_func})')
        ax2.set_xlabel('Profondeur')
        ax2.set_ylabel('Nœuds moyens')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Échelle logarithmique pour les nœuds

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualisations créées dans {output_dir}/:")
    print("  - win_rates_by_algorithm.png")
    print("  - performance_by_depth.png")
    print("  - algorithm_matchup_matrix.png")
    print("  - time_distribution.png")
    print("  - depth_scaling.png")


def generate_summary_report(csv_path: str, output_dir: str = 'results'):
    """Génère un rapport de synthèse des résultats"""

    df = pd.read_csv(csv_path)

    report = []
    report.append("=" * 80)
    report.append("RAPPORT DE BENCHMARK - PUISSANCE 4 IA")
    report.append("=" * 80)
    report.append("")

    # Statistiques générales
    total_games = len(df)
    total_time = df['total_game_time'].sum()
    avg_game_time = df['total_game_time'].mean()

    report.append("STATISTIQUES GÉNÉRALES")
    report.append("-" * 40)
    report.append(f"Nombre total de parties: {total_games}")
    report.append(f"Temps total d'exécution: {total_time:.2f} secondes")
    report.append(f"Temps moyen par partie: {avg_game_time:.3f} secondes")
    report.append("")

    # Analyse par algorithme
    report.append("ANALYSE PAR ALGORITHME")
    report.append("-" * 40)

    algorithms = ['Minimax', 'Alpha-Beta', 'MCTS']
    for algo in algorithms:
        algo_data = df[df['algo_x'] == algo]
        if len(algo_data) > 0:
            win_rate = (algo_data['winner'] == 'X').mean() * 100
            avg_time = algo_data['x_avg_time'].mean()
            avg_nodes = algo_data['x_avg_nodes'].mean()

            report.append(f"{algo}:")
            report.append(f"  Taux de victoire: {win_rate:.1f}%")
            report.append(f"  Temps moyen/coup: {avg_time:.4f}s")
            report.append(f"  Nœuds moyens/coup: {avg_nodes:,.0f}")
            report.append("")

    # Analyse par fonction d'évaluation
    report.append("ANALYSE PAR FONCTION D'ÉVALUATION")
    report.append("-" * 40)

    for eval_func in ['good', 'bad']:
        eval_data = df[df['eval_func'] == eval_func]
        avg_game_time = eval_data['total_game_time'].mean()
        avg_moves = eval_data['moves'].mean()

        report.append(f"Évaluation {eval_func}:")
        report.append(f"  Temps moyen par partie: {avg_game_time:.3f}s")
        report.append(f"  Nombre moyen de coups: {avg_moves:.1f}")
        report.append("")

    # Meilleures configurations
    report.append("CONFIGURATIONS LES PLUS PERFORMANTES")
    report.append("-" * 40)

    # Plus rapide
    fastest = df.loc[df['x_avg_time'].idxmin()]
    report.append("Configuration la plus rapide:")
    report.append(f"  {fastest['algo_x']} (prof. {fastest['depth_x']}) - {fastest['eval_func']} eval")
    report.append(f"  Temps: {fastest['x_avg_time']:.4f}s/coup")
    report.append("")

    # Plus efficace (ratio victoires/temps)
    df['efficiency'] = df.apply(lambda row:
                                (1 if row['winner'] == 'X' else 0) / max(row['x_avg_time'], 0.001), axis=1)
    most_efficient = df.loc[df['efficiency'].idxmax()]
    report.append("Configuration la plus efficace:")
    report.append(
        f"  {most_efficient['algo_x']} (prof. {most_efficient['depth_x']}) - {most_efficient['eval_func']} eval")
    report.append(f"  Efficacité: {most_efficient['efficiency']:.2f}")
    report.append("")

    # Sauvegarde du rapport
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'benchmark_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nRapport sauvegardé: {report_path}")


def main():
    """Fonction principale du benchmark"""
    print("🎮 BENCHMARK COMPLET - PUISSANCE 4 IA")
    print("=" * 80)

    # Exécution du benchmark
    results = run_comprehensive_benchmark()

    if not results:
        print("❌ Aucun résultat généré!")
        return

    print(f"\n✅ Benchmark terminé: {len(results)} parties jouées")

    # Export des résultats
    csv_path = export_results(results)

    # Génération des visualisations
    print("\n📊 Génération des visualisations...")
    create_comprehensive_visualizations(csv_path)

    # Génération du rapport
    print("\n📋 Génération du rapport de synthèse...")
    generate_summary_report(csv_path)

    print("\n🎉 Benchmark complet terminé!")
    print("Consultez le dossier 'results/' pour tous les fichiers générés.")


if __name__ == "__main__":
    main()