#!/usr/bin/env python3
"""
Helix - Self-Replicating Code Evolution Engine

Command-line interface for running genetic evolution experiments.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import EvolutionEngine, EvolutionConfig
from src.fitness import (
    fitness_symbolic_regression,
    fitness_classification,
    fitness_parsimony,
)
from src.environment import (
    Environment,
    create_task_suite,
    create_symbolic_regression_task,
    DifficultyLevel,
)
from src.visualization import EvolutionTree, EvolutionVisualizer


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Helix - Self-Replicating Code Evolution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evolution with default settings
  python -m src.main

  # Evolve for 100 generations
  python -m src.main --generations 100

  # Run with custom population size
  python -m src.main --population 200 --generations 500

  # Use symbolic regression task
  python -m src.main --task symbolic --generations 200

  # Export results
  python -m src.main --export-results results.json --export-code solution.py
        """
    )
    
    # Evolution settings
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=100,
        help='Maximum number of generations (default: 100)'
    )
    parser.add_argument(
        '--population', '-p',
        type=int,
        default=100,
        help='Population size (default: 100)'
    )
    parser.add_argument(
        '--mutation-rate', '-m',
        type=float,
        default=0.1,
        help='Mutation rate (default: 0.1)'
    )
    parser.add_argument(
        '--crossover-rate', '-c',
        type=float,
        default=0.7,
        help='Crossover rate (default: 0.7)'
    )
    
    # Task selection
    parser.add_argument(
        '--task', '-t',
        choices=['symbolic', 'classification', 'optimization', 'pattern', 'multi'],
        default='multi',
        help='Evolution task (default: multi)'
    )
    
    # Output options
    parser.add_argument(
        '--export-results',
        type=str,
        help='Export statistics to JSON file'
    )
    parser.add_argument(
        '--export-code',
        type=str,
        help='Export best genome as Python code'
    )
    parser.add_argument(
        '--export-tree',
        type=str,
        help='Export evolution tree to JSON file'
    )
    parser.add_argument(
        '--export-html',
        type=str,
        help='Export HTML dashboard'
    )
    
    # Display options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Generations between log output (default: 10)'
    )
    
    # Seed
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    # Target fitness
    parser.add_argument(
        '--target',
        type=float,
        default=0.99,
        help='Target fitness to reach (default: 0.99)'
    )
    
    return parser


def create_environment(task: str) -> Environment:
    """Create environment based on task selection."""
    if task == 'symbolic':
        task_dist = create_task_suite(1)
        task_dist.tasks = [create_symbolic_regression_task(DifficultyLevel.MEDIUM)]
    elif task == 'classification':
        from src.environment import create_classification_task
        task_dist = create_task_suite(1)
        task_dist.tasks = [create_classification_task(2, DifficultyLevel.MEDIUM)]
    elif task == 'optimization':
        from src.environment import create_optimization_task
        task_dist = create_task_suite(1)
        task_dist.tasks = [create_optimization_task(DifficultyLevel.MEDIUM)]
    elif task == 'pattern':
        from src.environment import create_pattern_generation_task
        task_dist = create_task_suite(1)
        task_dist.tasks = [create_pattern_generation_task(DifficultyLevel.MEDIUM)]
    else:  # multi
        task_dist = create_task_suite(5)
    
    return Environment(name=task, task_distribution=task_dist)


def run_evolution(args) -> tuple:
    """Run the evolution and return results."""
    # Create configuration
    config = EvolutionConfig(
        population_size=args.population,
        max_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        verbose=not args.quiet and not args.verbose,
        log_interval=args.log_interval,
        target_fitness=args.target,
        random_seed=args.seed
    )
    
    # Create environment
    environment = create_environment(args.task)
    
    # Create engine
    engine = EvolutionEngine(config, environment=environment)
    
    # Track evolution tree
    tree = EvolutionTree()
    
    # Add callback to track tree
    def track_evolution(eng):
        best = eng.population.get_best(1)[0]
        # Get parent (best from previous generation)
        if len(eng.population.history) > 1:
            prev_best = eng.population.history[-2].best_genome_id
            prev_genome = None
            for g in eng.population.individuals:
                if g.id == prev_best:
                    prev_genome = g
                    break
        else:
            prev_genome = None
        tree.add_genome(best, parent=prev_genome)
    
    engine.add_callback(track_evolution)
    
    # Run evolution
    print(f"Starting Helix Evolution Engine")
    print(f"  Task: {args.task}")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Mutation Rate: {args.mutation_rate}")
    print(f"  Crossover Rate: {args.crossover_rate}")
    print("-" * 50)
    
    start_time = time.time()
    best_genome = engine.evolve()
    elapsed = time.time() - start_time
    
    # Get statistics
    stats = engine.get_statistics()
    
    return best_genome, stats, tree, engine


def export_results(args, best_genome, stats, tree, engine):
    """Export results to files."""
    # Export JSON results
    if args.export_results:
        results = {
            'statistics': stats.to_dict(),
            'config': {
                'population_size': args.population,
                'max_generations': args.generations,
                'mutation_rate': args.mutation_rate,
                'crossover_rate': args.crossover_rate,
                'task': args.task
            },
            'best_genome': {
                'id': best_genome.id,
                'fitness': best_genome.fitness,
                'complexity': best_genome.complexity,
                'num_genes': len(best_genome.genes),
                'generation': best_genome.generation
            }
        }
        
        with open(args.export_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {args.export_results}")
    
    # Export Python code
    if args.export_code:
        code = best_genome.to_code()
        with open(args.export_code, 'w') as f:
            f.write(code)
        print(f"Code exported to {args.export_code}")
    
    # Export evolution tree
    if args.export_tree:
        visualizer = EvolutionVisualizer(tree)
        tree_json = visualizer.generate_json_export()
        with open(args.export_tree, 'w') as f:
            f.write(tree_json)
        print(f"Evolution tree exported to {args.export_tree}")
    
    # Export HTML dashboard
    if args.export_html:
        visualizer = EvolutionVisualizer(tree)
        html = visualizer.generate_html_dashboard(
            stats.fitness_history,
            stats.diversity_history,
            title=f"Helix Evolution - {args.task.title()} Task"
        )
        with open(args.export_html, 'w') as f:
            f.write(html)
        print(f"HTML dashboard exported to {args.export_html}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        best_genome, stats, tree, engine = run_evolution(args)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVOLUTION COMPLETE")
        print("=" * 50)
        print(f"Generations:    {stats.generations_completed}")
        print(f"Total Time:     {stats.total_time:.2f} seconds")
        print(f"Best Fitness:  {stats.best_fitness_ever:.4f}")
        print(f"Converged:     {stats.converged}")
        if stats.convergence_generation:
            print(f"Converged at:  Generation {stats.convergence_generation}")
        print("=" * 50)
        
        # Export results
        export_results(args, best_genome, stats, tree, engine)
        
        # Show best code if verbose
        if args.verbose:
            print("\n--- Best Genome Code ---")
            print(best_genome.to_code())
        
        return 0 if stats.converged else 1
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
