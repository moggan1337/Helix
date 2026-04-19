#!/usr/bin/env python3
"""
Interactive Evolution Demo

This example provides an interactive demonstration of the evolution
process, allowing step-by-step execution and visualization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import EvolutionEngine, EvolutionConfig
from src.population import Population
from src.visualization import EvolutionTree, EvolutionVisualizer
from src.environment import create_task_suite, Environment


def run_interactive_demo():
    """Run interactive evolution demo."""
    
    print("=" * 60)
    print("HELIX INTERACTIVE EVOLUTION DEMO")
    print("=" * 60)
    print()
    print("This demo shows evolution step-by-step.")
    print()
    
    # Create engine
    config = EvolutionConfig(
        population_size=30,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.7,
        verbose=False,  # We'll handle output manually
        random_seed=789
    )
    
    environment = Environment(name="interactive", task_distribution=create_task_suite(3))
    engine = EvolutionEngine(config, environment=environment)
    
    # Initialize population
    engine.population.initialize_random(genes_per_genome=4, complexity_per_gene=3)
    
    # Evaluate initial population
    engine._evaluate_population()
    
    print(f"Initial population: {len(engine.population)} individuals")
    print()
    
    # Track evolution
    tree = EvolutionTree()
    
    # Initial evaluation
    best = engine.population.get_best(1)[0]
    tree.add_genome(best)
    
    print("Initial Best:")
    print(f"  Fitness: {best.fitness:.4f}")
    print(f"  Complexity: {best.complexity}")
    print()
    
    # Evolution loop with visualization
    for gen in range(config.max_generations):
        # Show progress every 5 generations
        if gen % 5 == 0:
            stats = engine.population.get_statistics()
            print(f"Generation {gen}:")
            print(f"  Best: {stats.best_fitness:.4f} | " +
                  f"Mean: {stats.mean_fitness:.4f} | " +
                  f"Diversity: {stats.diversity:.2f}")
        
        # Perform one evolution step
        engine._evolve_step()
        
        # Track in tree
        best = engine.population.get_best(1)[0]
        prev_best_id = tree.get_best_lineage()[-1].genome_id if tree.get_best_lineage() else None
        prev_genome = engine.population.get_best(1)[0]
        tree.add_genome(best, parent=prev_genome)
        
        # Check for convergence
        if best.fitness >= config.target_fitness:
            print(f"\nTarget fitness reached at generation {gen}!")
            break
    
    # Final results
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    stats = engine.population.get_statistics()
    print(f"Generations: {gen + 1}")
    print(f"Final Best Fitness: {stats.best_fitness:.4f}")
    print(f"Mean Fitness: {stats.mean_fitness:.4f}")
    print(f"Diversity: {stats.diversity:.4f}")
    print()
    
    # Show evolution tree summary
    print("Evolution Tree Statistics:")
    print("-" * 40)
    tree_stats = tree.calculate_statistics()
    print(f"  Total Nodes: {tree_stats['total_nodes']}")
    print(f"  Total Generations: {tree_stats['total_generations']}")
    print(f"  Avg Branching Factor: {tree_stats['avg_branching_factor']:.2f}")
    print(f"  Best Fitness: {tree_stats['best_fitness']:.4f}")
    print()
    
    # Show best lineage
    print("Best Fitness Lineage:")
    lineage = tree.get_best_lineage()
    for i, node in enumerate(lineage):
        change = f"+{node.fitness_change:.4f}" if node.fitness_change >= 0 else f"{node.fitness_change:.4f}"
        print(f"  Gen {node.generation}: {node.fitness:.4f} ({change})")
    
    return engine, tree


def visualize_progress(engine: EvolutionEngine, tree: EvolutionTree):
    """Create visualization of evolution progress."""
    
    print()
    print("=" * 60)
    print("VISUALIZATION DATA")
    print("=" * 60)
    
    # Create visualizer
    visualizer = EvolutionVisualizer(tree)
    
    # ASCII tree
    print("\nEvolution Tree (ASCII):")
    print("-" * 40)
    print(visualizer.generate_console_tree(max_depth=min(10, tree.total_generations)))
    
    # Statistics report
    print()
    print("Statistics Report:")
    print("-" * 40)
    print(visualizer.generate_statistics_report())
    
    # HTML dashboard
    html = visualizer.generate_html_dashboard(
        engine.statistics.fitness_history,
        engine.statistics.diversity_history
    )
    
    html_file = Path(__file__).parent.parent / "output" / "dashboard.html"
    html_file.parent.mkdir(exist_ok=True)
    with open(html_file, 'w') as f:
        f.write(html)
    
    print(f"\nHTML Dashboard saved to: {html_file}")


def show_genome_details(engine: EvolutionEngine):
    """Show detailed information about top genomes."""
    
    print()
    print("=" * 60)
    print("TOP GENOMES")
    print("=" * 60)
    
    top_5 = engine.population.get_best(5)
    
    for i, genome in enumerate(top_5, 1):
        print(f"\n#{i} (Fitness: {genome.fitness:.4f})")
        print("-" * 40)
        print(f"  ID: {genome.id[:16]}")
        print(f"  Generation: {genome.generation}")
        print(f"  Complexity: {genome.complexity}")
        print(f"  Genes: {len(genome.genes)}")
        print(f"  Active Genes: {len(genome.active_genes)}")
        
        if genome.genes:
            print("  Gene Types:")
            gene_types = {}
            for gene in genome.genes:
                gt_name = gene.gene_type.name
                gene_types[gt_name] = gene_types.get(gt_name, 0) + 1
            for gt, count in sorted(gene_types.items()):
                print(f"    - {gt}: {count}")


if __name__ == '__main__':
    engine, tree = run_interactive_demo()
    visualize_progress(engine, tree)
    show_genome_details(engine)
