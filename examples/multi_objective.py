#!/usr/bin/env python3
"""
Multi-Objective Evolution Example

This example demonstrates multi-objective optimization using Helix.
The goal is to simultaneously optimize multiple conflicting objectives.

Objectives:
1. Maximize accuracy on task
2. Minimize code complexity
3. Maximize diversity
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import EvolutionEngine, EvolutionConfig
from src.fitness import FitnessConfig, FitnessEvaluator, fitness_symbolic_regression
from src.environment import Environment, TaskDistribution, create_task_suite
from src.genome import Genome
from src.selection import AdaptiveSelection


def run_multi_objective():
    """Run multi-objective evolution."""
    
    print("=" * 60)
    print("MULTI-OBJECTIVE EVOLUTION EXAMPLE")
    print("=" * 60)
    print()
    print("Objectives:")
    print("  1. Maximize accuracy")
    print("  2. Minimize complexity")
    print("  3. Maintain diversity")
    print()
    
    # Generate training data
    x_data = [i * 0.2 for i in range(-10, 11)]
    y_data = [x**2 for x in x_data]
    
    # Create fitness functions for each objective
    accuracy_fn = fitness_symbolic_regression(x_data, y_data)
    
    def complexity_fn(genome: Genome, context: dict) -> float:
        """Penalize complex genomes."""
        optimal = 10
        complexity = genome.complexity
        if complexity <= optimal:
            return 1.0 - (optimal - complexity) * 0.01
        else:
            return max(0.0, 1.0 - (complexity - optimal) * 0.05)
    
    def diversity_fn(genome: Genome, context: dict) -> float:
        """Reward diverse gene types."""
        gene_types = set(g.gene_type for g in genome.genes)
        return len(gene_types) / 15  # Normalize by number of gene types
    
    # Create multi-objective fitness evaluator
    fitness_config = FitnessConfig(
        primary_weight=0.5,
        complexity_penalty=0.2,
        diversity_bonus=0.3,
        multi_objective=True
    )
    evaluator = FitnessEvaluator(fitness_config)
    evaluator.add_fitness_function(accuracy_fn, weight=0.5)
    evaluator.add_fitness_function(complexity_fn, weight=0.3)
    evaluator.add_fitness_function(diversity_fn, weight=0.2)
    
    # Create environment
    task_dist = create_task_suite(3)
    environment = Environment(name="multi_objective", task_distribution=task_dist)
    
    # Create engine with adaptive selection
    config = EvolutionConfig(
        population_size=120,
        max_generations=200,
        mutation_rate=0.12,
        crossover_rate=0.7,
        selection_method='adaptive',
        verbose=True,
        log_interval=25,
        random_seed=456
    )
    
    engine = EvolutionEngine(
        config,
        fitness_evaluator=evaluator,
        environment=environment
    )
    
    print("Starting multi-objective evolution...")
    print("-" * 60)
    
    best = engine.evolve()
    
    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best Fitness: {best.fitness:.4f}")
    print(f"Complexity: {best.complexity}")
    print(f"Num Genes: {len(best.genes)}")
    print(f"Gene Types: {set(g.gene_type.name for g in best.genes)}")
    print()
    
    # Evaluate each objective separately
    print("Objective Breakdown:")
    print("-" * 40)
    accuracy = accuracy_fn(best, {})
    complexity = complexity_fn(best, {})
    diversity = diversity_fn(best, {})
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Complexity: {complexity:.4f}")
    print(f"  Diversity: {diversity:.4f}")
    print()
    
    # Compare with single-objective evolution
    print("Comparison with Single-Objective Evolution:")
    print("-" * 40)
    
    # Run single-objective
    single_evaluator = FitnessEvaluator()
    single_evaluator.add_fitness_function(accuracy_fn, weight=1.0)
    
    single_engine = EvolutionEngine(
        EvolutionConfig(
            population_size=120,
            max_generations=200,
            mutation_rate=0.12,
            crossover_rate=0.7,
            verbose=False,
            random_seed=456
        ),
        fitness_evaluator=single_evaluator,
        environment=environment
    )
    
    single_best = single_engine.evolve()
    
    print(f"  Multi-objective best: accuracy={accuracy:.4f}, complexity={best.complexity}")
    print(f"  Single-objective best: accuracy={single_best.fitness:.4f}, complexity={single_best.complexity}")
    
    return best, engine


def analyze_pareto_front(population: list) -> list:
    """Identify Pareto-optimal individuals."""
    pareto_front = []
    
    for individual in population:
        is_dominated = False
        for other in population:
            if other.id == individual.id:
                continue
            
            # Check if 'other' dominates 'individual'
            # (Better in at least one objective, not worse in any)
            better_or_equal = 0
            strictly_better = 0
            
            objectives = [
                individual.fitness,
                -individual.complexity,  # Negate because we minimize
            ]
            other_objectives = [
                other.fitness,
                -other.complexity,
            ]
            
            for obj, other_obj in zip(objectives, other_objectives):
                if obj <= other_obj:  # <= because higher fitness is better
                    better_or_equal += 1
                if obj < other_obj:
                    strictly_better += 1
            
            if better_or_equal == len(objectives) and strictly_better > 0:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(individual)
    
    return pareto_front


if __name__ == '__main__':
    run_multi_objective()
