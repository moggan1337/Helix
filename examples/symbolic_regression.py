#!/usr/bin/env python3
"""
Symbolic Regression Example

This example demonstrates using Helix to evolve mathematical functions
that approximate given data points. The goal is to find a function f(x)
that matches the training data.

Target function: f(x) = x^2 + 2*x + 1 (perfect square: (x+1)^2)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import EvolutionEngine, EvolutionConfig
from src.genome import Genome
from src.fitness import fitness_symbolic_regression
from src.environment import Environment, create_symbolic_regression_task, DifficultyLevel


def run_symbolic_regression():
    """Run symbolic regression evolution."""
    
    print("=" * 60)
    print("SYMBOLIC REGRESSION EXAMPLE")
    print("=" * 60)
    print()
    print("Target function: f(x) = x² + 2x + 1 = (x + 1)²")
    print("Training data: x in [-5, 5] with step 0.5")
    print()
    
    # Generate training data
    x_data = [i * 0.5 for i in range(-10, 11)]
    y_data = [x**2 + 2*x + 1 for x in x_data]
    
    print(f"Training points: {len(x_data)}")
    print()
    
    # Create fitness function
    fitness_fn = fitness_symbolic_regression(x_data, y_data)
    
    # Create environment with symbolic regression task
    from src.environment import TaskDistribution
    task_dist = TaskDistribution()
    task_dist.add_task(create_symbolic_regression_task(DifficultyLevel.MEDIUM))
    environment = Environment(name="symbolic_regression", task_distribution=task_dist)
    
    # Create engine configuration
    config = EvolutionConfig(
        population_size=100,
        max_generations=200,
        mutation_rate=0.15,
        crossover_rate=0.7,
        verbose=True,
        log_interval=20,
        target_fitness=0.9,
        random_seed=42
    )
    
    # Create and run engine
    engine = EvolutionEngine(config, environment=environment)
    engine.fitness_evaluator.add_fitness_function(fitness_fn)
    
    print("Starting evolution...")
    print("-" * 60)
    
    best = engine.evolve()
    
    # Results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best Fitness: {best.fitness:.4f}")
    print(f"Generations: {engine.statistics.generations_completed}")
    print(f"Time: {engine.statistics.total_time:.2f}s")
    print()
    
    # Test on additional points
    print("Testing on unseen data:")
    test_x = [0.25, 1.5, 2.75, -3.25]
    expected_y = [x**2 + 2*x + 1 for x in test_x]
    
    print("-" * 40)
    for x, expected in zip(test_x, expected_y):
        try:
            context = {'x': x}
            predicted = best.execute(context)
            if predicted is None:
                predicted = "N/A"
            error = abs(predicted - expected) if isinstance(predicted, (int, float)) else "N/A"
            print(f"  x = {x:6.2f} | Expected: {expected:7.2f} | Got: {str(predicted):>7} | Error: {error}")
        except Exception as e:
            print(f"  x = {x:6.2f} | Expected: {expected:7.2f} | Error: Execution failed")
    
    print()
    print("Best evolved code:")
    print("-" * 40)
    print(best.to_code())
    
    return best, engine


def evaluate_quality(genome: Genome, x_data, y_data) -> dict:
    """Evaluate the quality of an evolved function."""
    total_error = 0.0
    max_error = 0.0
    
    predictions = []
    for x, y_expected in zip(x_data, y_data):
        try:
            result = genome.execute({'x': x})
            if result is not None:
                pred = float(result)
                predictions.append(pred)
                error = abs(pred - y_expected)
                total_error += error
                max_error = max(max_error, error)
            else:
                predictions.append(None)
        except Exception:
            predictions.append(None)
    
    valid_count = sum(1 for p in predictions if p is not None)
    
    return {
        'mean_absolute_error': total_error / len(x_data) if len(x_data) > 0 else float('inf'),
        'max_error': max_error,
        'valid_predictions': valid_count,
        'total_points': len(x_data),
        'coverage': valid_count / len(x_data) if len(x_data) > 0 else 0
    }


if __name__ == '__main__':
    run_symbolic_regression()
