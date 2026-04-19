#!/usr/bin/env python3
"""
Classification Example

This example demonstrates using Helix to evolve classifiers that
categorize inputs into discrete classes.

Task: Classify 2D points into two categories based on their position.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolution import EvolutionEngine, EvolutionConfig
from src.genome import Genome
from src.environment import (
    Environment,
    TaskDistribution,
    create_classification_task,
    DifficultyLevel
)
from src.fitness import fitness_classification


def run_classification():
    """Run classification evolution."""
    
    print("=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)
    print()
    print("Task: Classify 2D points into two categories")
    print("  Class 0: Points near (0, 0)")
    print("  Class 1: Points near (1, 1)")
    print()
    
    # Generate synthetic classification data
    import random
    random.seed(42)
    
    patterns = []
    for _ in range(100):
        if random.random() < 0.5:
            # Class 0: centered at (0, 0)
            x = random.gauss(0, 0.5)
            y = random.gauss(0, 0.5)
            label = 0
        else:
            # Class 1: centered at (1, 1)
            x = random.gauss(1, 0.5)
            y = random.gauss(1, 0.5)
            label = 1
        patterns.append(((x, y), label))
    
    print(f"Training samples: {len(patterns)}")
    print()
    
    # Create fitness function
    fitness_fn = fitness_classification(patterns, num_classes=2)
    
    # Create environment
    task_dist = TaskDistribution()
    task_dist.add_task(create_classification_task(2, DifficultyLevel.MEDIUM))
    environment = Environment(name="classification", task_distribution=task_dist)
    
    # Create engine configuration
    config = EvolutionConfig(
        population_size=80,
        max_generations=150,
        mutation_rate=0.1,
        crossover_rate=0.75,
        verbose=True,
        log_interval=20,
        target_fitness=0.85,
        random_seed=123
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
    print(f"Accuracy: {best.fitness * 100:.1f}%")
    print(f"Generations: {engine.statistics.generations_completed}")
    print()
    
    # Test on sample data
    print("Sample predictions:")
    print("-" * 40)
    test_cases = [
        ((0.0, 0.0), 0, "Origin"),
        ((1.0, 1.0), 1, "(1,1)"),
        ((0.5, 0.5), "?", "Midpoint"),
        ((0.2, 0.8), "?", "Near both"),
        ((-0.5, 0.3), 0, "Negative x"),
    ]
    
    for (x, y), expected, desc in test_cases:
        try:
            result = best.execute({'x': x, 'y': y})
            predicted = int(result) % 2 if result is not None else -1
            correct = "✓" if predicted == expected else "✗"
            print(f"  ({x:5.1f}, {y:5.1f}) = {predicted} (expected {expected}) {correct}")
        except Exception:
            print(f"  ({x:5.1f}, {y:5.1f}) = Error")
    
    print()
    print("Evolved classifier code:")
    print("-" * 40)
    print(best.to_code())
    
    return best, engine


def calculate_confusion_matrix(genome: Genome, patterns):
    """Calculate confusion matrix for a classifier."""
    tp = fp = tn = fn = 0
    
    for (x, y), label in patterns:
        try:
            result = genome.execute({'x': x, 'y': y})
            predicted = int(result) % 2 if result is not None else -1
            
            if predicted == 1:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if label == 0:
                    tn += 1
                else:
                    fn += 1
        except Exception:
            pass
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }


if __name__ == '__main__':
    run_classification()
