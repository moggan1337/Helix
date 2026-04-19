"""
Fitness Module - Evaluation and Selection Pressure

This module implements fitness functions for evaluating how well
genomes solve problems, and mechanisms for applying selection
pressure based on fitness.
"""

from __future__ import annotations

import copy
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .genome import Gene, GeneType, Genome


# Type alias for fitness functions
FitnessFunction = Callable[[Genome, Dict[str, Any]], float]


class FitnessMetrics(Enum):
    """Metrics tracked during fitness evaluation."""
    
    FITNESS = auto()           # Primary fitness score
    COMPLEXITY = auto()        # Genome complexity
    DIVERSITY = auto()         # Population diversity
    CONVERGENCE = auto()       # Convergence measure
    SPECIFICITY = auto()       # Task-specific performance
    GENERALITY = auto()        # Cross-task performance


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""
    
    fitness: float                          # Primary fitness score
    metrics: Dict[FitnessMetrics, float] = field(default_factory=dict)  # Detailed metrics
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details
    tasks_scores: Dict[str, float] = field(default_factory=dict)  # Per-task scores
    
    def __add__(self, other: FitnessResult) -> FitnessResult:
        """Combine two fitness results."""
        return FitnessResult(
            fitness=(self.fitness + other.fitness) / 2,
            metrics={k: (self.metrics.get(k, 0) + other.metrics.get(k, 0)) / 2
                     for k in set(self.metrics) | set(other.metrics)},
            details={**self.details, **other.details},
            tasks_scores={**self.tasks_scores, **other.tasks_scores}
        )
    
    def __mul__(self, scalar: float) -> FitnessResult:
        """Multiply fitness by scalar."""
        return FitnessResult(
            fitness=self.fitness * scalar,
            metrics={k: v * scalar for k, v in self.metrics.items()},
            details=self.details,
            tasks_scores={k: v * scalar for k, v in self.tasks_scores.items()}
        )


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""
    
    # Multi-objective weights
    primary_weight: float = 0.6        # Weight for primary fitness
    complexity_penalty: float = 0.1    # Penalty for excessive complexity
    diversity_bonus: float = 0.1       # Bonus for diverse solutions
    efficiency_bonus: float = 0.2       # Bonus for efficient solutions
    
    # Normalization
    normalize_fitness: bool = True    # Normalize fitness to [0, 1]
    fitness_min: float = 0.0           # Minimum fitness value
    fitness_max: float = 1.0           # Maximum fitness value
    
    # Adaptive fitness
    adaptive_scaling: bool = True      # Use adaptive fitness scaling
    elitism_count: int = 2             # Number of elite individuals
    
    # Task-specific
    multi_objective: bool = True        # Use multi-objective evaluation
    task_distribution: Optional[Dict[str, float]] = None  # Task weights


class FitnessEvaluator:
    """
    Evaluates fitness of genomes against defined fitness functions.
    
    Fitness evaluation is the core mechanism for natural selection -
    genomes that solve problems better get higher fitness and
    are more likely to reproduce.
    """
    
    def __init__(
        self,
        config: Optional[FitnessConfig] = None,
        fitness_functions: Optional[List[FitnessFunction]] = None
    ):
        self.config = config or FitnessConfig()
        self.fitness_functions = fitness_functions or []
        self.evaluation_history: List[FitnessResult] = []
        self.best_fitness = float('-inf')
        self.generation_best: List[Tuple[int, float]] = []
    
    def add_fitness_function(self, func: FitnessFunction, weight: float = 1.0) -> None:
        """Add a fitness function with optional weight."""
        self.fitness_functions.append((func, weight))
    
    def evaluate(self, genome: Genome, context: Optional[Dict[str, Any]] = None) -> FitnessResult:
        """
        Evaluate the fitness of a genome.
        
        Args:
            genome: The genome to evaluate
            context: Evaluation context with test cases, inputs, etc.
            
        Returns:
            FitnessResult with detailed evaluation metrics
        """
        if context is None:
            context = {}
        
        # Execute genome to get result
        try:
            result = genome.execute(context)
        except Exception as e:
            # Handle execution errors gracefully
            result = None
        
        # Calculate base fitness from fitness functions
        total_fitness = 0.0
        total_weight = 0.0
        
        for func, weight in self.fitness_functions:
            try:
                func_fitness = func(genome, context)
                total_fitness += func_fitness * weight
                total_weight += weight
            except Exception:
                continue
        
        if total_weight > 0:
            base_fitness = total_fitness / total_weight
        else:
            # Default fitness based on viability
            base_fitness = 1.0 if genome.is_viable else 0.0
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(genome, base_fitness, context)
        
        # Combine into final fitness
        final_fitness = self._combine_fitness(base_fitness, metrics)
        
        # Update best fitness
        if final_fitness > self.best_fitness:
            self.best_fitness = final_fitness
        
        # Create result
        result_obj = FitnessResult(
            fitness=final_fitness,
            metrics=metrics,
            details={
                'genome_id': genome.id,
                'generation': genome.generation,
                'complexity': genome.complexity,
                'num_genes': len(genome.genes),
                'execution_result': str(result)[:100] if result is not None else None
            }
        )
        
        self.evaluation_history.append(result_obj)
        
        # Update genome fitness
        genome.fitness = final_fitness
        
        return result_obj
    
    def _calculate_metrics(
        self,
        genome: Genome,
        base_fitness: float,
        context: Dict[str, Any]
    ) -> Dict[FitnessMetrics, float]:
        """Calculate additional fitness metrics."""
        metrics = {}
        
        # Complexity penalty (prefer simpler solutions)
        complexity = genome.complexity
        optimal_complexity = 10  # Assume optimal around 10 instructions
        complexity_factor = abs(complexity - optimal_complexity) / (optimal_complexity + 1)
        metrics[FitnessMetrics.COMPLEXITY] = 1.0 - min(complexity_factor, 1.0)
        
        # Diversity (based on gene types and instructions)
        gene_types = set(g.gene_type for g in genome.genes)
        metrics[FitnessMetrics.DIVERSITY] = len(gene_types) / len(GeneType)
        
        # Convergence (based on lineage length)
        metrics[FitnessMetrics.CONVERGENCE] = 1.0 / (1.0 + len(genome.lineage) * 0.01)
        
        return metrics
    
    def _combine_fitness(
        self,
        base_fitness: float,
        metrics: Dict[FitnessMetrics, float]
    ) -> float:
        """Combine base fitness with metrics into final score."""
        fitness = base_fitness * self.config.primary_weight
        
        # Add complexity penalty
        complexity_score = metrics.get(FitnessMetrics.COMPLEXITY, 1.0)
        fitness += complexity_score * self.config.complexity_penalty
        
        # Add diversity bonus
        diversity_score = metrics.get(FitnessMetrics.DIVERSITY, 0.0)
        fitness += diversity_score * self.config.diversity_bonus
        
        # Add convergence bonus
        convergence_score = metrics.get(FitnessMetrics.CONVERGENCE, 0.0)
        fitness += convergence_score * 0.1
        
        # Normalize to [0, 1] if enabled
        if self.config.normalize_fitness:
            fitness = max(0.0, min(1.0, fitness))
        
        return fitness
    
    def evaluate_population(
        self,
        population: List[Genome],
        context: Optional[Dict[str, Any]] = None
    ) -> List[FitnessResult]:
        """Evaluate fitness for entire population."""
        return [self.evaluate(genome, context) for genome in population]
    
    def rank_population(self, population: List[Genome]) -> List[Tuple[int, Genome]]:
        """
        Rank population by fitness.
        
        Returns:
            List of (rank, genome) tuples, sorted by fitness descending
        """
        fitnesses = [(i, g, g.fitness) for i, g in enumerate(population)]
        fitnesses.sort(key=lambda x: x[2], reverse=True)
        return [(rank, g) for rank, (_, g, _) in enumerate(fitnesses)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fitness evaluation statistics."""
        if not self.evaluation_history:
            return {'count': 0}
        
        fitnesses = [r.fitness for r in self.evaluation_history]
        
        return {
            'count': len(self.evaluation_history),
            'best_fitness': self.best_fitness,
            'mean_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'std_fitness': self._standard_deviation(fitnesses),
            'recent_trend': self._calculate_trend(fitnesses[-20:])
        }
    
    def _standard_deviation(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = improving)."""
        if len(values) < 2:
            return 0.0
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        return numerator / denominator


# =============================================================================
# Common Fitness Functions
# =============================================================================

def fitness_execution_speed(genome: Genome, context: Dict[str, Any]) -> float:
    """
    Fitness based on execution speed.
    
    Measures how quickly the genome completes its task.
    """
    import time
    
    start = time.time()
    try:
        genome.execute(context)
        elapsed = time.time() - start
        # Faster = higher fitness, with diminishing returns
        return 1.0 / (1.0 + elapsed * 10)
    except Exception:
        return 0.0


def fitness_code_similarity(target: str) -> FitnessFunction:
    """
    Create fitness function measuring similarity to target code.
    
    Args:
        target: Target code string to match
        
    Returns:
        Fitness function
    """
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        code = genome.to_code()
        
        # Simple character-level similarity
        if not target or not code:
            return 0.0
        
        matches = sum(1 for a, b in zip(code, target) if a == b)
        max_len = max(len(code), len(target))
        
        return matches / max_len if max_len > 0 else 0.0
    
    return fitness


def fitness_output_match(expected: Any) -> FitnessFunction:
    """
    Create fitness function for exact output matching.
    
    Args:
        expected: Expected output value
        
    Returns:
        Fitness function
    """
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        try:
            result = genome.execute(context)
            
            if result == expected:
                return 1.0
            elif isinstance(result, (int, float)) and isinstance(expected, (int, float)):
                # Allow close matches for numeric outputs
                diff = abs(result - expected)
                return max(0.0, 1.0 - diff / (abs(expected) + 1))
            else:
                return 0.0
        except Exception:
            return 0.0
    
    return fitness


def fitness_symbolic_regression(x_data: List[float], y_data: List[float]) -> FitnessFunction:
    """
    Create fitness function for symbolic regression problems.
    
    Measures how well the evolved function matches target data.
    
    Args:
        x_data: Input data points
        y_data: Expected output data points
        
    Returns:
        Fitness function
    """
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        if len(x_data) != len(y_data):
            return 0.0
        
        total_error = 0.0
        
        for x, y_expected in zip(x_data, y_data):
            try:
                # Set up context with input
                ctx = {'x': x, **context}
                
                # Execute and get result
                y_actual = genome.execute(ctx)
                
                # Calculate squared error
                if y_actual is not None:
                    error = (y_actual - y_expected) ** 2
                    total_error += error
                else:
                    total_error += y_expected ** 2  # Max error if no output
                    
            except Exception:
                total_error += y_expected ** 2
        
        # Convert error to fitness (lower error = higher fitness)
        mean_error = total_error / len(x_data)
        return 1.0 / (1.0 + mean_error)
    
    return fitness


def fitness_classification(patterns: List[Tuple[Any, int]], num_classes: int) -> FitnessFunction:
    """
    Create fitness function for classification problems.
    
    Args:
        patterns: List of (input, correct_class) tuples
        num_classes: Number of possible classes
        
    Returns:
        Fitness function
    """
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        if not patterns:
            return 0.0
        
        correct = 0
        
        for pattern, correct_class in patterns:
            try:
                ctx = {'input': pattern, **context}
                result = genome.execute(ctx)
                
                # Interpret result as class (modulo number of classes)
                predicted_class = int(result) % num_classes if result is not None else -1
                
                if predicted_class == correct_class:
                    correct += 1
                    
            except Exception:
                continue
        
        return correct / len(patterns)
    
    return fitness


def fitness_knapsack(
    items: List[Tuple[float, float]],
    capacity: float
) -> FitnessFunction:
    """
    Create fitness function for knapsack problem.
    
    Args:
        items: List of (weight, value) tuples
        capacity: Maximum weight capacity
        
    Returns:
        Fitness function
    """
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        total_weight = 0.0
        total_value = 0.0
        
        # Assume genome output is a list of 0/1 indicating item selection
        try:
            ctx = {**context}
            selections = genome.execute(ctx)
            
            if selections is None:
                return 0.0
            
            for i, selected in enumerate(selections):
                if i < len(items) and selected:
                    weight, value = items[i]
                    
                    if total_weight + weight <= capacity:
                        total_weight += weight
                        total_value += value
                    else:
                        # Penalize overweight solutions
                        return total_value * 0.5
            
            # Higher value = higher fitness
            return total_value
            
        except Exception:
            return 0.0
    
    return fitness


def fitness_traveling_salesman(distances: List[List[float]]) -> FitnessFunction:
    """
    Create fitness function for Traveling Salesman Problem.
    
    Args:
        distances: 2D matrix of distances between cities
        
    Returns:
        Fitness function
    """
    n = len(distances)
    
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        try:
            ctx = {**context}
            tour = genome.execute(ctx)
            
            if tour is None or len(tour) != n:
                return 0.0
            
            # Calculate total distance
            total_distance = 0.0
            for i in range(n):
                from_city = int(tour[i]) % n
                to_city = int(tour[(i + 1) % n]) % n
                total_distance += distances[from_city][to_city]
            
            # Lower distance = higher fitness
            return 1.0 / (1.0 + total_distance / n)
            
        except Exception:
            return 0.0
    
    return fitness


def fitness_multi_objective(
    objectives: List[FitnessFunction],
    weights: Optional[List[float]] = None
) -> FitnessFunction:
    """
    Create multi-objective fitness function.
    
    Args:
        objectives: List of fitness functions
        weights: Optional weights for each objective
        
    Returns:
        Combined fitness function
    """
    if weights is None:
        weights = [1.0] * len(objectives)
    
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        total = 0.0
        total_weight = 0.0
        
        for obj, weight in zip(objectives, weights):
            try:
                score = obj(genome, context)
                total += score * weight
                total_weight += weight
            except Exception:
                continue
        
        return total / total_weight if total_weight > 0 else 0.0
    
    return fitness


def fitness_pareto_dominated(
    objectives: List[FitnessFunction]
) -> FitnessFunction:
    """
    Create Pareto-based fitness for multi-objective optimization.
    
    Measures how many solutions a genome dominates in objective space.
    
    Args:
        objectives: List of fitness functions (each to be maximized)
        
    Returns:
        Pareto fitness function
    """
    # Store population for comparison
    population_cache: List[Genome] = []
    
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        # Calculate objective scores for this genome
        scores = []
        for obj in objectives:
            try:
                scores.append(obj(genome, context))
            except Exception:
                scores.append(0.0)
        
        # Count how many individuals this genome dominates
        domination_count = 0
        
        for other in population_cache:
            other_scores = []
            for obj in objectives:
                try:
                    other_scores.append(obj(other, context))
                except Exception:
                    other_scores.append(0.0)
            
            # Check if this genome dominates other
            at_least_as_good = all(s >= o for s, o in zip(scores, other_scores))
            strictly_better = any(s > o for s, o in zip(scores, other_scores))
            
            if at_least_as_good and strictly_better:
                domination_count += 1
        
        # Add to cache
        population_cache.append(genome)
        
        # Keep cache manageable
        if len(population_cache) > 1000:
            population_cache.pop(0)
        
        return domination_count
    
    return fitness


def fitness_parsimony(genome: Genome, context: Dict[str, Any]) -> float:
    """
    Fitness based on parsimony (prefer simpler solutions).
    
    Uses size as a tiebreaker - smaller genomes preferred.
    """
    # Base fitness on execution result if available
    try:
        result = genome.execute(context)
        base_fitness = 1.0 if result is not None else 0.0
    except Exception:
        base_fitness = 0.0
    
    # Penalty for complexity
    complexity_penalty = genome.complexity * 0.01
    
    return base_fitness - complexity_penalty


def fitness_diversity_bonus(population: List[Genome]) -> FitnessFunction:
    """
    Create fitness function with diversity bonus.
    
    Genomes that are more different from others get a fitness bonus.
    
    Args:
        population: Current population for diversity comparison
        
    Returns:
        Diversity-aware fitness function
    """
    def fitness(genome: Genome, context: Dict[str, Any]) -> float:
        # Base fitness
        try:
            base = 1.0 if genome.is_viable else 0.0
        except Exception:
            base = 0.0
        
        # Calculate diversity bonus
        if not population:
            return base
        
        diversity = 0.0
        for other in population:
            if other.id != genome.id:
                # Measure genetic distance
                distance = _genetic_distance(genome, other)
                diversity += distance
        
        avg_diversity = diversity / len(population)
        
        return base + avg_diversity * 0.2
    
    return fitness


def _genetic_distance(genome1: Genome, genome2: Genome) -> float:
    """Calculate genetic distance between two genomes."""
    # Simple Hamming distance on gene types
    types1 = [g.gene_type for g in genome1.genes]
    types2 = [g.gene_type for g in genome2.genes]
    
    if not types1 or not types2:
        return 1.0
    
    common = sum(1 for t in types1 if t in types2)
    return 1.0 - common / max(len(types1), len(types2))
