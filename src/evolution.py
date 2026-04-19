"""
Evolution Engine - Core Orchestration

This module provides the main EvolutionEngine class that orchestrates
all components of the genetic algorithm - selection, mutation, crossover,
and fitness evaluation.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .crossover import (
    CrossoverConfig,
    CrossoverOperator,
    CrossoverPipeline,
    SinglePointCrossover,
)
from .environment import Environment, TaskDistribution, create_task_suite
from .fitness import FitnessConfig, FitnessEvaluator, FitnessResult
from .genome import Genome
from .mutation import (
    MutationConfig,
    MutationOperator,
    MutationPipeline,
    PointMutation,
    InsertionMutation,
    DeletionMutation,
)
from .population import Population, PopulationStatistics
from .selection import (
    SelectionConfig,
    SelectionMethod,
    SelectionOperator,
    AdaptiveSelection,
    TournamentSelection,
    RouletteWheelSelection,
)


@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine."""
    
    # Population settings
    population_size: int = 100
    min_population_size: int = 20
    max_population_size: int = 200
    
    # Evolution operators
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_count: int = 2
    
    # Selection
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    
    # Termination
    max_generations: int = 500
    target_fitness: float = 0.99
    early_stopping_patience: int = 50
    
    # Diversity
    min_diversity: float = 0.1
    diversity_boost_threshold: float = 0.2
    
    # Output
    verbose: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 50
    
    # Parallelism
    use_multiprocessing: bool = False
    num_workers: int = 4
    
    # Seed
    random_seed: Optional[int] = None


@dataclass
class EvolutionStatistics:
    """Statistics about the evolution process."""
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    total_time: float = 0.0
    
    # Generations
    generations_completed: int = 0
    max_generations: int = 0
    
    # Fitness
    initial_best_fitness: float = 0.0
    final_best_fitness: float = 0.0
    best_fitness_ever: float = 0.0
    fitness_improvement: float = 0.0
    
    # Diversity
    initial_diversity: float = 0.0
    final_diversity: float = 0.0
    
    # Convergence
    converged: bool = False
    convergence_generation: Optional[int] = None
    
    # Best genome
    best_genome_id: str = ""
    
    # Population history
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    complexity_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_time_seconds': self.total_time,
            'generations_completed': self.generations_completed,
            'initial_best_fitness': self.initial_best_fitness,
            'final_best_fitness': self.final_best_fitness,
            'best_fitness_ever': self.best_fitness_ever,
            'fitness_improvement': self.fitness_improvement,
            'initial_diversity': self.initial_diversity,
            'final_diversity': self.final_diversity,
            'converged': self.converged,
            'convergence_generation': self.convergence_generation,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history
        }


class EvolutionEngine:
    """
    Main evolution engine that orchestrates the genetic algorithm.
    
    The engine manages:
    - Population initialization and maintenance
    - Fitness evaluation
    - Selection of parents
    - Crossover to produce offspring
    - Mutation of offspring
    - Environmental pressure
    - Statistics tracking and logging
    """
    
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        environment: Optional[Environment] = None
    ):
        """
        Initialize evolution engine.
        
        Args:
            config: Evolution configuration
            fitness_evaluator: Fitness evaluator (creates default if None)
            environment: Environment with tasks (creates default if None)
        """
        self.config = config or EvolutionConfig()
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        
        # Initialize components
        self.fitness_evaluator = fitness_evaluator or self._create_default_fitness_evaluator()
        self.environment = environment or self._create_default_environment()
        
        # Initialize operators
        self.selection = self._create_selection_operator()
        self.crossover = self._create_crossover_operator()
        self.mutation = self._create_mutation_pipeline()
        
        # Initialize population
        self.population = Population(
            size=self.config.population_size,
            min_size=self.config.min_population_size,
            max_size=self.config.max_population_size
        )
        
        # Statistics
        self.statistics = EvolutionStatistics()
        self.best_genome: Optional[Genome] = None
        
        # Callbacks
        self.callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._paused = False
    
    def _create_default_fitness_evaluator(self) -> FitnessEvaluator:
        """Create default fitness evaluator."""
        config = FitnessConfig(
            primary_weight=0.7,
            complexity_penalty=0.1,
            diversity_bonus=0.1,
            efficiency_bonus=0.1,
            normalize_fitness=True
        )
        return FitnessEvaluator(config)
    
    def _create_default_environment(self) -> Environment:
        """Create default environment."""
        task_dist = create_task_suite(num_tasks=5, adaptive=True)
        return Environment(
            name="default",
            task_distribution=task_dist
        )
    
    def _create_selection_operator(self) -> SelectionOperator:
        """Create selection operator based on config."""
        config = SelectionConfig(
            population_size=self.config.population_size,
            elite_count=self.config.elite_count,
            tournament_size=self.config.tournament_size
        )
        
        if self.config.selection_method == SelectionMethod.ADAPTIVE:
            return AdaptiveSelection(config)
        elif self.config.selection_method == SelectionMethod.TOURNAMENT:
            return TournamentSelection(config)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return RouletteWheelSelection(config)
        else:
            return TournamentSelection(config)
    
    def _create_crossover_operator(self) -> CrossoverOperator:
        """Create crossover operator."""
        config = CrossoverConfig(
            crossover_rate=self.config.crossover_rate
        )
        return SinglePointCrossover(config)
    
    def _create_mutation_pipeline(self) -> MutationPipeline:
        """Create mutation pipeline."""
        config = MutationConfig(
            point_rate=self.config.mutation_rate,
            insertion_rate=self.config.mutation_rate * 0.3,
            deletion_rate=self.config.mutation_rate * 0.2,
            structural_rate=self.config.mutation_rate * 0.1
        )
        return MutationPipeline(config)
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def initialize(self, mode: str = "random") -> None:
        """
        Initialize the population.
        
        Args:
            mode: Initialization mode ('random', 'minimal', 'adaptive')
        """
        if mode == "random":
            self.population.initialize_random(
                genes_per_genome=5,
                complexity_per_gene=3
            )
        elif mode == "minimal":
            self.population.initialize_minimal()
        elif mode == "adaptive":
            self.population.initialize_adaptive(task_complexity=0.5)
        else:
            raise ValueError(f"Unknown initialization mode: {mode}")
        
        # Reset statistics
        self.statistics = EvolutionStatistics()
        self.statistics.start_time = time.time()
        self.statistics.max_generations = self.config.max_generations
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback function called after each generation."""
        self.callbacks.append(callback)
    
    # =========================================================================
    # Evolution Loop
    # =========================================================================
    
    def evolve(self, max_generations: Optional[int] = None) -> Genome:
        """
        Run the evolution process.
        
        Args:
            max_generations: Override max generations for this run
            
        Returns:
            Best genome found
        """
        if max_generations is not None:
            original_max = self.config.max_generations
            self.config.max_generations = max_generations
        
        self._running = True
        
        try:
            # Initialize if needed
            if not self.population.individuals:
                self.initialize()
            
            # Evaluate initial population
            self._evaluate_population()
            
            # Track initial statistics
            stats = self.population.get_statistics()
            self.statistics.initial_best_fitness = stats.best_fitness
            self.statistics.initial_diversity = stats.diversity
            self.statistics.fitness_history.append(stats.best_fitness)
            self.statistics.diversity_history.append(stats.diversity)
            
            # Main evolution loop
            generations_without_improvement = 0
            
            while self._running:
                # Check termination conditions
                if self._check_termination():
                    break
                
                # Perform one generation
                self._evolve_step()
                
                # Check for improvement
                current_best = self.population.get_best(1)[0]
                if current_best.fitness > self.statistics.best_fitness_ever:
                    self.statistics.best_fitness_ever = current_best.fitness
                    self.best_genome = current_best.copy()
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Update diversity
                self.population.maintain_diversity()
                
                # Log progress
                if self.config.verbose and self.population.generation % self.config.log_interval == 0:
                    self._log_progress()
                
                # Run callbacks
                for callback in self.callbacks:
                    callback(self)
        
        finally:
            self._running = False
            self.statistics.end_time = time.time()
            self.statistics.total_time = self.statistics.end_time - self.statistics.start_time
        
        # Finalize
        if max_generations is not None:
            self.config.max_generations = original_max
        
        self._finalize()
        
        return self.best_genome or self.population.get_best(1)[0]
    
    def _evolve_step(self) -> None:
        """Perform one generation of evolution."""
        generation = self.population.generation
        
        # Select parents
        parents = self.selection.select(self.population.individuals)
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover.crossover(parents[i], parents[i + 1])
                offspring.extend([child1, child2])
        
        # Mutate offspring
        offspring = [self.mutation.mutate(child) for child in offspring]
        
        # Evaluate offspring
        for child in offspring:
            fitness = self.environment.evaluate(child)
            child.fitness = fitness
        
        # Environmental selection - replace worst individuals
        self.population.replace_worst(offspring)
        
        # Evaluate entire population
        self._evaluate_population()
        
        # Advance generation
        self.population.next_generation()
        self.environment.step()
        
        # Update statistics
        stats = self.population.get_statistics()
        self.statistics.generations_completed = generation + 1
        self.statistics.final_best_fitness = stats.best_fitness
        self.statistics.fitness_history.append(stats.best_fitness)
        self.statistics.diversity_history.append(stats.diversity)
        self.statistics.complexity_history.append(stats.mean_complexity)
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for entire population."""
        for genome in self.population.individuals:
            fitness = self.environment.evaluate(genome)
            genome.fitness = fitness
    
    def _check_termination(self) -> bool:
        """Check if evolution should terminate."""
        # Check max generations
        if self.population.generation >= self.config.max_generations:
            return True
        
        # Check target fitness
        best = self.population.get_best(1)[0] if self.population.individuals else None
        if best and best.fitness >= self.config.target_fitness:
            self.statistics.converged = True
            self.statistics.convergence_generation = self.population.generation
            return True
        
        # Check environment
        if not self.environment.should_evolve():
            return True
        
        return False
    
    def _finalize(self) -> None:
        """Finalize evolution statistics."""
        stats = self.population.get_statistics()
        
        self.statistics.final_best_fitness = stats.best_fitness
        self.statistics.final_diversity = stats.diversity
        
        if self.best_genome:
            self.statistics.best_genome_id = self.best_genome.id
        
        self.statistics.fitness_improvement = (
            self.statistics.final_best_fitness - self.statistics.initial_best_fitness
        )
    
    def _log_progress(self) -> None:
        """Log evolution progress."""
        stats = self.population.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"Generation {self.population.generation}/{self.config.max_generations}")
        print(f"{'='*60}")
        print(f"  Best Fitness:  {stats.best_fitness:.4f}")
        print(f"  Mean Fitness:  {stats.mean_fitness:.4f}")
        print(f"  Diversity:     {stats.diversity:.4f}")
        print(f"  Complexity:    {stats.mean_complexity:.1f}")
        print(f"  Environment:   {self.environment.name}")
        print(f"  Best Genome:   {stats.best_genome_id[:16]}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # Control Methods
    # =========================================================================
    
    def pause(self) -> None:
        """Pause evolution."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume evolution."""
        self._paused = False
    
    def stop(self) -> None:
        """Stop evolution."""
        self._running = False
    
    def step(self) -> None:
        """Execute one generation step."""
        if not self.population.individuals:
            self.initialize()
            self._evaluate_population()
        else:
            self._evolve_step()
    
    # =========================================================================
    # Analysis and Export
    # =========================================================================
    
    def get_statistics(self) -> EvolutionStatistics:
        """Get evolution statistics."""
        return self.statistics
    
    def get_best_genome(self) -> Optional[Genome]:
        """Get the best genome found."""
        return self.best_genome
    
    def get_population_snapshot(self) -> List[Dict[str, Any]]:
        """Get snapshot of current population."""
        return self.population.to_list()
    
    def export_best_genome(self, filepath: str) -> None:
        """Export the best genome as executable code."""
        if not self.best_genome:
            return
        
        code = self.best_genome.to_code()
        with open(filepath, 'w') as f:
            f.write(code)
    
    def export_statistics(self, filepath: str) -> None:
        """Export statistics to JSON file."""
        import json
        
        stats_dict = self.statistics.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    def visualize_evolution(self) -> Dict[str, Any]:
        """
        Get data for visualization of evolution.
        
        Returns:
            Dictionary with visualization data
        """
        return {
            'fitness_history': self.statistics.fitness_history,
            'diversity_history': self.statistics.diversity_history,
            'complexity_history': self.statistics.complexity_history,
            'generations': len(self.statistics.fitness_history),
            'best_fitness': self.statistics.best_fitness_ever,
            'converged': self.statistics.converged
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def reset(self) -> None:
        """Reset the engine to initial state."""
        self.population.clear()
        self.statistics = EvolutionStatistics()
        self.best_genome = None
        self._running = False
        self._paused = False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state for serialization."""
        return {
            'generation': self.population.generation,
            'population_size': len(self.population),
            'best_fitness': self.population.get_best(1)[0].fitness if self.population else 0.0,
            'running': self._running,
            'paused': self._paused
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def evolve_simple(
    fitness_function: Callable[[Genome, Dict[str, Any]], float],
    max_generations: int = 100,
    population_size: int = 50
) -> Tuple[Genome, EvolutionStatistics]:
    """
    Simple evolution with a custom fitness function.
    
    Args:
        fitness_function: Function to evaluate genome fitness
        max_generations: Maximum generations to evolve
        population_size: Population size
        
    Returns:
        Tuple of (best_genome, statistics)
    """
    # Create engine
    config = EvolutionConfig(
        population_size=population_size,
        max_generations=max_generations,
        verbose=False
    )
    
    engine = EvolutionEngine(config)
    
    # Set up fitness evaluator
    engine.fitness_evaluator.add_fitness_function(fitness_function)
    
    # Run evolution
    best = engine.evolve()
    
    return best, engine.statistics


def evolve_to_target(
    fitness_function: Callable[[Genome, Dict[str, Any]], float],
    target_fitness: float = 0.99,
    max_generations: int = 1000,
    population_size: int = 100
) -> Tuple[Optional[Genome], EvolutionStatistics]:
    """
    Evolve until target fitness is reached or max generations.
    
    Args:
        fitness_function: Function to evaluate genome fitness
        target_fitness: Target fitness to reach
        max_generations: Maximum generations
        population_size: Population size
        
    Returns:
        Tuple of (best_genome if target reached, statistics)
    """
    config = EvolutionConfig(
        population_size=population_size,
        max_generations=max_generations,
        target_fitness=target_fitness,
        verbose=True
    )
    
    engine = EvolutionEngine(config)
    engine.fitness_evaluator.add_fitness_function(fitness_function)
    
    best = engine.evolve()
    
    if best.fitness >= target_fitness:
        return best, engine.statistics
    else:
        return None, engine.statistics
