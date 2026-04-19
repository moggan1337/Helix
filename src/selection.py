"""
Selection Module - Natural Selection Pressure

This module implements various selection mechanisms that apply
selective pressure to the population, favoring genomes with
higher fitness for reproduction.
"""

from __future__ import annotations

import copy
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .genome import Genome


class SelectionMethod(Enum):
    """Types of selection operators."""
    
    # Fitness-proportional selection
    ROULETTE = auto()           # Proportional to fitness
    SUS = auto()                # Stochastic universal sampling
    
    # Order-based selection
    RANK = auto()               # Based on rank, not fitness
    LINEAR_RANK = auto()        # Linear rank selection
    EXPONENTIAL_RANK = auto()   # Exponential rank selection
    
    # Tournament selection
    TOURNAMENT = auto()         # Binary or multi-way tournament
    DETERMINISTIC_TOURNAMENT = auto()
    PROBABILISTIC_TOURNAMENT = auto()
    
    # Truncation selection
    TRUNCATION = auto()         # Select top N%
    
    # Multi-objective selection
    NSGA2 = auto()              # Non-dominated sorting
    SPEA2 = auto()              # Strength Pareto
    
    # Special selection
    BOLTZMANN = auto()          # Temperature-based
    ADAPTIVE = auto()           # Adaptive selection pressure


@dataclass
class SelectionPressure:
    """
    Manages selection pressure in the population.
    
    Selection pressure determines how strongly fitness influences
    reproduction. High pressure accelerates convergence but risks
    premature convergence. Low pressure maintains diversity but
    may slow evolution.
    """
    
    base_pressure: float = 1.5          # Base selection intensity
    min_pressure: float = 1.0            # Minimum pressure
    max_pressure: float = 3.0            # Maximum pressure
    
    pressure_adaptation: bool = True     # Adapt pressure dynamically
    stagnation_threshold: int = 10       # Generations without improvement
    diversity_threshold: float = 0.1     # Minimum diversity
    
    current_pressure: float = 1.5        # Current selection pressure
    stagnation_count: int = 0            # Generations since improvement
    
    history: List[float] = field(default_factory=list)  # Pressure history
    
    def increase_pressure(self) -> None:
        """Increase selection pressure."""
        self.current_pressure = min(
            self.current_pressure * 1.1,
            self.max_pressure
        )
        self.history.append(self.current_pressure)
    
    def decrease_pressure(self) -> None:
        """Decrease selection pressure."""
        self.current_pressure = max(
            self.current_pressure * 0.9,
            self.min_pressure
        )
        self.history.append(self.current_pressure)
    
    def adapt(self, fitness_history: List[float], diversity: float) -> float:
        """
        Adapt selection pressure based on population state.
        
        Args:
            fitness_history: Recent fitness values
            diversity: Current population diversity
            
        Returns:
            Adapted selection pressure
        """
        if not self.pressure_adaptation:
            return self.current_pressure
        
        # Check for stagnation
        if len(fitness_history) >= 2:
            recent_improvement = fitness_history[-1] - fitness_history[-2]
            
            if recent_improvement < 0.001:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        
        # Adapt based on conditions
        if self.stagnation_count > self.stagnation_threshold:
            # Stagnating - decrease pressure to explore more
            self.decrease_pressure()
        elif diversity < self.diversity_threshold:
            # Low diversity - decrease pressure to maintain variation
            self.decrease_pressure()
        elif self.stagnation_count == 0 and diversity > 2 * self.diversity_threshold:
            # Good progress and high diversity - can increase pressure
            self.increase_pressure()
        
        return self.current_pressure
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selection pressure statistics."""
        return {
            'current': self.current_pressure,
            'base': self.base_pressure,
            'min': self.min_pressure,
            'max': self.max_pressure,
            'stagnation_count': self.stagnation_count,
            'history_length': len(self.history)
        }


@dataclass
class SelectionConfig:
    """Configuration for selection operators."""
    
    # Population parameters
    population_size: int = 100
    elite_count: int = 2           # Number of elite individuals to preserve
    
    # Tournament parameters
    tournament_size: int = 3
    tournament_probability: float = 0.9  # For probabilistic tournament
    
    # Rank parameters
    rank_base: float = 1.5         # Selection pressure for rank selection
    
    # Truncation parameters
    truncation_ratio: float = 0.1  # Select top 10%
    
    # Boltzmann parameters
    temperature: float = 1.0
    cooling_rate: float = 0.99
    
    # Adaptive parameters
    adaptive: bool = True


class SelectionOperator(ABC):
    """
    Base class for selection operators.
    
    Selection operators determine which individuals are chosen
    for reproduction based on their fitness.
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self.pressure = SelectionPressure()
    
    @abstractmethod
    def select(self, population: List[Genome]) -> List[Genome]:
        """
        Select individuals for reproduction.
        
        Args:
            population: Current population of genomes
            
        Returns:
            Selected individuals (may include duplicates)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this selection operator."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this selection operator."""
        pass


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette wheel (fitness-proportional) selection.
    
    Each individual has a probability of selection proportional
    to its fitness. Higher fitness = higher chance of selection.
    """
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Select using roulette wheel."""
        if not population:
            return []
        
        # Calculate fitness sums
        fitnesses = [max(0.0, g.fitness) for g in population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            # All fitnesses are zero, select uniformly
            return [random.choice(population) for _ in range(len(population))]
        
        # Normalize to probabilities
        probabilities = [f / total_fitness for f in fitnesses]
        
        # Select individuals
        selected = []
        for _ in range(len(population)):
            r = random.random()
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(copy.deepcopy(population[i]))
                    break
            else:
                # Fallback to last individual
                selected.append(copy.deepcopy(population[-1]))
        
        return selected
    
    def get_name(self) -> str:
        return "RouletteWheelSelection"
    
    def get_description(self) -> str:
        return "Fitness-proportional selection (roulette wheel)"


class StochasticUniversalSampling(SelectionOperator):
    """
    Stochastic Universal Sampling (SUS).
    
    A deterministic version of roulette wheel selection that
    guarantees better coverage of the fitness range.
    """
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Select using SUS."""
        if not population:
            return []
        
        fitnesses = [max(0.0, g.fitness) for g in population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            return [random.choice(population).copy() for _ in range(len(population))]
        
        # Calculate pointer spacing
        n = len(population)
        spacing = total_fitness / n
        
        # Random starting point
        start = random.uniform(0, spacing)
        
        # Generate pointers
        pointers = [start + i * spacing for i in range(n)]
        
        # Select individuals
        selected = []
        cumulative = 0.0
        fitness_idx = 0
        
        for pointer in pointers:
            while cumulative <= pointer and fitness_idx < n:
                cumulative += fitnesses[fitness_idx]
                fitness_idx += 1
            
            # Select the individual before the pointer exceeded
            selected.append(copy.deepcopy(population[min(fitness_idx - 1, n - 1)]))
        
        return selected
    
    def get_name(self) -> str:
        return "StochasticUniversalSampling"
    
    def get_description(self) -> str:
        return "Deterministic fitness-proportional with uniform spacing"


class RankSelection(SelectionOperator):
    """
    Rank-based selection.
    
    Selection probability is based on rank rather than raw fitness,
    which can help maintain diversity and prevent premature convergence.
    """
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Select using rank-based selection."""
        if not population:
            return []
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda g: g.fitness)
        n = len(sorted_pop)
        
        # Assign ranks (0 = worst, n-1 = best)
        # Higher rank = higher selection probability
        ranks = list(range(n))
        
        # Calculate selection probabilities based on rank
        # Using linear ranking with pressure parameter
        pressure = self.pressure.current_pressure
        probabilities = [
            (2 - pressure + 2 * (pressure - 1) * r / (n - 1)) / n
            for r in ranks
        ]
        
        # Select individuals
        selected = []
        for _ in range(n):
            r = random.random()
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(copy.deepcopy(sorted_pop[i]))
                    break
            else:
                selected.append(copy.deepcopy(sorted_pop[-1]))
        
        return selected
    
    def get_name(self) -> str:
        return "RankSelection"
    
    def get_description(self) -> str:
        return "Selection based on fitness rank"


class TournamentSelection(SelectionOperator):
    """
    Tournament selection.
    
    Randomly selects k individuals and picks the best (or
    probabilistically selects the best based on fitness).
    """
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Select using tournament selection."""
        if not population:
            return []
        
        n = len(population)
        k = self.config.tournament_size
        p = self.config.tournament_probability
        
        selected = []
        
        for _ in range(n):
            # Randomly select tournament participants
            participants = random.sample(population, min(k, n))
            
            # Sort by fitness
            participants.sort(key=lambda g: g.fitness, reverse=True)
            
            # Probabilistic selection
            if random.random() < p:
                # Select best
                selected.append(copy.deepcopy(participants[0]))
            else:
                # Select randomly from tournament
                selected.append(random.choice(participants).copy())
        
        return selected
    
    def get_name(self) -> str:
        return "TournamentSelection"
    
    def get_description(self) -> str:
        return f"Tournament selection (k={self.config.tournament_size})"


class TruncationSelection(SelectionOperator):
    """
    Truncation selection.
    
    Selects the top N% of individuals uniformly.
    Simple and efficient for large populations.
    """
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Select using truncation."""
        if not population:
            return []
        
        n = len(population)
        ratio = self.config.truncation_ratio
        keep_count = max(1, int(n * ratio))
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        
        # Select from top individuals
        selected = []
        for _ in range(n):
            selected.append(random.choice(sorted_pop[:keep_count]).copy())
        
        return selected
    
    def get_name(self) -> str:
        return "TruncationSelection"
    
    def get_description(self) -> str:
        return f"Select top {self.config.truncation_ratio*100}%"


class BoltzmannSelection(SelectionOperator):
    """
    Boltzmann selection.
    
    Uses a temperature parameter that decreases over time,
    starting with high temperature (near-uniform selection)
    and ending with low temperature (strong selection pressure).
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        super().__init__(config)
        self.temperature = self.config.temperature
        self.cooling_rate = self.config.cooling_rate
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Select using Boltzmann selection."""
        if not population:
            return []
        
        fitnesses = [g.fitness for g in population]
        
        # Calculate Boltzmann probabilities
        exp_fitnesses = [
            math.exp(f / max(0.001, self.temperature))
            for f in fitnesses
        ]
        total = sum(exp_fitnesses)
        
        if total == 0:
            return [random.choice(population).copy() for _ in range(len(population))]
        
        probabilities = [ef / total for ef in exp_fitnesses]
        
        # Select individuals
        n = len(population)
        selected = []
        
        for _ in range(n):
            r = random.random()
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(copy.deepcopy(population[i]))
                    break
            else:
                selected.append(copy.deepcopy(population[-1]))
        
        # Cool down
        self.temperature *= self.cooling_rate
        self.temperature = max(0.01, self.temperature)
        
        return selected
    
    def get_name(self) -> str:
        return "BoltzmannSelection"
    
    def get_description(self) -> str:
        return f"Temperature-based selection (T={self.temperature:.4f})"


class ElitePreservation(SelectionOperator):
    """
    Elite preservation - always keeps the best individuals.
    
    Prevents loss of the best solutions found so far.
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        super().__init__(config)
        self.elites: List[Genome] = []
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Preserve elites and select rest using another operator."""
        if not population:
            return []
        
        n = len(population)
        elite_count = self.config.elite_count
        
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        
        # Preserve elites
        self.elites = [g.copy() for g in sorted_pop[:elite_count]]
        
        # Select remaining from rest of population
        remaining = n - elite_count
        if remaining > 0:
            non_elites = sorted_pop[elite_count:]
            selected = [random.choice(non_elites).copy() for _ in range(remaining)]
            selected.extend(self.elites)
            return selected
        
        return self.elites
    
    def get_name(self) -> str:
        return "ElitePreservation"
    
    def get_description(self) -> str:
        return f"Preserves top {self.config.elite_count} individuals"
    
    def get_elites(self) -> List[Genome]:
        """Get current elite individuals."""
        return self.elites


class AdaptiveSelection(SelectionOperator):
    """
    Adaptive selection - automatically adjusts selection method.
    
    Switches between different selection methods based on
    population state (diversity, convergence, etc.).
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        super().__init__(config)
        
        # Create sub-selectors
        self.roulette = RouletteWheelSelection(config)
        self.tournament = TournamentSelection(config)
        self.rank = RankSelection(config)
        self.truncation = TruncationSelection(config)
        
        self.fitness_history: List[float] = []
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Adaptively select individuals."""
        if not population:
            return []
        
        # Track fitness history
        best_fitness = max(g.fitness for g in population)
        self.fitness_history.append(best_fitness)
        
        if len(self.fitness_history) > 20:
            self.fitness_history.pop(0)
        
        # Calculate diversity
        diversity = self._calculate_diversity(population)
        
        # Adapt selection pressure
        self.pressure.adapt(self.fitness_history, diversity)
        
        # Choose selection method based on state
        if self.pressure.stagnation_count > self.pressure.stagnation_threshold:
            # Stagnation - use diversity-promoting selection
            return self.rank.select(population)
        elif diversity < self.pressure.diversity_threshold:
            # Low diversity - use tournament (stronger selection)
            return self.tournament.select(population)
        elif best_fitness < 0.5:
            # Early evolution - use roulette (exploration)
            return self.roulette.select(population)
        else:
            # Late evolution - use tournament (exploitation)
            return self.tournament.select(population)
    
    def _calculate_diversity(self, population: List[Genome]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity: number of unique fitness values / population size
        unique_fitnesses = len(set(g.fitness for g in population))
        return unique_fitnesses / len(population)
    
    def get_name(self) -> str:
        return "AdaptiveSelection"
    
    def get_description(self) -> str:
        return "Automatically adapts selection method based on population state"


class SelectionPipeline:
    """
    Pipeline of selection operators.
    
    Allows combining multiple selection methods in sequence.
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self.operators: List[SelectionOperator] = []
        self.current_index = 0
    
    def add_operator(self, operator: SelectionOperator) -> None:
        """Add an operator to the pipeline."""
        self.operators.append(operator)
    
    def select(self, population: List[Genome]) -> List[Genome]:
        """Apply selection pipeline."""
        if not self.operators:
            return population
        
        result = population
        for operator in self.operators:
            result = operator.select(result)
        
        return result
    
    def select_parents(
        self,
        population: List[Genome],
        num_parents: int = 2
    ) -> List[List[Genome]]:
        """
        Select parent pairs for crossover.
        
        Args:
            population: Current population
            num_parents: Number of parents per offspring (usually 2)
            
        Returns:
            List of parent tuples
        """
        if not population:
            return []
        
        selected = self.select(population)
        parents = []
        
        for _ in range(len(population) // 2):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            parents.append([parent1, parent2])
        
        return parents


def calculate_selection_intensity(
    population: List[Genome],
    selection_fraction: float = 0.5
) -> float:
    """
    Calculate selection intensity for a population.
    
    Selection intensity measures the strength of selection pressure.
    
    Args:
        population: Current population
        selection_fraction: Fraction selected for reproduction
        
    Returns:
        Selection intensity
    """
    if not population or len(population) < 2:
        return 0.0
    
    # Sort by fitness
    sorted_pop = sorted(population, key=lambda g: g.fitness)
    n = len(sorted_pop)
    k = max(1, int(n * selection_fraction))
    
    # Calculate mean fitness of selected vs whole population
    mean_all = sum(g.fitness for g in sorted_pop) / n
    mean_selected = sum(g.fitness for g in sorted_pop[-k:]) / k
    
    # Calculate standard deviation
    variance = sum((g.fitness - mean_all) ** 2 for g in sorted_pop) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    
    if std == 0:
        return 0.0
    
    return (mean_selected - mean_all) / std


def calculate_diversity_metrics(population: List[Genome]) -> Dict[str, float]:
    """
    Calculate diversity metrics for a population.
    
    Args:
        population: Population to analyze
        
    Returns:
        Dictionary of diversity metrics
    """
    if not population:
        return {'fitness_variance': 0.0, 'genetic_diversity': 0.0}
    
    # Fitness variance
    fitnesses = [g.fitness for g in population]
    mean_fitness = sum(fitnesses) / len(fitnesses)
    fitness_variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / len(fitnesses)
    
    # Genetic diversity (unique genomes / total)
    unique_genomes = len(set(g.id for g in population))
    genetic_diversity = unique_genomes / len(population)
    
    # Gene type diversity
    all_gene_types = set()
    for g in population:
        for gene in g.genes:
            all_gene_types.add(gene.gene_type)
    
    max_types = len(GeneType)
    type_diversity = len(all_gene_types) / max_types
    
    return {
        'fitness_variance': fitness_variance,
        'fitness_std': math.sqrt(fitness_variance),
        'genetic_diversity': genetic_diversity,
        'gene_type_diversity': type_diversity,
        'unique_genomes': unique_genomes
    }
