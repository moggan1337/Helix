"""
Population Module - Population Management and Statistics

This module manages populations of genomes, tracking their properties,
maintaining diversity, and providing statistics about the population.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .genome import Genome


@dataclass
class PopulationStatistics:
    """Statistics about a population."""
    
    generation: int = 0
    size: int = 0
    
    # Fitness statistics
    best_fitness: float = 0.0
    worst_fitness: float = 0.0
    mean_fitness: float = 0.0
    median_fitness: float = 0.0
    std_fitness: float = 0.0
    
    # Diversity statistics
    diversity: float = 0.0
    unique_genomes: int = 0
    gene_type_diversity: float = 0.0
    
    # Complexity statistics
    mean_complexity: float = 0.0
    max_complexity: int = 0
    min_complexity: int = 0
    
    # Selection statistics
    selection_pressure: float = 0.0
    
    # Best genome
    best_genome_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'generation': self.generation,
            'size': self.size,
            'best_fitness': self.best_fitness,
            'worst_fitness': self.worst_fitness,
            'mean_fitness': self.mean_fitness,
            'median_fitness': self.median_fitness,
            'std_fitness': self.std_fitness,
            'diversity': self.diversity,
            'unique_genomes': self.unique_genomes,
            'gene_type_diversity': self.gene_type_diversity,
            'mean_complexity': self.mean_complexity,
            'max_complexity': self.max_complexity,
            'min_complexity': self.min_complexity,
            'best_genome_id': self.best_genome_id
        }


class Population:
    """
    Manages a population of genomes.
    
    The population is the central entity in genetic algorithms -
    it holds all individuals, applies evolutionary operators,
    and tracks statistics.
    """
    
    def __init__(
        self,
        size: int = 100,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None
    ):
        """
        Initialize population.
        
        Args:
            size: Target population size
            min_size: Minimum population size
            max_size: Maximum population size
        """
        self.individuals: List[Genome] = []
        self.target_size = size
        self.min_size = min_size or max(10, size // 5)
        self.max_size = max_size or size * 2
        
        self.generation = 0
        self.history: List[PopulationStatistics] = []
        
        # Diversity management
        self.min_diversity = 0.1
        self.max_age = 50  # Maximum generations without improvement
    
    def __len__(self) -> int:
        """Return population size."""
        return len(self.individuals)
    
    def __getitem__(self, index: int) -> Genome:
        """Get individual by index."""
        return self.individuals[index]
    
    def __iter__(self):
        """Iterate over population."""
        return iter(self.individuals)
    
    # =========================================================================
    # Population Initialization
    # =========================================================================
    
    def initialize_random(
        self,
        genes_per_genome: int = 5,
        complexity_per_gene: int = 3
    ) -> None:
        """
        Initialize population with random genomes.
        
        Args:
            genes_per_genome: Number of genes per genome
            complexity_per_gene: Instructions per gene
        """
        self.individuals = []
        
        for i in range(self.target_size):
            genome = Genome.random(
                num_genes=genes_per_genome,
                complexity_per_gene=complexity_per_gene
            )
            genome.name = f"gen_{self.generation}_{i}"
            genome.generation = self.generation
            self.individuals.append(genome)
    
    def initialize_minimal(self) -> None:
        """
        Initialize population with minimal seed genomes.
        
        Creates population from minimal viable genomes that can
        evolve into complex solutions.
        """
        self.individuals = []
        
        for i in range(self.target_size):
            # Start with minimal genome
            genome = Genome.minimal()
            
            # Add some random variation
            for _ in range(random.randint(0, 3)):
                genome = genome.mutate(mutation_rate=0.3)
            
            genome.name = f"seed_{self.generation}_{i}"
            genome.generation = self.generation
            self.individuals.append(genome)
    
    def initialize_from_seeds(self, seeds: List[Genome]) -> None:
        """
        Initialize population from seed genomes.
        
        Args:
            seeds: List of seed genomes to start with
        """
        self.individuals = []
        
        # Fill with variations of seeds
        seed_idx = 0
        while len(self.individuals) < self.target_size:
            seed = seeds[seed_idx % len(seeds)]
            genome = seed.mutate(mutation_rate=0.5)
            genome.name = f"seed_{self.generation}_{len(self.individuals)}"
            genome.generation = self.generation
            self.individuals.append(genome)
            seed_idx += 1
    
    def initialize_adaptive(
        self,
        task_complexity: float = 0.5
    ) -> None:
        """
        Initialize with adaptive complexity based on task.
        
        Args:
            task_complexity: Expected task complexity (0-1)
        """
        # Base genome count scales with task complexity
        base_genes = int(3 + task_complexity * 10)
        complexity = int(2 + task_complexity * 5)
        
        self.initialize_random(
            genes_per_genome=base_genes,
            complexity_per_gene=complexity
        )
    
    # =========================================================================
    # Population Operations
    # =========================================================================
    
    def add(self, genome: Genome) -> None:
        """Add a genome to the population."""
        genome.generation = self.generation
        self.individuals.append(genome)
    
    def remove(self, index: int) -> Genome:
        """Remove a genome by index."""
        return self.individuals.pop(index)
    
    def get_best(self, count: int = 1) -> List[Genome]:
        """
        Get the best genomes by fitness.
        
        Args:
            count: Number of genomes to return
            
        Returns:
            List of best genomes (sorted by fitness descending)
        """
        sorted_pop = sorted(self.individuals, key=lambda g: g.fitness, reverse=True)
        return sorted_pop[:count]
    
    def get_worst(self, count: int = 1) -> List[Genome]:
        """Get the worst genomes by fitness."""
        sorted_pop = sorted(self.individuals, key=lambda g: g.fitness)
        return sorted_pop[:count]
    
    def get_random(self, count: int = 1) -> List[Genome]:
        """Get random genomes."""
        return random.sample(self.individuals, min(count, len(self.individuals)))
    
    def select_parents(self, count: int = 2) -> List[Genome]:
        """
        Select parents for reproduction.
        
        Uses fitness-proportional selection with some randomization.
        
        Args:
            count: Number of parents to select
            
        Returns:
            Selected parent genomes
        """
        if not self.individuals:
            return []
        
        # Calculate selection probabilities
        fitnesses = [max(0.0, g.fitness) for g in self.individuals]
        total = sum(fitnesses)
        
        if total == 0:
            # Uniform selection if all fitnesses are zero
            return random.sample(self.individuals, min(count, len(self.individuals)))
        
        probabilities = [f / total for f in fitnesses]
        
        parents = []
        for _ in range(count):
            r = random.random()
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    parents.append(copy.deepcopy(self.individuals[i]))
                    break
            else:
                parents.append(copy.deepcopy(self.individuals[-1]))
        
        return parents
    
    def cull(self, fraction: float = 0.5) -> None:
        """
        Remove the worst fraction of the population.
        
        Args:
            fraction: Fraction to remove (0-1)
        """
        if not self.individuals:
            return
        
        sorted_pop = sorted(self.individuals, key=lambda g: g.fitness)
        keep_count = int(len(self.individuals) * (1 - fraction))
        
        self.individuals = sorted_pop[keep_count:]
    
    def grow(self, offspring: List[Genome]) -> None:
        """
        Add offspring to the population.
        
        Args:
            offspring: New genomes to add
        """
        for genome in offspring:
            genome.generation = self.generation
        self.individuals.extend(offspring)
    
    def replace_worst(self, new_genomes: List[Genome]) -> None:
        """Replace worst genomes with new ones."""
        sorted_pop = sorted(enumerate(self.individuals), key=lambda x: x[1].fitness)
        
        # Replace worst genomes
        for (idx, _), new_genome in zip(sorted_pop[:len(new_genomes)], new_genomes):
            self.individuals[idx] = new_genome
    
    def enforce_size(self) -> None:
        """Ensure population stays within size bounds."""
        while len(self.individuals) > self.max_size:
            # Remove worst individual
            worst_idx = min(
                range(len(self.individuals)),
                key=lambda i: self.individuals[i].fitness
            )
            self.individuals.pop(worst_idx)
        
        while len(self.individuals) < self.min_size:
            # Add mutated copy of best
            if self.individuals:
                best = self.get_best(1)[0]
                new_genome = best.mutate(mutation_rate=0.3)
                new_genome.name = f"gen_{self.generation}_replenish"
                new_genome.generation = self.generation
                self.individuals.append(new_genome)
            else:
                # Create random genome if empty
                self.individuals.append(Genome.random())
    
    def next_generation(self) -> None:
        """Advance to next generation."""
        self.generation += 1
        for genome in self.individuals:
            genome.generation = self.generation
    
    # =========================================================================
    # Diversity Management
    # =========================================================================
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity.
        
        Returns:
            Diversity score (0-1)
        """
        if len(self.individuals) < 2:
            return 0.0
        
        # Fitness diversity
        fitnesses = [g.fitness for g in self.individuals]
        fitness_set = len(set(fitnesses))
        fitness_diversity = fitness_set / len(self.individuals)
        
        # Genetic diversity (based on genome IDs)
        unique_ids = len(set(g.id for g in self.individuals))
        genetic_diversity = unique_ids / len(self.individuals)
        
        # Gene type diversity
        all_types = set()
        for g in self.individuals:
            for gene in g.genes:
                all_types.add(gene.gene_type)
        type_diversity = len(all_types) / 50  # Approximate number of gene types
        
        return (fitness_diversity + genetic_diversity + type_diversity) / 3
    
    def maintain_diversity(self) -> None:
        """
        Maintain population diversity.
        
        If diversity is too low, adds random individuals.
        """
        diversity = self.calculate_diversity()
        
        if diversity < self.min_diversity:
            # Add random individuals to increase diversity
            needed = int(len(self.individuals) * (self.min_diversity - diversity))
            
            for _ in range(needed):
                new_genome = Genome.random(
                    num_genes=random.randint(2, 6),
                    complexity_per_gene=random.randint(2, 5)
                )
                new_genome.name = f"diversity_{self.generation}"
                new_genome.generation = self.generation
                self.individuals.append(new_genome)
            
            self.enforce_size()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> PopulationStatistics:
        """
        Calculate population statistics.
        
        Returns:
            PopulationStatistics object
        """
        if not self.individuals:
            return PopulationStatistics(generation=self.generation)
        
        fitnesses = sorted([g.fitness for g in self.individuals])
        complexities = [g.complexity for g in self.individuals]
        
        n = len(fitnesses)
        mean_fitness = sum(fitnesses) / n
        
        # Variance and std
        variance = sum((f - mean_fitness) ** 2 for f in fitnesses) / n
        std_fitness = variance ** 0.5
        
        # Median
        if n % 2 == 0:
            median_fitness = (fitnesses[n // 2 - 1] + fitnesses[n // 2]) / 2
        else:
            median_fitness = fitnesses[n // 2]
        
        # Diversity
        diversity = self.calculate_diversity()
        
        # Unique genomes
        unique_genomes = len(set(g.id for g in self.individuals))
        
        # Gene type diversity
        all_types = set()
        for g in self.individuals:
            for gene in g.genes:
                all_types.add(gene.gene_type)
        gene_type_diversity = len(all_types) / 50
        
        # Best genome
        best = self.get_best(1)[0] if self.individuals else None
        
        stats = PopulationStatistics(
            generation=self.generation,
            size=n,
            best_fitness=max(fitnesses),
            worst_fitness=min(fitnesses),
            mean_fitness=mean_fitness,
            median_fitness=median_fitness,
            std_fitness=std_fitness,
            diversity=diversity,
            unique_genomes=unique_genomes,
            gene_type_diversity=gene_type_diversity,
            mean_complexity=sum(complexities) / n,
            max_complexity=max(complexities),
            min_complexity=min(complexities),
            best_genome_id=best.id if best else ""
        )
        
        self.history.append(stats)
        return stats
    
    def get_history(self) -> List[PopulationStatistics]:
        """Get population history."""
        return self.history
    
    def get_fitness_history(self) -> List[float]:
        """Get best fitness per generation."""
        return [h.best_fitness for h in self.history]
    
    def get_diversity_history(self) -> List[float]:
        """Get diversity per generation."""
        return [h.diversity for h in self.history]
    
    # =========================================================================
    # Special Operations
    # =========================================================================
    
    def clone(self) -> Population:
        """Create a deep copy of this population."""
        new_pop = Population(
            size=self.target_size,
            min_size=self.min_size,
            max_size=self.max_size
        )
        new_pop.individuals = [g.copy() for g in self.individuals]
        new_pop.generation = self.generation
        new_pop.history = copy.deepcopy(self.history)
        new_pop.min_diversity = self.min_diversity
        return new_pop
    
    def merge(self, other: Population) -> None:
        """
        Merge another population into this one.
        
        Args:
            other: Population to merge
        """
        for genome in other.individuals:
            genome.generation = self.generation
        self.individuals.extend(other.individuals)
        self.enforce_size()
    
    def clear(self) -> None:
        """Clear the population."""
        self.individuals.clear()
        self.history.clear()
        self.generation = 0
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert population to serializable list."""
        return [
            {
                'id': g.id,
                'name': g.name,
                'generation': g.generation,
                'fitness': g.fitness,
                'complexity': g.complexity,
                'num_genes': len(g.genes),
                'is_viable': g.is_viable
            }
            for g in self.individuals
        ]
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> Population:
        """
        Create population from serialized list.
        
        Note: Genomes will be minimal - only metadata is preserved.
        """
        pop = cls()
        for item in data:
            genome = Genome(name=item.get('name', 'loaded'))
            genome.id = item.get('id', '')
            genome.generation = item.get('generation', 0)
            genome.fitness = item.get('fitness', 0.0)
            pop.individuals.append(genome)
        return pop
