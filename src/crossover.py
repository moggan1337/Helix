"""
Crossover Module - Sexual Reproduction Operators

This module implements crossover (recombination) operators that
combine genetic material from two parent genomes to create offspring,
simulating sexual reproduction in biological systems.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Type

from .genome import Gene, GeneType, Genome, Instruction


class CrossoverType(Enum):
    """Types of crossover (recombination) operators."""
    
    # Gene-level crossovers
    SINGLE_POINT = auto()       # Single crossover point
    TWO_POINT = auto()          # Two crossover points
    UNIFORM = auto()            # Uniform gene selection
    ARITHMETIC = auto()         # Arithmetic combination
    
    # Instruction-level crossovers
    ONE_POINT_INST = auto()     # Single point within instructions
    TWO_POINT_INST = auto()     # Two points within instructions
    UNIFORM_INST = auto()       # Uniform instruction selection
    
    # Structural crossovers
    GENE_SWAP = auto()          # Swap entire genes
    GENE_SHUFFLE = auto()       # Shuffle genes from both parents
    SUBTREE = auto()            # Subtree exchange (for tree genomes)
    
    # Advanced crossovers
    SIMULATED_BINARY = auto()   # Simulated binary crossover (SBX)
    BLEND = auto()              # Blend crossover (BLX-alpha)
    PARETO = auto()             # Pareto-aware crossover


@dataclass
class CrossoverConfig:
    """Configuration for crossover operators."""
    
    crossover_rate: float = 0.7    # Probability of crossover occurring
    elite_preservation: float = 0.1  # Fraction of elite individuals preserved unchanged
    
    # Gene-level settings
    min_genes: int = 1              # Minimum genes in offspring
    max_genes: int = 50             # Maximum genes in offspring
    
    # Instruction-level settings
    min_instructions: int = 1       # Minimum instructions per gene
    max_instructions: int = 20      # Maximum instructions per gene
    
    # Advanced settings
    allow_empty: bool = False       # Allow empty offspring
    preserve_best: bool = True      # Always preserve best individual
    diversity_pressure: float = 0.2  # Pressure to increase diversity


class CrossoverOperator(ABC):
    """
    Base class for crossover operators.
    
    Crossover operators define how genetic material from two
    parent genomes is combined to produce offspring.
    """
    
    def __init__(self, config: Optional[CrossoverConfig] = None):
        self.config = config or CrossoverConfig()
    
    @abstractmethod
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """
        Perform crossover between two parent genomes.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Tuple of two offspring genomes
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this crossover operator."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this crossover operator."""
        pass


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover - splits genomes at one point and swaps segments.
    
    The most common crossover operator, analogous to homologous
    recombination in biology.
    """
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform single-point crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Ensure both parents have genes
        if not parent1.genes or not parent2.genes:
            return parent1.copy(), parent2.copy()
        
        # Calculate crossover points
        max_point1 = len(parent1.genes)
        max_point2 = len(parent2.genes)
        
        if max_point1 < 2 or max_point2 < 2:
            return parent1.copy(), parent2.copy()
        
        point1 = random.randint(1, max_point1 - 1)
        point2 = random.randint(1, max_point2 - 1)
        
        # Create offspring
        offspring1 = Genome(
            genes=parent1.genes[:point1] + parent2.genes[point2:],
            name=f"{parent1.name}_x_{parent2.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=parent1.lineage + [parent1.id] + parent2.lineage + [parent2.id]
        )
        
        offspring2 = Genome(
            genes=parent2.genes[:point2] + parent1.genes[point1:],
            name=f"{parent2.name}_x_{parent1.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=parent2.lineage + [parent2.id] + parent1.lineage + [parent1.id]
        )
        
        # Enforce gene limits
        self._enforce_gene_limits(offspring1)
        self._enforce_gene_limits(offspring2)
        
        return offspring1, offspring2
    
    def _enforce_gene_limits(self, genome: Genome) -> None:
        """Ensure genome stays within gene limits."""
        while len(genome.genes) > self.config.max_genes:
            # Remove random gene
            genome.genes.pop(random.randint(0, len(genome.genes) - 1))
        
        while len(genome.genes) < self.config.min_genes:
            # Duplicate a random gene
            if genome.genes:
                genome.genes.append(random.choice(genome.genes).copy())
            else:
                break
    
    def get_name(self) -> str:
        return "SinglePointCrossover"
    
    def get_description(self) -> str:
        return "Single crossover point with gene segment exchange"


class TwoPointCrossover(CrossoverOperator):
    """
    Two-point crossover - uses two crossover points.
    
    Can exchange middle segments while preserving ends,
    potentially preserving more structure.
    """
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform two-point crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if not parent1.genes or not parent2.genes:
            return parent1.copy(), parent2.copy()
        
        max_point1 = len(parent1.genes)
        max_point2 = len(parent2.genes)
        
        if max_point1 < 3 or max_point2 < 3:
            return parent1.copy(), parent2.copy()
        
        # Get two distinct points for each parent
        points1 = sorted(random.sample(range(1, max_point1), 2))
        points2 = sorted(random.sample(range(1, max_point2), 2))
        
        # Create offspring with middle segments exchanged
        offspring1_genes = (
            parent1.genes[:points1[0]] +
            parent2.genes[points2[0]:points2[1]] +
            parent1.genes[points1[1]:]
        )
        
        offspring2_genes = (
            parent2.genes[:points2[0]] +
            parent1.genes[points1[0]:points1[1]] +
            parent2.genes[points2[1]:]
        )
        
        offspring1 = Genome(
            genes=offspring1_genes,
            name=f"{parent1.name}_x2_{parent2.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=parent1.lineage + [parent1.id] + parent2.lineage + [parent2.id]
        )
        
        offspring2 = Genome(
            genes=offspring2_genes,
            name=f"{parent2.name}_x2_{parent1.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=parent2.lineage + [parent2.id] + parent1.lineage + [parent1.id]
        )
        
        self._enforce_gene_limits(offspring1)
        self._enforce_gene_limits(offspring2)
        
        return offspring1, offspring2
    
    def _enforce_gene_limits(self, genome: Genome) -> None:
        """Ensure genome stays within gene limits."""
        while len(genome.genes) > self.config.max_genes:
            genome.genes.pop(random.randint(0, len(genome.genes) - 1))
        
        while len(genome.genes) < self.config.min_genes:
            if genome.genes:
                genome.genes.append(random.choice(genome.genes).copy())
            else:
                break
    
    def get_name(self) -> str:
        return "TwoPointCrossover"
    
    def get_description(self) -> str:
        return "Two crossover points with middle segment exchange"


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover - randomly selects genes from each parent.
    
    Each gene has equal probability of coming from either parent,
    allowing more mixing of genetic material.
    """
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform uniform crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if not parent1.genes or not parent2.genes:
            return parent1.copy(), parent2.copy()
        
        # Get all unique genes from both parents
        all_genes_1 = list(parent1.genes)
        all_genes_2 = list(parent2.genes)
        
        # Shuffle and split for offspring 1
        combined_1 = all_genes_1 + all_genes_2
        random.shuffle(combined_1)
        mid_1 = len(combined_1) // 2
        
        # Shuffle and split for offspring 2 (different shuffle)
        combined_2 = all_genes_1 + all_genes_2
        random.shuffle(combined_2)
        mid_2 = len(combined_2) // 2
        
        offspring1 = Genome(
            genes=combined_1[:mid_1],
            name=f"{parent1.name}_xu_{parent2.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=parent1.lineage + [parent1.id] + parent2.lineage + [parent2.id]
        )
        
        offspring2 = Genome(
            genes=combined_2[:mid_2],
            name=f"{parent2.name}_xu_{parent1.name}",
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=parent2.lineage + [parent2.id] + parent1.lineage + [parent1.id]
        )
        
        self._enforce_gene_limits(offspring1)
        self._enforce_gene_limits(offspring2)
        
        return offspring1, offspring2
    
    def _enforce_gene_limits(self, genome: Genome) -> None:
        """Ensure genome stays within gene limits."""
        while len(genome.genes) > self.config.max_genes:
            genome.genes.pop(random.randint(0, len(genome.genes) - 1))
        
        while len(genome.genes) < self.config.min_genes:
            if genome.genes:
                genome.genes.append(random.choice(genome.genes).copy())
            else:
                break
    
    def get_name(self) -> str:
        return "UniformCrossover"
    
    def get_description(self) -> str:
        return "Random gene selection from both parents"


class GeneSwapCrossover(CrossoverOperator):
    """
    Gene swap crossover - swaps entire genes between parents.
    
    A simpler form that maintains gene integrity while
    allowing gene-level recombination.
    """
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform gene swap crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if not parent1.genes or not parent2.genes:
            return parent1.copy(), parent2.copy()
        
        # Pick random genes to swap
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Swap 1-3 random genes
        num_swaps = min(random.randint(1, 3), min(len(parent1.genes), len(parent2.genes)))
        
        for _ in range(num_swaps):
            idx1 = random.randint(0, len(offspring1.genes) - 1)
            idx2 = random.randint(0, len(offspring2.genes) - 1)
            offspring1.genes[idx1], offspring2.genes[idx2] = (
                offspring2.genes[idx2].copy(),
                offspring1.genes[idx1].copy()
            )
        
        return offspring1, offspring2
    
    def get_name(self) -> str:
        return "GeneSwapCrossover"
    
    def get_description(self) -> str:
        return "Swaps entire genes between parents"


class InstructionCrossover(CrossoverOperator):
    """
    Instruction-level crossover - recombines instructions within genes.
    
    Performs crossover at the instruction level within genes,
    allowing finer-grained recombination.
    """
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform instruction-level crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Find matching genes (by type or name) for crossover
        for gene1 in offspring1.genes:
            for gene2 in offspring2.genes:
                if gene1.gene_type == gene2.gene_type and gene1.instructions and gene2.instructions:
                    if random.random() < 0.5:
                        self._crossover_instructions(gene1, gene2)
        
        return offspring1, offspring2
    
    def _crossover_instructions(self, gene1: Gene, gene2: Gene) -> None:
        """Crossover instructions between two genes."""
        if len(gene1.instructions) < 2 or len(gene2.instructions) < 2:
            return
        
        # Single point crossover on instructions
        point1 = random.randint(1, len(gene1.instructions) - 1)
        point2 = random.randint(1, len(gene2.instructions) - 1)
        
        # Swap instruction segments
        temp = gene1.instructions[point1:]
        gene1.instructions[point1:] = gene2.instructions[point2:]
        gene2.instructions[point2:] = temp
    
    def get_name(self) -> str:
        return "InstructionCrossover"
    
    def get_description(self) -> str:
        return "Recombines instructions within genes"


class ArithmeticCrossover(CrossoverOperator):
    """
    Arithmetic crossover - combines numeric parameters arithmetically.
    
    Useful for evolving numeric constants, combining values
    from both parents using arithmetic operations.
    """
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform arithmetic crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        alpha = random.uniform(-0.5, 1.5)  # Blend factor
        
        for gene1, gene2 in zip(offspring1.genes, offspring2.genes):
            self._blend_parameters(gene1, gene2, alpha)
        
        return offspring1, offspring2
    
    def _blend_parameters(self, gene1: Gene, gene2: Gene, alpha: float) -> None:
        """Blend numeric parameters between two genes."""
        # Simple averaging for parameters
        for key in set(gene1.parameters.keys()) | set(gene2.parameters.keys()):
            val1 = gene1.parameters.get(key, 0)
            val2 = gene2.parameters.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Blend the values
                blended = alpha * val1 + (1 - alpha) * val2
                
                if isinstance(val1, int) and isinstance(val2, int):
                    blended = int(blended)
                
                gene1.parameters[key] = blended
                gene2.parameters[key] = alpha * val2 + (1 - alpha) * val1
    
    def get_name(self) -> str:
        return "ArithmeticCrossover"
    
    def get_description(self) -> str:
        return "Arithmetically combines numeric parameters"


class SimulatedBinaryCrossover(CrossoverOperator):
    """
    Simulated Binary Crossover (SBX) - mimics binary crossover in real-coded GAs.
    
    Popular in evolutionary algorithms for real-valued optimization,
    with controllable distribution index.
    """
    
    def __init__(self, config: Optional[CrossoverConfig] = None, distribution_index: float = 20.0):
        super().__init__(config)
        self.distribution_index = distribution_index
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform simulated binary crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for gene1, gene2 in zip(offspring1.genes, offspring2.genes):
            self._sbx_blend(gene1, gene2)
        
        return offspring1, offspring2
    
    def _sbx_blend(self, gene1: Gene, gene2: Gene) -> None:
        """Apply SBX blending to gene parameters."""
        for key in set(gene1.parameters.keys()) & set(gene2.parameters.keys()):
            val1 = gene1.parameters.get(key, 0)
            val2 = gene2.parameters.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Calculate spread
                if val1 == val2:
                    continue
                
                u = random.random()
                
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.distribution_index + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.distribution_index + 1))
                
                child1 = 0.5 * ((1 + beta) * val1 + (1 - beta) * val2)
                child2 = 0.5 * ((1 - beta) * val1 + (1 + beta) * val2)
                
                if isinstance(val1, int):
                    child1, child2 = int(child1), int(child2)
                
                gene1.parameters[key] = child1
                gene2.parameters[key] = child2
    
    def get_name(self) -> str:
        return "SimulatedBinaryCrossover"
    
    def get_description(self) -> str:
        return "Simulated binary crossover for real-coded optimization"


class BlendCrossover(CrossoverOperator):
    """
    Blend Crossover (BLX-alpha) - creates offspring in extended range.
    
    Extends the range of values beyond parent bounds by alpha factor.
    """
    
    def __init__(self, config: Optional[CrossoverConfig] = None, alpha: float = 0.5):
        super().__init__(config)
        self.alpha = alpha
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform blend crossover."""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        for gene1, gene2 in zip(offspring1.genes, offspring2.genes):
            self._blend_extend(gene1, gene2)
        
        return offspring1, offspring2
    
    def _blend_extend(self, gene1: Gene, gene2: Gene) -> None:
        """Apply BLX-alpha blending to gene parameters."""
        for key in set(gene1.parameters.keys()) | set(gene2.parameters.keys()):
            val1 = gene1.parameters.get(key, 0)
            val2 = gene2.parameters.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                min_val = min(val1, val2)
                max_val = max(val1, val2)
                i = max_val - min_val
                
                lower = min_val - self.alpha * i
                upper = max_val + self.alpha * i
                
                gene1.parameters[key] = random.uniform(lower, upper)
                gene2.parameters[key] = random.uniform(lower, upper)
                
                if isinstance(val1, int):
                    gene1.parameters[key] = int(gene1.parameters[key])
                    gene2.parameters[key] = int(gene2.parameters[key])
    
    def get_name(self) -> str:
        return "BlendCrossover"
    
    def get_description(self) -> str:
        return f"Blend crossover with alpha={self.alpha}"


class CrossoverPipeline:
    """
    Pipeline of crossover operators with selection.
    
    Allows combining multiple crossover operators and
    selecting which to use based on various criteria.
    """
    
    def __init__(self, config: Optional[CrossoverConfig] = None):
        self.config = config or CrossoverConfig()
        self.operators: List[CrossoverOperator] = []
        self.operator_weights: List[float] = []
        
        # Add default operators
        self.add_operator(SinglePointCrossover(self.config), weight=1.0)
        self.add_operator(TwoPointCrossover(self.config), weight=0.8)
        self.add_operator(UniformCrossover(self.config), weight=0.6)
        self.add_operator(GeneSwapCrossover(self.config), weight=0.4)
        self.add_operator(InstructionCrossover(self.config), weight=0.5)
    
    def add_operator(self, operator: CrossoverOperator, weight: float = 1.0) -> None:
        """Add an operator to the pipeline."""
        self.operators.append(operator)
        self.operator_weights.append(weight)
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """Perform crossover using selected operator."""
        if not self.operators:
            return parent1.copy(), parent2.copy()
        
        # Normalize weights
        total = sum(self.operator_weights)
        if total == 0:
            return parent1.copy(), parent2.copy()
        
        probs = [w / total for w in self.operator_weights]
        
        # Select and apply operator
        operator = random.choices(self.operators, weights=probs)[0]
        return operator.crossover(parent1, parent2)
    
    def crossover_with_strategy(
        self,
        parent1: Genome,
        parent2: Genome,
        strategy: str = "random"
    ) -> Tuple[Genome, Genome]:
        """
        Perform crossover with specific strategy.
        
        Args:
            parent1: First parent
            parent2: Second parent
            strategy: One of 'random', 'diverse', 'similar'
        """
        if strategy == "random":
            return self.crossover(parent1, parent2)
        
        elif strategy == "diverse":
            # Use operators that maximize diversity
            diverse_ops = [
                op for op in self.operators
                if isinstance(op, (UniformCrossover, TwoPointCrossover))
            ]
            if diverse_ops:
                operator = random.choice(diverse_ops)
                return operator.crossover(parent1, parent2)
        
        elif strategy == "similar":
            # Use operators that preserve similarity
            similar_ops = [
                op for op in self.operators
                if isinstance(op, (SinglePointCrossover, GeneSwapCrossover))
            ]
            if similar_ops:
                operator = random.choice(similar_ops)
                return operator.crossover(parent1, parent2)
        
        return self.crossover(parent1, parent2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about crossover operators."""
        return {
            'num_operators': len(self.operators),
            'operators': [
                {
                    'name': op.get_name(),
                    'description': op.get_description(),
                    'weight': weight
                }
                for op, weight in zip(self.operators, self.operator_weights)
            ]
        }
