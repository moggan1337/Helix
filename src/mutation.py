"""
Mutation Module - Genetic Variation Operators

This module implements various mutation operators that introduce
genetic variation into the population, simulating biological mutations.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Type

from .genome import Gene, GeneType, Genome, Instruction


class MutationType(Enum):
    """Types of mutation operators."""
    
    # Point mutations (single instruction changes)
    POINT = auto()              # Single instruction change
    INSERT = auto()             # Insert new instruction
    DELETE = auto()             # Delete existing instruction
    REPLACE = auto()            # Replace instruction
    
    # Structural mutations (gene-level changes)
    DUPLICATE = auto()          # Duplicate a gene
    DELETE_GENE = auto()        # Delete entire gene
    SWAP_GENES = auto()         # Swap two genes
    TRANSPOSE = auto()          # Move gene to different position
    
    # Parameter mutations
    PARAMETER = auto()          # Modify gene parameters
    ENABLE = auto()             # Enable disabled gene
    DISABLE = auto()            # Disable active gene
    
    # Complex mutations
    SCRAMBLE = auto()           # Randomly shuffle instructions
    INVERSION = auto()          # Reverse order of instructions
    
    # Meta mutations
    COPY_FROM_BEST = auto()     # Copy genes from best individual
    HYBRID = auto()             # Combine multiple mutation types


@dataclass
class MutationConfig:
    """Configuration for mutation operators."""
    
    point_rate: float = 0.1           # Rate for point mutations
    insertion_rate: float = 0.05      # Rate for insertions
    deletion_rate: float = 0.03       # Rate for deletions
    structural_rate: float = 0.02     # Rate for structural mutations
    parameter_rate: float = 0.08      # Rate for parameter mutations
    enable_disable_rate: float = 0.01 # Rate for enable/disable mutations
    
    max_instructions_per_gene: int = 20  # Maximum instructions in a gene
    max_genes: int = 50                   # Maximum genes in genome
    min_instructions_per_gene: int = 1   # Minimum instructions in a gene
    
    adaptive: bool = True              # Adapt rates based on fitness
    diversity_weight: float = 0.3       # Weight for diversity in adaptation


class MutationOperator(ABC):
    """
    Base class for mutation operators.
    
    Mutation operators define how genetic variation is introduced
    into individuals during evolution.
    """
    
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()
    
    @abstractmethod
    def mutate(self, genome: Genome) -> Genome:
        """
        Apply mutation to a genome.
        
        Args:
            genome: The genome to mutate
            
        Returns:
            Mutated genome (may be same as input if no mutation)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this mutation operator."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this mutation operator."""
        pass


class PointMutation(MutationOperator):
    """
    Point mutation - changes single instructions within genes.
    
    This is the most common form of mutation, analogous to point
    mutations in DNA where single nucleotides change.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply point mutation to genome."""
        if not genome.genes:
            return genome
        
        new_genome = genome.copy()
        
        for gene in new_genome.genes:
            if not gene.instructions:
                continue
            
            # Mutate each instruction with probability
            for i in range(len(gene.instructions)):
                if random.random() < self.config.point_rate:
                    gene.instructions[i] = gene.instructions[i].mutate(
                        mutation_rate=0.5  # High internal mutation rate
                    )
        
        return new_genome
    
    def get_name(self) -> str:
        return "PointMutation"
    
    def get_description(self) -> str:
        return "Mutates individual instructions within genes"


class InsertionMutation(MutationOperator):
    """
    Insertion mutation - adds new instructions to genes.
    
    Analogous to DNA insertion mutations where new base pairs
    are inserted into the sequence.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply insertion mutation to genome."""
        if random.random() > self.config.insertion_rate:
            return genome
        
        new_genome = genome.copy()
        
        # Pick a random gene to insert into
        if not new_genome.genes:
            return new_genome
        
        gene_idx = random.randint(0, len(new_genome.genes) - 1)
        gene = new_genome.genes[gene_idx]
        
        # Check max limit
        if len(gene.instructions) >= self.config.max_instructions_per_gene:
            return new_genome
        
        # Create new instruction
        new_instruction = self._create_random_instruction()
        
        # Insert at random position
        pos = random.randint(0, len(gene.instructions))
        gene.instructions.insert(pos, new_instruction)
        
        return new_genome
    
    def _create_random_instruction(self) -> Instruction:
        """Create a random instruction."""
        opcodes = [
            'add', 'sub', 'mul', 'div', 'mod', 'pow',
            'and', 'or', 'not', 'eq', 'ne', 'lt', 'le', 'gt', 'ge',
            'assign', 'load', 'store', 'print'
        ]
        
        opcode = random.choice(opcodes)
        num_operands = random.randint(0, 3)
        operands = tuple(random.randint(0, 100) for _ in range(num_operands))
        
        return Instruction(opcode=opcode, operands=operands)
    
    def get_name(self) -> str:
        return "InsertionMutation"
    
    def get_description(self) -> str:
        return "Inserts new instructions into genes"


class DeletionMutation(MutationOperator):
    """
    Deletion mutation - removes instructions from genes.
    
    Analogous to DNA deletion mutations where base pairs
    are removed from the sequence.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply deletion mutation to genome."""
        if random.random() > self.config.deletion_rate:
            return genome
        
        new_genome = genome.copy()
        
        for gene in new_genome.genes:
            # Check min limit
            if len(gene.instructions) <= self.config.min_instructions_per_gene:
                continue
            
            # Delete instruction at random position
            pos = random.randint(0, len(gene.instructions) - 1)
            gene.instructions.pop(pos)
        
        return new_genome
    
    def get_name(self) -> str:
        return "DeletionMutation"
    
    def get_description(self) -> str:
        return "Deletes instructions from genes"


class StructuralMutation(MutationOperator):
    """
    Structural mutation - modifies gene-level structure.
    
    Includes gene duplication, deletion, swapping, and transposition.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply structural mutation to genome."""
        if random.random() > self.config.structural_rate:
            return genome
        
        if len(genome.genes) < 1:
            return genome
        
        new_genome = genome.copy()
        mutation_type = random.choice([
            'duplicate', 'delete', 'swap', 'transpose'
        ])
        
        if mutation_type == 'duplicate' and len(new_genome.genes) < self.config.max_genes:
            gene_idx = random.randint(0, len(new_genome.genes) - 1)
            duplicated = new_genome.genes[gene_idx].copy()
            duplicated.name = f"{duplicated.name}_dup"
            new_genome.genes.append(duplicated)
        
        elif mutation_type == 'delete' and len(new_genome.genes) > 1:
            gene_idx = random.randint(0, len(new_genome.genes) - 1)
            new_genome.genes.pop(gene_idx)
        
        elif mutation_type == 'swap' and len(new_genome.genes) >= 2:
            idx1, idx2 = random.sample(range(len(new_genome.genes)), 2)
            new_genome.genes[idx1], new_genome.genes[idx2] = (
                new_genome.genes[idx2], new_genome.genes[idx1]
            )
        
        elif mutation_type == 'transpose':
            # Move gene from one position to another
            if len(new_genome.genes) >= 2:
                from_idx = random.randint(0, len(new_genome.genes) - 1)
                gene = new_genome.genes.pop(from_idx)
                to_idx = random.randint(0, len(new_genome.genes))
                new_genome.genes.insert(to_idx, gene)
        
        return new_genome
    
    def get_name(self) -> str:
        return "StructuralMutation"
    
    def get_description(self) -> str:
        return "Modifies gene-level structure (duplicate, delete, swap, transpose)"


class ParameterMutation(MutationOperator):
    """
    Parameter mutation - modifies gene parameters.
    
    Changes the configuration values stored in gene parameters,
    potentially altering gene behavior.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply parameter mutation to genome."""
        if random.random() > self.config.parameter_rate:
            return genome
        
        new_genome = genome.copy()
        
        for gene in new_genome.genes:
            if not gene.parameters:
                # Create some initial parameters
                gene.parameters = {
                    'threshold': random.uniform(0, 1),
                    'weight': random.uniform(-1, 1),
                    'enabled': True
                }
                continue
            
            # Mutate existing parameters
            for key in gene.parameters:
                if random.random() < 0.5:  # 50% chance per parameter
                    value = gene.parameters[key]
                    gene.parameters[key] = self._mutate_value(value)
        
        return new_genome
    
    def _mutate_value(self, value: Any) -> Any:
        """Mutate a parameter value."""
        if isinstance(value, int):
            # Gaussian mutation
            return int(value + random.gauss(0, value / 10 + 1))
        elif isinstance(value, float):
            # Gaussian mutation
            return value + random.gauss(0, abs(value) / 10 + 0.1)
        elif isinstance(value, bool):
            return not value
        elif isinstance(value, str):
            # Random string of similar length
            length = len(value)
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        else:
            return value
    
    def get_name(self) -> str:
        return "ParameterMutation"
    
    def get_description(self) -> str:
        return "Modifies gene parameter values"


class EnableDisableMutation(MutationOperator):
    """
    Enable/Disable mutation - toggles gene activity.
    
    Can disable beneficial genes temporarily (neutral drift)
    or re-enable previously disabled genes.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply enable/disable mutation to genome."""
        if random.random() > self.config.enable_disable_rate:
            return genome
        
        new_genome = genome.copy()
        
        # Count enabled/disabled genes
        enabled = [i for i, g in enumerate(new_genome.genes) if g.enabled]
        disabled = [i for i, g in enumerate(new_genome.genes) if not g.enabled]
        
        if enabled and random.random() < 0.7:
            #倾向于禁用（模拟基因沉默）
            idx = random.choice(enabled)
            new_genome.genes[idx].enabled = False
        elif disabled and random.random() < 0.3:
            #倾向于启用（模拟基因激活）
            idx = random.choice(disabled)
            new_genome.genes[idx].enabled = True
        
        return new_genome
    
    def get_name(self) -> str:
        return "EnableDisableMutation"
    
    def get_description(self) -> str:
        return "Enables or disables genes"


class ScrambleMutation(MutationOperator):
    """
    Scramble mutation - randomly shuffles instructions.
    
    Completely randomizes the order of instructions within
    a gene, causing potentially large behavioral changes.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply scramble mutation to genome."""
        if random.random() > self.config.point_rate:
            return genome
        
        new_genome = genome.copy()
        
        for gene in new_genome.genes:
            if len(gene.instructions) > 2:
                # Shuffle instructions
                start = random.randint(0, len(gene.instructions) - 2)
                end = random.randint(start + 1, len(gene.instructions))
                segment = gene.instructions[start:end]
                random.shuffle(segment)
                gene.instructions[start:end] = segment
        
        return new_genome
    
    def get_name(self) -> str:
        return "ScrambleMutation"
    
    def get_description(self) -> str:
        return "Randomly shuffles instruction order within genes"


class InversionMutation(MutationOperator):
    """
    Inversion mutation - reverses instruction order.
    
    Reverses a segment of instructions within a gene,
    potentially preserving or changing overall behavior.
    """
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply inversion mutation to genome."""
        if random.random() > self.config.point_rate * 0.5:
            return genome
        
        new_genome = genome.copy()
        
        for gene in new_genome.genes:
            if len(gene.instructions) > 2:
                # Pick a segment to invert
                start = random.randint(0, len(gene.instructions) - 2)
                end = random.randint(start + 1, len(gene.instructions))
                gene.instructions[start:end] = reversed(gene.instructions[start:end])
        
        return new_genome
    
    def get_name(self) -> str:
        return "InversionMutation"
    
    def get_description(self) -> str:
        return "Reverses instruction order within genes"


class AdaptiveMutation(MutationOperator):
    """
    Adaptive mutation - adjusts mutation rates based on fitness.
    
    When fitness is low, uses higher mutation rates to explore
    more of the search space. When fitness is high, uses lower
    rates to fine-tune existing solutions.
    """
    
    def __init__(self, base_config: MutationConfig, operators: List[MutationOperator]):
        super().__init__(base_config)
        self.operators = operators
        self.fitness_history: List[float] = []
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply adaptive mutation to genome."""
        # Update fitness history
        self.fitness_history.append(genome.fitness)
        if len(self.fitness_history) > 10:
            self.fitness_history.pop(0)
        
        # Calculate adaptation factor
        if len(self.fitness_history) >= 2:
            fitness_variance = self._calculate_variance(self.fitness_history)
            stagnation = self._detect_stagnation()
            
            # Higher mutation when stagnant or high variance
            if stagnation > 5 or fitness_variance > 0.1:
                mutation_multiplier = 2.0
            else:
                mutation_multiplier = 0.5
        else:
            mutation_multiplier = 1.0
        
        # Apply mutation with adapted rate
        if random.random() < self.config.point_rate * mutation_multiplier:
            operator = random.choice(self.operators)
            return operator.mutate(genome)
        
        return genome
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of fitness values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def _detect_stagnation(self) -> int:
        """Detect how many generations since fitness improvement."""
        if len(self.fitness_history) < 2:
            return 0
        
        stagnation = 0
        for i in range(len(self.fitness_history) - 1, 0, -1):
            if abs(self.fitness_history[i] - self.fitness_history[i-1]) < 0.001:
                stagnation += 1
            else:
                break
        return stagnation
    
    def get_name(self) -> str:
        return "AdaptiveMutation"
    
    def get_description(self) -> str:
        return "Adaptively adjusts mutation rates based on fitness progress"


class MutationPipeline:
    """
    Pipeline of mutation operators applied in sequence.
    
    Allows combining multiple mutation operators with
    different rates and configurations.
    """
    
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()
        self.operators: List[MutationOperator] = []
        self.operator_rates: List[float] = []
        
        # Add default operators
        self.add_operator(PointMutation(self.config), rate=1.0)
        self.add_operator(InsertionMutation(self.config), rate=0.3)
        self.add_operator(DeletionMutation(self.config), rate=0.2)
        self.add_operator(StructuralMutation(self.config), rate=0.1)
        self.add_operator(ParameterMutation(self.config), rate=0.5)
        self.add_operator(EnableDisableMutation(self.config), rate=0.1)
        self.add_operator(ScrambleMutation(self.config), rate=0.1)
        self.add_operator(InversionMutation(self.config), rate=0.05)
    
    def add_operator(self, operator: MutationOperator, rate: float = 1.0) -> None:
        """Add an operator to the pipeline."""
        self.operators.append(operator)
        self.operator_rates.append(rate)
    
    def mutate(self, genome: Genome) -> Genome:
        """Apply all mutations in pipeline."""
        result = genome
        
        for operator, rate in zip(self.operators, self.operator_rates):
            if random.random() < rate:
                result = operator.mutate(result)
        
        return result
    
    def mutate_single(self, genome: Genome) -> Genome:
        """Apply a single random mutation from pipeline."""
        # Normalize rates
        total = sum(self.operator_rates)
        if total == 0:
            return genome
        
        probs = [r / total for r in self.operator_rates]
        
        # Select operator
        operator = random.choices(self.operators, weights=probs)[0]
        return operator.mutate(genome)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about mutation operators."""
        return {
            'num_operators': len(self.operators),
            'operators': [
                {
                    'name': op.get_name(),
                    'description': op.get_description(),
                    'rate': rate
                }
                for op, rate in zip(self.operators, self.operator_rates)
            ]
        }
