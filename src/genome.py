"""
Genome Module - Genetic Representation of Code

This module implements the core genetic representation system where code is
treated as a biological genome. Each genome consists of genes that encode
instructions, control flow, and data operations.
"""

from __future__ import annotations

import copy
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


class GeneType(Enum):
    """Types of genes that can exist in a genome."""
    
    # Control flow genes
    SEQUENCE = auto()      # Sequential execution
    BRANCH = auto()        # Conditional branching (if/else)
    LOOP = auto()          # Iteration (for/while)
    FUNCTION = auto()      # Function definition
    CALL = auto()          # Function call
    
    # Data genes
    VARIABLE = auto()      # Variable declaration
    CONSTANT = auto()      # Literal value
    ARRAY = auto()         # Array/list operations
    OBJECT = auto()        # Object/dict operations
    
    # Operations genes
    ARITHMETIC = auto()    # Math operations (+, -, *, /)
    LOGICAL = auto()       # Boolean operations (and, or, not)
    COMPARISON = auto()    # Comparison operations (<, >, ==)
    STRING = auto()        # String operations
    
    # I/O genes
    INPUT = auto()         # Input operations
    OUTPUT = auto()        # Print/output operations
    
    # Meta genes
    COMMENT = auto()       # Documentation/comment
    ANNOTATION = auto()    # Type hints, decorators


@dataclass
class Instruction:
    """
    Atomic unit of computation in the genome.
    
    Each instruction represents a basic operation that can be executed
    as part of a gene's behavior.
    """
    
    opcode: str                           # Operation code
    operands: Tuple[Any, ...] = field(default_factory=tuple)  # Input values
    metadata: Dict[str, Any] = field(default_factory=dict)    # Additional info
    
    def __hash__(self) -> int:
        """Hash based on opcode and operands for comparison."""
        key = (self.opcode, self.operands)
        return int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)
    
    def mutate(self, mutation_rate: float = 0.1) -> Instruction:
        """
        Mutate this instruction with given probability.
        
        Args:
            mutation_rate: Probability of each operand mutating
            
        Returns:
            New Instruction (may be same if no mutation occurred)
        """
        if random.random() > mutation_rate:
            return copy.deepcopy(self)
        
        new_operands = list(self.operands)
        
        # Point mutation: change one operand
        if new_operands:
            idx = random.randint(0, len(new_operands) - 1)
            
            mutation_type = random.choice([
                'swap',      # Swap with another operand
                'replace',   # Replace with random value
                'increment', # Add/subtract small amount
                'nullify',   # Set to None
            ])
            
            if mutation_type == 'swap' and len(new_operands) > 1:
                idx2 = random.randint(0, len(new_operands) - 1)
                new_operands[idx], new_operands[idx2] = new_operands[idx2], new_operands[idx]
            elif mutation_type == 'replace':
                new_operands[idx] = self._generate_random_value()
            elif mutation_type == 'increment':
                try:
                    new_operands[idx] = new_operands[idx] + random.uniform(-1, 1)
                except (TypeError, ValueError):
                    new_operands[idx] = self._generate_random_value()
            elif mutation_type == 'nullify':
                new_operands[idx] = None
        
        return Instruction(
            opcode=self.opcode,
            operands=tuple(new_operands),
            metadata=copy.deepcopy(self.metadata)
        )
    
    def _generate_random_value(self) -> Any:
        """Generate a random value for mutation."""
        choice = random.randint(0, 4)
        if choice == 0:
            return random.randint(-100, 100)
        elif choice == 1:
            return random.uniform(-100.0, 100.0)
        elif choice == 2:
            return random.choice([True, False])
        elif choice == 3:
            length = random.randint(1, 10)
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        else:
            return None
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute this instruction in the given context.
        
        Args:
            context: Execution context with variable bindings
            
        Returns:
            Result of execution
        """
        # Built-in operation handlers
        handlers = {
            'add': lambda ctx: ctx.get('_op1', 0) + ctx.get('_op2', 0),
            'sub': lambda ctx: ctx.get('_op1', 0) - ctx.get('_op2', 0),
            'mul': lambda ctx: ctx.get('_op1', 1) * ctx.get('_op2', 1),
            'div': lambda ctx: ctx.get('_op1', 1) / ctx.get('_op2', 1) if ctx.get('_op2', 1) != 0 else 0,
            'mod': lambda ctx: ctx.get('_op1', 0) % ctx.get('_op2', 1),
            'pow': lambda ctx: ctx.get('_op1', 1) ** ctx.get('_op2', 2),
            'and': lambda ctx: ctx.get('_op1', False) and ctx.get('_op2', False),
            'or': lambda ctx: ctx.get('_op1', False) or ctx.get('_op2', False),
            'not': lambda ctx: not ctx.get('_op1', False),
            'eq': lambda ctx: ctx.get('_op1', 0) == ctx.get('_op2', 0),
            'ne': lambda ctx: ctx.get('_op1', 0) != ctx.get('_op2', 0),
            'lt': lambda ctx: ctx.get('_op1', 0) < ctx.get('_op2', 0),
            'le': lambda ctx: ctx.get('_op1', 0) <= ctx.get('_op2', 0),
            'gt': lambda ctx: ctx.get('_op1', 0) > ctx.get('_op2', 0),
            'ge': lambda ctx: ctx.get('_op1', 0) >= ctx.get('_op2', 0),
            'assign': lambda ctx: ctx.get('_value', None),
            'load': lambda ctx: ctx.get('_name', None),
            'store': lambda ctx: ctx.get('_name', None),
            'print': lambda ctx: print(ctx.get('_value', '')),
            'input': lambda ctx: input(str(ctx.get('_prompt', ''))),
            'if': lambda ctx: ctx.get('_condition', False),
            'while': lambda ctx: ctx.get('_condition', False),
            'return': lambda ctx: ctx.get('_value', None),
        }
        
        handler = handlers.get(self.opcode, lambda ctx: None)
        return handler(self.metadata)
    
    def to_code(self, indent: int = 0) -> str:
        """Convert instruction to executable code string."""
        indent_str = '    ' * indent
        op = self.opcode
        
        if op == 'add':
            return f"{indent_str}{self.operands[0] if len(self.operands) > 0 else '_'} + {self.operands[1] if len(self.operands) > 1 else '_'}"
        elif op == 'sub':
            return f"{indent_str}{self.operands[0] if len(self.operands) > 0 else '_'} - {self.operands[1] if len(self.operands) > 1 else '_'}"
        elif op == 'mul':
            return f"{indent_str}{self.operands[0] if len(self.operands) > 0 else '_'} * {self.operands[1] if len(self.operands) > 1 else '_'}"
        elif op == 'div':
            return f"{indent_str}{self.operands[0] if len(self.operands) > 0 else '_'} / {self.operands[1] if len(self.operands) > 1 else '_'}"
        elif op == 'assign':
            return f"{indent_str}{self.operands[0]} = {self.operands[1]}"
        elif op == 'print':
            return f"{indent_str}print({self.operands[0] if self.operands else ''})"
        elif op == 'return':
            return f"{indent_str}return {self.operands[0] if self.operands else ''}"
        elif op == 'if':
            return f"{indent_str}if {self.operands[0]}:"
        elif op == 'while':
            return f"{indent_str}while {self.operands[0]}:"
        else:
            return f"{indent_str}{op}({', '.join(str(o) for o in self.operands)})"


@dataclass
class Gene:
    """
    A gene is a functional unit of the genome that encodes a behavior.
    
    Genes are the primary units of evolution - they can be mutated,
    crossed over, and selected for or against based on fitness.
    """
    
    gene_type: GeneType                    # Type of gene
    instructions: List[Instruction] = field(default_factory=list)  # Instructions
    name: str = ""                         # Gene name (optional)
    parameters: Dict[str, Any] = field(default_factory=dict)       # Gene parameters
    fitness_contribution: float = 0.0      # Gene's contribution to fitness
    age: int = 0                           # Generations since last modification
    enabled: bool = True                   # Whether gene is active
    parent_ids: List[str] = field(default_factory=list)  # Ancestor gene IDs
    
    def __post_init__(self):
        """Generate unique ID after initialization."""
        if not hasattr(self, 'id'):
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique identifier for this gene."""
        content = f"{self.gene_type.name}{self.name}{len(self.instructions)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def complexity(self) -> int:
        """Measure of gene complexity (number of instructions)."""
        return len(self.instructions)
    
    @property
    def is_expressible(self) -> bool:
        """Whether this gene can be expressed as code."""
        return len(self.instructions) > 0 and self.enabled
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """
        Execute all instructions in this gene.
        
        Args:
            context: Execution context with variable bindings
            
        Returns:
            Final result of gene execution
        """
        result = None
        for instruction in self.instructions:
            result = instruction.execute(context)
        return result
    
    def mutate(self, mutation_rate: float = 0.1) -> Gene:
        """
        Create a mutated copy of this gene.
        
        Args:
            mutation_rate: Probability of mutation per instruction
            
        Returns:
            New Gene (potentially mutated)
        """
        new_instructions = [inst.mutate(mutation_rate) for inst in self.instructions]
        
        # Also potentially mutate parameters
        new_params = copy.deepcopy(self.parameters)
        if random.random() < mutation_rate and new_params:
            key = random.choice(list(new_params.keys()))
            new_params[key] = self._randomize_param(new_params[key])
        
        return Gene(
            gene_type=self.gene_type,
            instructions=new_instructions,
            name=self.name,
            parameters=new_params,
            age=0,  # Reset age on mutation
            enabled=self.enabled,
            parent_ids=[self.id] + self.parent_ids[:5]  # Keep last 5 ancestors
        )
    
    def _randomize_param(self, value: Any) -> Any:
        """Randomize a parameter value based on its type."""
        if isinstance(value, int):
            return random.randint(-100, 100)
        elif isinstance(value, float):
            return random.uniform(-100.0, 100.0)
        elif isinstance(value, bool):
            return not value
        elif isinstance(value, str):
            length = random.randint(1, 10)
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        elif isinstance(value, list):
            return [self._randomize_param(v) for v in value[:3]] if value else []
        else:
            return None
    
    def to_code(self, indent: int = 0) -> str:
        """Convert gene to executable code string."""
        if not self.instructions:
            return ""
        
        lines = []
        indent_str = '    ' * indent
        
        # Add gene-specific header
        if self.gene_type == GeneType.FUNCTION:
            params = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
            lines.append(f"{indent_str}def {self.name or 'func'}({params}):")
        elif self.gene_type == GeneType.BRANCH:
            lines.append(f"{indent_str}if {self.parameters.get('condition', 'True')}:")
        elif self.gene_type == GeneType.LOOP:
            lines.append(f"{indent_str}while {self.parameters.get('condition', 'True')}:")
        
        # Add instructions
        for inst in self.instructions:
            lines.append(inst.to_code(indent + 1))
        
        return '\n'.join(lines)
    
    def copy(self) -> Gene:
        """Create a deep copy of this gene."""
        return Gene(
            gene_type=self.gene_type,
            instructions=copy.deepcopy(self.instructions),
            name=self.name,
            parameters=copy.deepcopy(self.parameters),
            fitness_contribution=self.fitness_contribution,
            age=self.age,
            enabled=self.enabled,
            parent_ids=self.parent_ids.copy()
        )


@dataclass
class Genome:
    """
    A genome is a collection of genes that together form a complete program.
    
    The genome is the primary unit of evolution - it has a fitness score,
    can undergo mutation and crossover, and represents a potential solution
    to the evolutionary problem.
    """
    
    genes: List[Gene] = field(default_factory=list)
    name: str = "unnamed"
    generation: int = 0
    fitness: float = 0.0
    lineage: List[str] = field(default_factory=list)  # Ancestor genome IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate unique ID after initialization."""
        if not hasattr(self, 'id'):
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique identifier for this genome."""
        content = f"{self.name}{self.generation}{len(self.genes)}{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    @property
    def complexity(self) -> int:
        """Total complexity (sum of gene complexities)."""
        return sum(g.complexity for g in self.genes)
    
    @property
    def active_genes(self) -> List[Gene]:
        """List of enabled genes."""
        return [g for g in self.genes if g.enabled]
    
    @property
    def is_viable(self) -> bool:
        """Whether this genome has at least one active gene."""
        return len(self.active_genes) > 0
    
    def add_gene(self, gene: Gene) -> None:
        """Add a gene to this genome."""
        self.genes.append(gene)
    
    def remove_gene(self, gene_id: str) -> Optional[Gene]:
        """Remove a gene by ID."""
        for i, gene in enumerate(self.genes):
            if gene.id == gene_id:
                return self.genes.pop(i)
        return None
    
    def get_gene(self, gene_id: str) -> Optional[Gene]:
        """Get a gene by ID."""
        for gene in self.genes:
            if gene.id == gene_id:
                return gene
        return None
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the genome's genes in sequence.
        
        Args:
            context: Execution context for variable bindings
            
        Returns:
            Final result of execution
        """
        if context is None:
            context = {}
        
        result = None
        for gene in self.active_genes:
            result = gene.execute(context)
        return result
    
    def to_code(self) -> str:
        """Convert genome to executable Python code."""
        lines = [
            "#!/usr/bin/env python3",
            "# Generated by Helix Evolution Engine",
            f"# Generation: {self.generation}, Fitness: {self.fitness:.4f}",
            "",
            "def main():",
        ]
        
        for gene in self.active_genes:
            gene_code = gene.to_code(indent=1)
            if gene_code:
                lines.append(gene_code)
        
        lines.extend([
            "",
            "if __name__ == '__main__':",
            "    main()",
        ])
        
        return '\n'.join(lines)
    
    def mutate(self, mutation_rate: float = 0.1) -> Genome:
        """
        Create a mutated copy of this genome.
        
        Mutation operators:
        1. Point mutation: Change individual instructions
        2. Gene mutation: Enable/disable genes
        3. Parameter mutation: Modify gene parameters
        
        Args:
            mutation_rate: Probability of mutation
            
        Returns:
            New Genome (potentially mutated)
        """
        new_genes = [gene.mutate(mutation_rate) for gene in self.genes]
        
        # Structural mutation: potentially add or remove genes
        if random.random() < mutation_rate * 0.5:
            if random.random() < 0.5 and new_genes:
                # Gene deletion (remove a random gene)
                idx = random.randint(0, len(new_genes) - 1)
                new_genes.pop(idx)
            else:
                # Gene duplication (copy a random gene)
                if new_genes:
                    idx = random.randint(0, len(new_genes) - 1)
                    new_genes.append(new_genes[idx].copy())
        
        return Genome(
            genes=new_genes,
            name=self.name,
            generation=self.generation + 1,
            lineage=self.lineage + [self.id],
            metadata=copy.deepcopy(self.metadata)
        )
    
    def copy(self) -> Genome:
        """Create a deep copy of this genome."""
        return Genome(
            genes=[gene.copy() for gene in self.genes],
            name=self.name,
            generation=self.generation,
            fitness=self.fitness,
            lineage=self.lineage.copy(),
            metadata=copy.deepcopy(self.metadata)
        )
    
    @classmethod
    def random(cls, num_genes: int = 5, complexity_per_gene: int = 3) -> Genome:
        """
        Create a random genome with given characteristics.
        
        Args:
            num_genes: Number of genes to generate
            complexity_per_gene: Instructions per gene
            
        Returns:
            New random Genome
        """
        genome = cls()
        
        gene_types = list(GeneType)
        opcodes = [
            'add', 'sub', 'mul', 'div', 'mod', 'pow',
            'and', 'or', 'not', 'eq', 'ne', 'lt', 'le', 'gt', 'ge',
            'assign', 'load', 'store', 'print', 'return'
        ]
        
        for i in range(num_genes):
            gene_type = random.choice(gene_types)
            
            instructions = []
            for _ in range(complexity_per_gene):
                opcode = random.choice(opcodes)
                num_operands = random.randint(0, 3)
                operands = tuple(random.randint(0, 100) for _ in range(num_operands))
                
                instructions.append(Instruction(
                    opcode=opcode,
                    operands=operands
                ))
            
            gene = Gene(
                gene_type=gene_type,
                instructions=instructions,
                name=f"gene_{i}"
            )
            genome.add_gene(gene)
        
        return genome
    
    @classmethod
    def minimal(cls) -> Genome:
        """
        Create a minimal starting genome.
        
        This represents the minimal viable genome from which
        evolution can produce complex solutions.
        """
        genome = cls(name="minimal_seed")
        
        # Single gene with minimal instructions
        gene = Gene(
            gene_type=GeneType.SEQUENCE,
            instructions=[
                Instruction(opcode='assign', operands=('x', 0)),
                Instruction(opcode='add', operands=('_op1', '_op2')),
            ],
            name="seed"
        )
        
        genome.add_gene(gene)
        return genome
    
    def __len__(self) -> int:
        """Number of genes in genome."""
        return len(self.genes)
    
    def __repr__(self) -> str:
        """String representation of genome."""
        return (f"Genome(id={self.id[:8]}, name={self.name}, "
                f"genes={len(self.genes)}, fitness={self.fitness:.4f})")
