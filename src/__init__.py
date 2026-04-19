"""
Helix - Self-Replicating Code Evolution Engine

A biological-inspired genetic programming system that evolves code through
natural selection, mutation, and sexual reproduction (crossover).
"""

from .genome import Genome, Gene, GeneType, Instruction
from .population import Population
from .evolution import EvolutionEngine
from .fitness import FitnessFunction, FitnessEvaluator
from .selection import SelectionPressure, SelectionMethod
from .mutation import MutationOperator, MutationType
from .crossover import CrossoverOperator, CrossoverType
from .environment import Environment, TaskDistribution
from .visualization import EvolutionTree, EvolutionVisualizer

__version__ = "1.0.0"
__all__ = [
    "Genome",
    "Gene",
    "GeneType", 
    "Instruction",
    "Population",
    "EvolutionEngine",
    "FitnessFunction",
    "FitnessEvaluator",
    "SelectionPressure",
    "SelectionMethod",
    "MutationOperator",
    "MutationType",
    "CrossoverOperator",
    "CrossoverType",
    "Environment",
    "TaskDistribution",
    "EvolutionTree",
    "EvolutionVisualizer",
]
