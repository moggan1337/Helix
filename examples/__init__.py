"""
Examples Module - Demonstrations and Use Cases

This module contains example scripts demonstrating various
capabilities of the Helix evolution engine.
"""

from .symbolic_regression import run_symbolic_regression
from .classification import run_classification
from .multi_objective import run_multi_objective
from .interactive import run_interactive_demo

__all__ = [
    'run_symbolic_regression',
    'run_classification',
    'run_multi_objective',
    'run_interactive_demo',
]
