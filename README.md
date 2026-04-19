# Helix - Self-Replicating Code Evolution Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Version-1.0.0-orange.svg" alt="Version">
</p>

**Helix** is a biological-inspired genetic programming framework that evolves code through natural selection, mutation, and sexual reproduction. Named after the DNA double helix, it treats code as genetic material that can mutate, recombine, and adapt to solve complex problems.

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Architecture](#architecture)
6. [Genetic Operators](#genetic-operators)
7. [Fitness Evaluation](#fitness-evaluation)
8. [Selection Mechanisms](#selection-mechanisms)
9. [Environment & Tasks](#environment--tasks)
10. [Visualization](#visualization)
11. [Examples](#examples)
12. [API Reference](#api-reference)
13. [Advanced Topics](#advanced-topics)
14. [Results & Benchmarks](#results--benchmarks)

---

## Overview

Helix implements a complete genetic programming system inspired by biological evolution:

- **Genetic Representation**: Code is encoded as a genome consisting of genes, each representing a functional unit
- **Mutation Operators**: Point mutations, insertions, deletions, and structural changes
- **Crossover (Sexual Reproduction)**: Genetic material is exchanged between parent genomes
- **Natural Selection**: Fitter genomes are more likely to reproduce
- **Environmental Pressure**: Tasks define selection pressure and adaptation challenges
- **Evolution Visualization**: Track lineage trees and fitness progression

### Why Helix?

```python
# Traditional approach: Write code by hand
def solve_problem(x):
    # Manual solution requires expertise
    return x**2 + 2*x + 1

# Helix approach: Evolve the solution
engine = EvolutionEngine(config)
best = engine.evolve()  # Let evolution discover the solution
```

---

## Key Concepts

### 1. Genome

The fundamental unit of evolution. A genome represents a complete program:

```
Genome
├── Gene 1 (Sequence)
│   ├── Instruction: ADD
│   └── Instruction: ASSIGN
├── Gene 2 (Branch)
│   └── Instruction: IF
├── Gene 3 (Loop)
│   ├── Instruction: WHILE
│   └── Instruction: PRINT
...
```

Each genome has:
- **Fitness**: How well it solves the problem
- **Generation**: Which evolution generation it belongs to
- **Lineage**: Ancestors that led to this genome
- **Genes**: Functional units encoding behavior

### 2. Gene

A gene is a functional unit within a genome:

| Property | Description |
|----------|-------------|
| `gene_type` | Category (SEQUENCE, BRANCH, LOOP, etc.) |
| `instructions` | Atomic operations |
| `parameters` | Configuration values |
| `fitness_contribution` | Contribution to overall fitness |
| `age` | Generations since last modification |

### 3. Instruction

The smallest unit of execution:

```python
Instruction(
    opcode='add',           # Operation to perform
    operands=(1, 2),         # Input values
    metadata={}             # Additional context
)
```

### 4. Population

A collection of genomes undergoing evolution:

- Maintains diversity through controlled reproduction
- Tracks statistics across generations
- Enforces size constraints
- Manages generational advancement

---

## Installation

### Requirements

- Python 3.8 or higher
- NumPy (for numerical operations)
- No other external dependencies (self-contained)

### Install from Source

```bash
git clone https://github.com/moggan1337/Helix.git
cd Helix
pip install -e .
```

### Quick Test

```bash
python -m src.main --task symbolic --generations 50
```

---

## Quick Start

### Minimal Example

```python
from src.evolution import EvolutionEngine, EvolutionConfig
from src.fitness import fitness_symbolic_regression

# Define training data
x_data = [i * 0.1 for i in range(-50, 51)]
y_data = [x**2 + 2*x + 1 for x in x_data]  # Target: (x+1)^2

# Create fitness function
fitness_fn = fitness_symbolic_regression(x_data, y_data)

# Configure evolution
config = EvolutionConfig(
    population_size=100,
    max_generations=200,
    mutation_rate=0.1,
    crossover_rate=0.7
)

# Run evolution
engine = EvolutionEngine(config)
engine.fitness_evaluator.add_fitness_function(fitness_fn)
best_genome = engine.evolve()

print(f"Best fitness: {best_genome.fitness:.4f}")
print(best_genome.to_code())  # Print evolved code
```

### Command-Line Usage

```bash
# Basic evolution
python -m src.main --generations 100

# With specific task
python -m src.main --task symbolic --generations 200

# Custom population
python -m src.main --population 200 --generations 500

# Export results
python -m src.main \
    --task classification \
    --generations 150 \
    --export-results results.json \
    --export-code solution.py
```

---

## Architecture

### Module Structure

```
Helix/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── genome.py            # Genetic representation
│   ├── mutation.py           # Mutation operators
│   ├── crossover.py          # Crossover operators
│   ├── fitness.py            # Fitness evaluation
│   ├── selection.py          # Selection mechanisms
│   ├── population.py         # Population management
│   ├── environment.py         # Task environments
│   ├── evolution.py           # Main evolution engine
│   ├── visualization.py       # Evolution visualization
│   └── main.py               # CLI entry point
├── examples/
│   ├── symbolic_regression.py
│   ├── classification.py
│   ├── multi_objective.py
│   └── interactive.py
├── tests/
└── docs/
```

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EvolutionEngine                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Selection  │→ │  Crossover  │→ │      Mutation       │  │
│  │  Operator   │  │  Operator   │  │      Pipeline       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         ↓                                       ↓           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Population                       │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │    │
│  │  │ Gen │ │ Gen │ │ Gen │ │ Gen │ │ Gen │  ...       │    │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              FitnessEvaluator                        │    │
│  │         + Environment (Task Distribution)           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Genetic Operators

### Mutation Operators

Mutation introduces genetic variation through various mechanisms:

| Operator | Description | Rate |
|----------|-------------|------|
| `PointMutation` | Changes individual instructions | 10% |
| `InsertionMutation` | Adds new instructions | 5% |
| `DeletionMutation` | Removes instructions | 3% |
| `StructuralMutation` | Duplicates/deletes genes | 2% |
| `ParameterMutation` | Modifies gene parameters | 8% |
| `EnableDisableMutation` | Toggles gene activity | 1% |
| `ScrambleMutation` | Shuffles instruction order | 10% |
| `InversionMutation` | Reverses instruction segments | 5% |

#### Example: Custom Mutation

```python
from src.mutation import MutationPipeline, MutationConfig, PointMutation

config = MutationConfig(
    point_rate=0.15,
    insertion_rate=0.05
)

pipeline = MutationPipeline(config)
pipeline.add_operator(PointMutation(config), rate=1.0)

# Mutate a genome
mutated = pipeline.mutate(genome)
```

### Crossover Operators

Crossover combines genetic material from two parents:

| Operator | Description |
|----------|-------------|
| `SinglePointCrossover` | Single crossover point |
| `TwoPointCrossover` | Two crossover points |
| `UniformCrossover` | Random gene selection |
| `GeneSwapCrossover` | Swaps entire genes |
| `InstructionCrossover` | Instruction-level recombination |
| `ArithmeticCrossover` | Combines numeric parameters |
| `SimulatedBinaryCrossover` | SBX for real-coded GA |
| `BlendCrossover` | BLX-alpha blending |

#### Example: Crossover

```python
from src.crossover import CrossoverPipeline, SinglePointCrossover

pipeline = CrossoverPipeline()
parent1, parent2 = population.select_parents(2)

crossover = SinglePointCrossover()
offspring1, offspring2 = crossover.crossover(parent1, parent2)
```

---

## Fitness Evaluation

Fitness evaluation measures how well a genome solves the problem.

### Predefined Fitness Functions

```python
from src.fitness import (
    fitness_symbolic_regression,
    fitness_classification,
    fitness_knapsack,
    fitness_traveling_salesman,
    fitness_multi_objective,
    fitness_parsimony
)
```

### Custom Fitness Function

```python
def my_fitness(genome, context):
    """Custom fitness evaluation."""
    try:
        result = genome.execute(context)
        
        # Example: reward correct output
        expected = context.get('expected')
        if result == expected:
            return 1.0
        
        # Partial credit for close values
        if isinstance(result, (int, float)):
            error = abs(result - expected)
            return max(0.0, 1.0 - error / abs(expected))
        
        return 0.0
    except Exception:
        return 0.0

engine.fitness_evaluator.add_fitness_function(my_fitness)
```

### Multi-Objective Fitness

```python
from src.fitness import FitnessEvaluator, FitnessConfig, fitness_multi_objective

# Combine multiple objectives
fitness_fn = fitness_multi_objective(
    objectives=[accuracy_fn, simplicity_fn, speed_fn],
    weights=[0.5, 0.3, 0.2]
)
```

---

## Selection Mechanisms

Selection pressure determines which individuals reproduce.

### Selection Methods

| Method | Description |
|--------|-------------|
| `RouletteWheelSelection` | Fitness-proportional |
| `StochasticUniversalSampling` | Deterministic SUS |
| `RankSelection` | Based on rank |
| `TournamentSelection` | k-way tournament |
| `TruncationSelection` | Select top N% |
| `BoltzmannSelection` | Temperature-based |
| `AdaptiveSelection` | Automatic adjustment |

### Example: Tournament Selection

```python
from src.selection import TournamentSelection, SelectionConfig

config = SelectionConfig(
    tournament_size=5,
    tournament_probability=0.9
)

selector = TournamentSelection(config)
selected = selector.select(population)
```

### Selection Pressure

```python
from src.selection import SelectionPressure

pressure = SelectionPressure(
    base_pressure=1.5,
    min_pressure=1.0,
    max_pressure=3.0,
    stagnation_threshold=10,
    diversity_threshold=0.1
)

# Adapt based on population state
new_pressure = pressure.adapt(
    fitness_history=[0.5, 0.55, 0.58, 0.58],
    diversity=0.15
)
```

---

## Environment & Tasks

The environment defines what problems the population must solve.

### Task Types

```python
from src.environment import (
    TaskType,
    DifficultyLevel,
    Task,
    TaskDistribution,
    Environment
)
```

### Creating Custom Tasks

```python
def evaluate_my_task(genome, context):
    """Custom task evaluation."""
    result = genome.execute(context)
    return 1.0 if result == context['expected'] else 0.0

task = Task(
    id="my_task",
    name="My Custom Task",
    task_type=TaskType.OPTIMIZATION,
    description="Solve my specific problem",
    evaluate=evaluate_my_task,
    difficulty=DifficultyLevel.MEDIUM,
    weight=1.0
)

# Add to environment
environment = Environment(name="custom")
environment.task_distribution.add_task(task)
```

### Predefined Task Suites

```python
from src.environment import create_task_suite

# Create multi-task environment
task_dist = create_task_suite(num_tasks=5, adaptive=True)
environment = Environment(task_distribution=task_dist)
```

---

## Visualization

Helix provides comprehensive visualization capabilities.

### Evolution Tree

```python
from src.visualization import EvolutionTree, EvolutionVisualizer

# Track evolution
tree = EvolutionTree()

def track_callback(engine):
    best = engine.population.get_best(1)[0]
    tree.add_genome(best, parent=prev_best)

engine.add_callback(track_callback)

# Generate visualization
visualizer = EvolutionVisualizer(tree)
svg = visualizer.generate_svg(width=1200, height=800)
```

### Statistics Dashboard

```python
# Generate HTML dashboard
html = visualizer.generate_html_dashboard(
    fitness_history=stats.fitness_history,
    diversity_history=stats.diversity_history,
    title="Evolution Dashboard"
)

with open("dashboard.html", "w") as f:
    f.write(html)
```

### ASCII Tree

```python
# Console-friendly tree view
print(visualizer.generate_console_tree(max_depth=10))
```

---

## Examples

### 1. Symbolic Regression

```bash
python examples/symbolic_regression.py
```

Evolves mathematical functions to match data points.

### 2. Classification

```bash
python examples/classification.py
```

Evolves classifiers for categorizing inputs.

### 3. Multi-Objective

```bash
python examples/multi_objective.py
```

Simultaneously optimizes multiple objectives.

### 4. Interactive Demo

```bash
python examples/interactive.py
```

Step-by-step evolution demonstration.

---

## API Reference

### EvolutionEngine

```python
class EvolutionEngine:
    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        environment: Optional[Environment] = None
    )
    
    def evolve(self, max_generations: Optional[int] = None) -> Genome
    def step(self) -> None
    def pause(self) -> None
    def resume(self) -> None
    def stop(self) -> None
    def reset(self) -> None
    def get_statistics(self) -> EvolutionStatistics
    def get_best_genome(self) -> Optional[Genome]
```

### Genome

```python
class Genome:
    genes: List[Gene]
    name: str
    generation: int
    fitness: float
    lineage: List[str]
    
    def mutate(self, mutation_rate: float = 0.1) -> Genome
    def execute(self, context: Optional[Dict] = None) -> Any
    def to_code(self) -> str
    def copy(self) -> Genome
```

### Configuration

```python
@dataclass
class EvolutionConfig:
    population_size: int = 100
    max_generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_count: int = 2
    target_fitness: float = 0.99
    verbose: bool = True
```

---

## Advanced Topics

### Adaptive Mutation

```python
from src.mutation import AdaptiveMutation

# Automatically adjusts mutation based on fitness progress
adaptive_mut = AdaptiveMutation(config, operators=[...])
```

### Parallel Evolution

```python
from src.evolution import EvolutionConfig

config = EvolutionConfig(
    use_multiprocessing=True,
    num_workers=4
)
```

### Custom Gene Types

```python
from src.genome import GeneType, Gene, Instruction

# Define new gene type
class MyGeneType:
    CUSTOM_OPERATION = auto()

# Create custom gene
gene = Gene(
    gene_type=GeneType.SEQUENCE,
    instructions=[Instruction(opcode='my_op', operands=(1, 2))],
    name="custom_gene"
)
```

### Island Model (Distributed Evolution)

```python
# Run multiple populations (islands)
islands = [Population() for _ in range(5)]
for island in islands:
    island.initialize_random()

# Periodically migrate individuals
def migrate(islands):
    for i in range(len(islands)):
        donor = islands[i].get_random(1)[0]
        recipient_idx = (i + 1) % len(islands)
        islands[recipient_idx].add(donor)
```

---

## Results & Benchmarks

### Typical Evolution Progress

| Generation | Best Fitness | Mean Fitness | Diversity |
|------------|--------------|--------------|-----------|
| 0 | 0.12 | 0.08 | 0.85 |
| 25 | 0.45 | 0.32 | 0.72 |
| 50 | 0.68 | 0.51 | 0.58 |
| 75 | 0.82 | 0.64 | 0.45 |
| 100 | 0.91 | 0.73 | 0.38 |
| 125 | 0.95 | 0.78 | 0.31 |
| 150 | 0.97 | 0.82 | 0.28 |

### Convergence Characteristics

- **Fast Initial Progress**: Early generations show rapid fitness improvement
- **Diminishing Returns**: Improvements become smaller as fitness increases
- **Diversity Collapse**: Population diversity decreases as selection dominates
- **Convergence**: Evolution stabilizes when target fitness is reached

### Factors Affecting Evolution

1. **Population Size**: Larger populations → more diversity → slower convergence
2. **Mutation Rate**: Higher rates → more exploration → less exploitation
3. **Selection Pressure**: Higher pressure → faster convergence → premature risk
4. **Crossover Rate**: More crossover → faster information sharing

---

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Helix draws inspiration from:
- John Holland's genetic algorithms
- John Koza's genetic programming
- Biological evolution principles

---

<p align="center">
  Built with ❤️ for evolutionary computation
</p>
