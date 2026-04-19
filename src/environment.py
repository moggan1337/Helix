"""
Environment Module - Task Distribution and Environmental Pressure

This module implements the environment that applies selective pressure
to the population, including task distributions, changing conditions,
and adaptive difficulty.
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .genome import Genome


class TaskType(Enum):
    """Types of evolutionary tasks."""
    
    # Optimization tasks
    OPTIMIZATION = auto()        # Find optimal solution
    SATISFACTION = auto()        # Satisfy constraints
    MINIMIZATION = auto()        # Minimize value
    MAXIMIZATION = auto()        # Maximize value
    
    # Learning tasks
    CLASSIFICATION = auto()      # Classify inputs
    REGRESSION = auto()          # Predict continuous values
    CLUSTERING = auto()          # Group similar items
    
    # Search tasks
    SEARCH = auto()              # Find target state
    PATHFINDING = auto()         # Find path
    SCHEDULING = auto()          # Optimize schedule
    
    # Creative tasks
    GENERATION = auto()          # Generate artifacts
    COMPOSITION = auto()         # Compose elements
    DESIGN = auto()              # Design solutions


class DifficultyLevel(Enum):
    """Task difficulty levels."""
    
    TRIVIAL = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class Task:
    """A single task in the environment."""
    
    id: str
    name: str
    task_type: TaskType
    description: str
    
    # Evaluation
    evaluate: Callable[[Genome, Dict[str, Any]], float]  # Fitness function
    context_generator: Optional[Callable[[], Dict[str, Any]]] = None  # Generate test context
    
    # Properties
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    weight: float = 1.0                     # Importance weight
    time_limit: Optional[float] = None      # Maximum execution time
    max_complexity: Optional[int] = None    # Maximum genome complexity
    
    # Adaptation
    target_accuracy: float = 0.95           # Target fitness level
    current_accuracy: float = 0.0           # Current best accuracy
    
    def execute(self, genome: Genome) -> Tuple[float, Dict[str, Any]]:
        """
        Execute task and return fitness score.
        
        Returns:
            Tuple of (fitness, execution_context)
        """
        # Generate context if generator provided
        context = self.context_generator() if self.context_generator else {}
        
        # Evaluate genome
        try:
            fitness = self.evaluate(genome, context)
            return max(0.0, min(1.0, fitness)), context
        except Exception as e:
            return 0.0, context


@dataclass
class TaskDistribution:
    """
    Defines the distribution of tasks in the environment.
    
    The task distribution determines which challenges the population
    faces and how selective pressure is applied.
    """
    
    tasks: List[Task] = field(default_factory=list)
    
    # Distribution parameters
    concurrent_tasks: int = 3           # Number of active tasks
    task_switch_interval: Optional[int] = None  # Generations between switches
    
    # Weighting
    uniform_weights: bool = True         # Equal task weights
    performance_based_weights: bool = False  # Weight by performance
    
    # Difficulty scaling
    adaptive_difficulty: bool = True     # Automatically adjust difficulty
    difficulty_increment: float = 0.1    # Difficulty increase rate
    
    # Current state
    current_task_index: int = 0
    generations_on_current_task: int = 0
    
    def add_task(self, task: Task) -> None:
        """Add a task to the distribution."""
        self.tasks.append(task)
    
    def remove_task(self, task_id: str) -> None:
        """Remove a task by ID."""
        self.tasks = [t for t in self.tasks if t.id != task_id]
    
    def get_active_tasks(self) -> List[Task]:
        """Get currently active tasks."""
        if not self.tasks:
            return []
        
        # Select tasks based on weighting
        if self.uniform_weights:
            weights = [1.0] * len(self.tasks)
        elif self.performance_based_weights:
            # Weight by inverse of current accuracy (harder tasks get more weight)
            weights = [1.0 - t.current_accuracy for t in self.tasks]
        else:
            weights = [t.weight for t in self.tasks]
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(self.tasks)] * len(self.tasks)
        
        # Select tasks
        num_tasks = min(self.concurrent_tasks, len(self.tasks))
        selected = random.choices(self.tasks, weights=weights, k=num_tasks)
        
        return selected
    
    def update_task_performance(self, task_id: str, fitness: float) -> None:
        """Update performance tracking for a task."""
        for task in self.tasks:
            if task.id == task_id:
                # Update with exponential moving average
                task.current_accuracy = 0.9 * task.current_accuracy + 0.1 * fitness
                break
    
    def should_switch_task(self) -> bool:
        """Check if environment should switch to different tasks."""
        if self.task_switch_interval is None:
            return False
        
        return self.generations_on_current_task >= self.task_switch_interval
    
    def switch_tasks(self) -> None:
        """Switch to new set of tasks."""
        self.current_task_index = (self.current_task_index + 1) % max(1, len(self.tasks))
        self.generations_on_current_task = 0
    
    def adapt_difficulty(self, population_best_fitness: float) -> None:
        """Adapt task difficulty based on population performance."""
        if not self.adaptive_difficulty:
            return
        
        for task in self.tasks:
            # If population exceeds target accuracy, increase difficulty
            if task.current_accuracy >= task.target_accuracy:
                current_level = task.difficulty.value
                if current_level < DifficultyLevel.MASTER.value:
                    task.difficulty = DifficultyLevel(current_level + 1)
                    # Reset accuracy tracking
                    task.current_accuracy = 0.0
            
            # If population is struggling, decrease difficulty
            elif task.current_accuracy < 0.1 and task.difficulty.value > 0:
                current_level = task.difficulty.value
                task.difficulty = DifficultyLevel(current_level - 1)
                task.current_accuracy = 0.0


class Environment:
    """
    The environment applies selective pressure to the population.
    
    It defines what problems need to be solved, how difficulty
    changes over time, and how success is measured.
    """
    
    def __init__(
        self,
        name: str = "default",
        task_distribution: Optional[TaskDistribution] = None
    ):
        """
        Initialize environment.
        
        Args:
            name: Environment name
            task_distribution: Task distribution configuration
        """
        self.name = name
        self.task_distribution = task_distribution or TaskDistribution()
        
        # Environmental factors
        self.temperature: float = 1.0      # For simulated annealing
        self.pressure: float = 1.0          # Selection pressure
        self.mutation_rate: float = 0.1      # Base mutation rate
        self.crossover_rate: float = 0.7     # Crossover probability
        
        # Change parameters
        self.change_frequency: int = 50      # Generations between changes
        self.change_magnitude: float = 0.2   # How much things change
        
        # Tracking
        self.generation: int = 0
        self.fitness_history: List[float] = []
        self.difficulty_history: List[DifficultyLevel] = []
        
        # Success criteria
        self.target_fitness: float = 0.99
        self.max_generations: int = 1000
    
    def evaluate(self, genome: Genome) -> float:
        """
        Evaluate a genome against the environment.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Fitness score
        """
        active_tasks = self.task_distribution.get_active_tasks()
        
        if not active_tasks:
            return genome.fitness
        
        total_fitness = 0.0
        total_weight = 0.0
        
        for task in active_tasks:
            fitness, context = task.execute(genome)
            
            # Apply environmental modifiers
            fitness *= self.temperature
            fitness = max(0.0, min(1.0, fitness))
            
            total_fitness += fitness * task.weight
            total_weight += task.weight
            
            # Update task performance
            self.task_distribution.update_task_performance(task.id, fitness)
        
        final_fitness = total_fitness / total_weight if total_weight > 0 else 0.0
        
        # Track history
        self.fitness_history.append(final_fitness)
        
        return final_fitness
    
    def evaluate_population(self, population: List[Genome]) -> List[float]:
        """
        Evaluate entire population.
        
        Args:
            population: List of genomes
            
        Returns:
            List of fitness scores
        """
        return [self.evaluate(genome) for genome in population]
    
    def should_evolve(self) -> bool:
        """Check if evolution should continue."""
        # Check if target fitness reached
        if self.fitness_history and self.fitness_history[-1] >= self.target_fitness:
            return False
        
        # Check max generations
        if self.generation >= self.max_generations:
            return False
        
        return True
    
    def step(self) -> None:
        """Advance environment to next generation."""
        self.generation += 1
        self.task_distribution.generations_on_current_task += 1
        
        # Apply environmental changes
        if self.generation % self.change_frequency == 0:
            self._apply_environmental_change()
        
        # Adapt difficulty
        if self.fitness_history:
            self.task_distribution.adapt_difficulty(self.fitness_history[-1])
    
    def _apply_environmental_change(self) -> None:
        """Apply environmental change."""
        # Change mutation rate
        self.mutation_rate *= (1 + random.uniform(-self.change_magnitude, self.change_magnitude))
        self.mutation_rate = max(0.01, min(0.5, self.mutation_rate))
        
        # Change selection pressure
        self.pressure *= (1 + random.uniform(-self.change_magnitude, self.change_magnitude))
        self.pressure = max(1.0, min(3.0, self.pressure))
        
        # Change temperature
        self.temperature *= 0.99
        self.temperature = max(0.1, self.temperature)
        
        # Possibly switch tasks
        if self.task_distribution.should_switch_task():
            self.task_distribution.switch_tasks()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            'name': self.name,
            'generation': self.generation,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'target_fitness': self.target_fitness,
            'num_tasks': len(self.task_distribution.tasks),
            'active_tasks': [t.name for t in self.task_distribution.get_active_tasks()],
            'current_difficulty': self.task_distribution.tasks[0].difficulty.name 
                                  if self.task_distribution.tasks else None,
            'fitness_trend': 'improving' if len(self.fitness_history) >= 2 and 
                             self.fitness_history[-1] > self.fitness_history[-2] else 'stable'
        }


# =============================================================================
# Predefined Task Factories
# =============================================================================

def create_symbolic_regression_task(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
) -> Task:
    """
    Create a symbolic regression task.
    
    The goal is to evolve a function that matches given data points.
    """
    # Generate target function
    if difficulty == DifficultyLevel.TRIVIAL:
        target_fn = lambda x: x + 1
        target_name = "f(x) = x + 1"
    elif difficulty == DifficultyLevel.EASY:
        target_fn = lambda x: 2 * x + 3
        target_name = "f(x) = 2x + 3"
    elif difficulty == DifficultyLevel.MEDIUM:
        target_fn = lambda x: x**2 + 2*x + 1
        target_name = "f(x) = x² + 2x + 1"
    elif difficulty == DifficultyLevel.HARD:
        target_fn = lambda x: (x**3 - 2*x**2 + 3*x - 1) / 3
        target_name = "f(x) = (x³ - 2x² + 3x - 1) / 3"
    else:
        target_fn = lambda x: 1/(1 + 2**(-x))  # Sigmoid
        target_name = "f(x) = sigmoid(x)"
    
    # Generate training data
    x_data = [i * 0.1 for i in range(-50, 51)]
    y_data = [target_fn(x) for x in x_data]
    
    def evaluate(genome: Genome, context: Dict[str, Any]) -> float:
        total_error = 0.0
        for x, y_expected in zip(x_data, y_data):
            try:
                ctx = {'x': x}
                y_actual = genome.execute(ctx)
                if y_actual is not None:
                    total_error += (y_actual - y_expected) ** 2
                else:
                    total_error += y_expected ** 2
            except Exception:
                total_error += y_expected ** 2
        
        # Convert error to fitness
        mean_error = total_error / len(x_data)
        return 1.0 / (1.0 + mean_error)
    
    return Task(
        id=f"symbolic_reg_{difficulty.name}",
        name=f"Symbolic Regression: {target_name}",
        task_type=TaskType.REGRESSION,
        description=f"Approximate the function {target_name}",
        evaluate=evaluate,
        difficulty=difficulty
    )


def create_classification_task(
    num_classes: int = 2,
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
) -> Task:
    """
    Create a classification task.
    
    The goal is to evolve a classifier that correctly categorizes inputs.
    """
    # Generate synthetic data
    if num_classes == 2:
        # Binary classification: separate two clusters
        patterns = []
        for _ in range(100):
            if random.random() < 0.5:
                # Class 0: centered at (0, 0)
                x = random.gauss(0, 0.5)
                y = random.gauss(0, 0.5)
                label = 0
            else:
                # Class 1: centered at (1, 1)
                x = random.gauss(1, 0.5)
                y = random.gauss(1, 0.5)
                label = 1
            patterns.append(((x, y), label))
    else:
        # Multi-class: multiple clusters
        patterns = []
        for i in range(num_classes):
            for _ in range(100 // num_classes):
                angle = 2 * 3.14159 * i / num_classes
                cx, cy = 0.5 * math.cos(angle), 0.5 * math.sin(angle)
                x = random.gauss(cx, 0.2)
                y = random.gauss(cy, 0.2)
                patterns.append(((x, y), i))
    
    def evaluate(genome: Genome, context: Dict[str, Any]) -> float:
        correct = 0
        for (x, y), label in patterns:
            try:
                ctx = {'x': x, 'y': y}
                output = genome.execute(ctx)
                predicted = int(output) % num_classes if output is not None else -1
                if predicted == label:
                    correct += 1
            except Exception:
                pass
        
        return correct / len(patterns)
    
    return Task(
        id=f"classification_{num_classes}_{difficulty.name}",
        name=f"{num_classes}-Class Classification",
        task_type=TaskType.CLASSIFICATION,
        description=f"Classify inputs into {num_classes} categories",
        evaluate=evaluate,
        difficulty=difficulty,
        max_complexity=50
    )


def create_optimization_task(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
) -> Task:
    """
    Create an optimization task.
    
    The goal is to find the maximum of a function.
    """
    if difficulty.value <= DifficultyLevel.EASY.value:
        # Simple function
        def target(x, y):
            return -((x - 1) ** 2 + (y - 1) ** 2)
        bounds = (-5, 5)
    else:
        # More complex function (Rastrigin-like)
        def target(x, y):
            return -(x**2 + y**2 - 10 * (math.cos(2*math.pi*x) + math.cos(2*math.pi*y)))
        bounds = (-5.12, 5.12)
    
    def evaluate(genome: Genome, context: Dict[str, Any]) -> float:
        # Try to find good x, y values
        best_fitness = float('-inf')
        
        # Simple parameter extraction from genome
        params = []
        for gene in genome.genes[:5]:
            for param in gene.parameters.values():
                if isinstance(param, (int, float)):
                    params.append(param)
        
        # If genome has parameters, use them
        if len(params) >= 2:
            x = params[0]
            y = params[1]
            x = max(bounds[0], min(bounds[1], x))
            y = max(bounds[0], min(bounds[1], y))
            return max(0, 1.0 + target(x, y) / 100)
        
        return 0.0
    
    return Task(
        id=f"optimization_{difficulty.name}",
        name="Function Optimization",
        task_type=TaskType.OPTIMIZATION,
        description="Find the maximum of a function",
        evaluate=evaluate,
        difficulty=difficulty
    )


def create_pattern_generation_task(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
) -> Task:
    """
    Create a pattern generation task.
    
    The goal is to evolve a genome that produces a specific pattern.
    """
    # Target pattern (e.g., Fibonacci-like sequence)
    target_pattern = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    target_sum = sum(target_pattern)
    
    def evaluate(genome: Genome, context: Dict[str, Any]) -> float:
        try:
            output = genome.execute({})
            
            if output is None:
                return 0.0
            
            # Convert output to list if needed
            if isinstance(output, (int, float)):
                result = [output]
            else:
                try:
                    result = list(output)
                except Exception:
                    result = [output]
            
            # Compare with target
            score = 0.0
            
            # Check length
            if len(result) == len(target_pattern):
                score += 0.3
            
            # Check sum
            result_sum = sum(result[:len(target_pattern)])
            sum_diff = abs(result_sum - target_sum)
            score += 0.3 * max(0, 1 - sum_diff / target_sum)
            
            # Check individual values
            matches = sum(1 for a, b in zip(result, target_pattern) if a == b)
            score += 0.4 * (matches / len(target_pattern))
            
            return score
            
        except Exception:
            return 0.0
    
    return Task(
        id=f"pattern_gen_{difficulty.name}",
        name="Pattern Generation",
        task_type=TaskType.GENERATION,
        description="Generate the Fibonacci sequence",
        evaluate=evaluate,
        difficulty=difficulty
    )


# Import math for the functions above
import math


def create_task_suite(
    num_tasks: int = 5,
    adaptive: bool = True
) -> TaskDistribution:
    """
    Create a suite of tasks with varying difficulty.
    
    Args:
        num_tasks: Number of tasks to create
        adaptive: Enable adaptive difficulty
        
    Returns:
        TaskDistribution with multiple tasks
    """
    distribution = TaskDistribution(
        adaptive_difficulty=adaptive,
        concurrent_tasks=min(3, num_tasks)
    )
    
    # Add symbolic regression tasks
    for diff in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        if len(distribution.tasks) < num_tasks:
            distribution.add_task(create_symbolic_regression_task(diff))
    
    # Add classification tasks
    if len(distribution.tasks) < num_tasks:
        distribution.add_task(create_classification_task(2, DifficultyLevel.MEDIUM))
    
    # Add optimization tasks
    if len(distribution.tasks) < num_tasks:
        distribution.add_task(create_optimization_task(DifficultyLevel.MEDIUM))
    
    # Add pattern generation
    if len(distribution.tasks) < num_tasks:
        distribution.add_task(create_pattern_generation_task(DifficultyLevel.EASY))
    
    return distribution
