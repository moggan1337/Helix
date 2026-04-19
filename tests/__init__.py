"""
Tests Module - Unit Tests for Helix Components

Run tests with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from src.genome import Genome, Gene, GeneType, Instruction
from src.mutation import MutationPipeline, MutationConfig, PointMutation
from src.crossover import CrossoverPipeline, CrossoverConfig, SinglePointCrossover
from src.fitness import FitnessEvaluator, FitnessConfig
from src.selection import TournamentSelection, SelectionConfig
from src.population import Population
from src.environment import Task, TaskDistribution, Environment, TaskType


class TestGenome(unittest.TestCase):
    """Test genome functionality."""
    
    def test_create_empty_genome(self):
        """Test creating an empty genome."""
        genome = Genome()
        self.assertEqual(len(genome.genes), 0)
        self.assertEqual(genome.fitness, 0.0)
        self.assertEqual(genome.generation, 0)
    
    def test_add_gene(self):
        """Test adding genes to genome."""
        genome = Genome()
        gene = Gene(gene_type=GeneType.SEQUENCE, name="test_gene")
        genome.add_gene(gene)
        self.assertEqual(len(genome.genes), 1)
    
    def test_random_genome(self):
        """Test creating random genome."""
        genome = Genome.random(num_genes=5, complexity_per_gene=3)
        self.assertEqual(len(genome.genes), 5)
        self.assertGreater(genome.complexity, 0)
    
    def test_minimal_genome(self):
        """Test minimal genome creation."""
        genome = Genome.minimal()
        self.assertEqual(len(genome.genes), 1)
        self.assertTrue(genome.is_viable)
    
    def test_genome_copy(self):
        """Test genome copying."""
        original = Genome.random(num_genes=3)
        copy = original.copy()
        self.assertEqual(len(copy.genes), len(original.genes))
        self.assertNotEqual(original.id, copy.id)
    
    def test_genome_mutate(self):
        """Test genome mutation."""
        original = Genome.random(num_genes=3)
        mutated = original.mutate(mutation_rate=0.5)
        self.assertIsInstance(mutated, Genome)


class TestGene(unittest.TestCase):
    """Test gene functionality."""
    
    def test_create_gene(self):
        """Test gene creation."""
        gene = Gene(gene_type=GeneType.SEQUENCE, name="test")
        self.assertEqual(gene.name, "test")
        self.assertEqual(gene.gene_type, GeneType.SEQUENCE)
        self.assertTrue(gene.enabled)
    
    def test_gene_instructions(self):
        """Test adding instructions to gene."""
        gene = Gene(gene_type=GeneType.SEQUENCE)
        inst = Instruction(opcode='add', operands=(1, 2))
        gene.instructions.append(inst)
        self.assertEqual(len(gene.instructions), 1)
    
    def test_gene_copy(self):
        """Test gene copying."""
        gene = Gene(gene_type=GeneType.LOOP, name="loop")
        copy = gene.copy()
        self.assertEqual(gene.name, copy.name)
        # ID may be same if content is identical (deterministic hashing)


class TestInstruction(unittest.TestCase):
    """Test instruction functionality."""
    
    def test_create_instruction(self):
        """Test instruction creation."""
        inst = Instruction(opcode='add', operands=(1, 2))
        self.assertEqual(inst.opcode, 'add')
        self.assertEqual(inst.operands, (1, 2))
    
    def test_instruction_mutate(self):
        """Test instruction mutation."""
        inst = Instruction(opcode='add', operands=(1, 2))
        mutated = inst.mutate(mutation_rate=1.0)  # Force mutation
        self.assertIsInstance(mutated, Instruction)
    
    def test_instruction_execute(self):
        """Test instruction execution."""
        inst = Instruction(opcode='add', operands=('_op1', '_op2'))
        inst.metadata = {'_op1': 5, '_op2': 3}
        result = inst.execute(inst.metadata)
        self.assertEqual(result, 8)


class TestMutation(unittest.TestCase):
    """Test mutation operators."""
    
    def test_point_mutation(self):
        """Test point mutation."""
        genome = Genome.random(num_genes=3)
        mutation = PointMutation()
        mutated = mutation.mutate(genome)
        self.assertIsInstance(mutated, Genome)
    
    def test_mutation_pipeline(self):
        """Test mutation pipeline."""
        config = MutationConfig()
        pipeline = MutationPipeline(config)
        genome = Genome.random(num_genes=3)
        mutated = pipeline.mutate(genome)
        self.assertIsInstance(mutated, Genome)


class TestCrossover(unittest.TestCase):
    """Test crossover operators."""
    
    def test_single_point_crossover(self):
        """Test single point crossover."""
        parent1 = Genome.random(num_genes=5)
        parent2 = Genome.random(num_genes=5)
        crossover = SinglePointCrossover()
        child1, child2 = crossover.crossover(parent1, parent2)
        self.assertIsInstance(child1, Genome)
        self.assertIsInstance(child2, Genome)
    
    def test_crossover_pipeline(self):
        """Test crossover pipeline."""
        pipeline = CrossoverPipeline()
        parent1 = Genome.random(num_genes=5)
        parent2 = Genome.random(num_genes=5)
        child1, child2 = pipeline.crossover(parent1, parent2)
        self.assertIsInstance(child1, Genome)


class TestPopulation(unittest.TestCase):
    """Test population management."""
    
    def test_create_population(self):
        """Test population creation."""
        pop = Population(size=50)
        self.assertEqual(pop.target_size, 50)
    
    def test_initialize_random(self):
        """Test random initialization."""
        pop = Population(size=20)
        pop.initialize_random(genes_per_genome=3)
        self.assertEqual(len(pop.individuals), 20)
    
    def test_get_best(self):
        """Test getting best individuals."""
        pop = Population(size=10)
        pop.initialize_random()
        for g in pop.individuals:
            g.fitness = hash(g.id) % 100 / 100
        best = pop.get_best(3)
        self.assertEqual(len(best), 3)
        self.assertGreaterEqual(best[0].fitness, best[1].fitness)
    
    def test_diversity_calculation(self):
        """Test diversity calculation."""
        pop = Population(size=20)
        pop.initialize_random()
        for g in pop.individuals:
            g.fitness = hash(g.id) % 100 / 100
        diversity = pop.calculate_diversity()
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)


class TestSelection(unittest.TestCase):
    """Test selection operators."""
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        pop = Population(size=20)
        pop.initialize_random()
        for g in pop.individuals:
            g.fitness = hash(g.id) % 100 / 100
        
        selector = TournamentSelection()
        selected = selector.select(pop.individuals)
        self.assertEqual(len(selected), len(pop.individuals))


class TestFitness(unittest.TestCase):
    """Test fitness evaluation."""
    
    def test_fitness_evaluator(self):
        """Test fitness evaluator."""
        config = FitnessConfig()
        evaluator = FitnessEvaluator(config)
        
        genome = Genome.random(num_genes=3)
        result = evaluator.evaluate(genome)
        self.assertIsInstance(result.fitness, float)


class TestEnvironment(unittest.TestCase):
    """Test environment functionality."""
    
    def test_task_creation(self):
        """Test task creation."""
        def eval_fn(genome, context):
            return genome.fitness
        
        task = Task(
            id="test",
            name="Test Task",
            task_type=TaskType.OPTIMIZATION,
            description="A test task",
            evaluate=eval_fn
        )
        self.assertEqual(task.name, "Test Task")
    
    def test_task_execution(self):
        """Test task execution."""
        def eval_fn(genome, context):
            return 1.0
        
        task = Task(
            id="test",
            name="Test",
            task_type=TaskType.OPTIMIZATION,
            description="Test",
            evaluate=eval_fn
        )
        genome = Genome()
        genome.fitness = 0.5
        fitness, _ = task.execute(genome)
        self.assertEqual(fitness, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_evolution(self):
        """Test complete evolution cycle."""
        # Create population
        pop = Population(size=20)
        pop.initialize_random(genes_per_genome=3)
        
        # Evaluate
        for g in pop.individuals:
            g.fitness = hash(g.id) % 100 / 100
        
        # Select
        selector = TournamentSelection()
        selected = selector.select(pop.individuals)
        
        # Crossover
        crossover = SinglePointCrossover()
        if len(selected) >= 2:
            child1, child2 = crossover.crossover(selected[0], selected[1])
        
        # Mutate
        pipeline = MutationPipeline()
        mutated = pipeline.mutate(selected[0])
        
        self.assertIsInstance(child1, Genome)
        self.assertIsInstance(mutated, Genome)


if __name__ == '__main__':
    unittest.main()
