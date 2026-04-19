"""
Visualization Module - Evolution Tree and Statistics Visualization

This module provides visualization capabilities for tracking
evolution progress, displaying the evolution tree, and
analyzing genetic lineages.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .genome import Genome


@dataclass
class EvolutionNode:
    """A node in the evolution tree."""
    
    genome_id: str
    genome: Genome
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    
    # Visualization properties
    x: float = 0.0
    y: float = 0.0
    depth: int = 0
    
    # Evolution metrics
    generation: int = 0
    fitness: float = 0.0
    fitness_change: float = 0.0  # Change from parent
    
    # Gene composition
    gene_types: Dict[str, int] = field(default_factory=dict)
    
    # Mutations applied
    mutations_applied: List[str] = field(default_factory=list)
    
    def __hash__(self) -> int:
        return hash(self.genome_id)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EvolutionNode):
            return False
        return self.genome_id == other.genome_id


class EvolutionTree:
    """
    Tracks the evolutionary history as a tree structure.
    
    The tree records each genome's lineage, allowing visualization
    of how solutions evolved over generations.
    """
    
    def __init__(self):
        """Initialize empty evolution tree."""
        self.nodes: Dict[str, EvolutionNode] = {}
        self.root_ids: List[str] = []
        self.generation_nodes: Dict[int, List[str]] = {}
        
        # Statistics
        self.total_nodes = 0
        self.total_generations = 0
    
    def add_genome(
        self,
        genome: Genome,
        parent: Optional[Genome] = None,
        mutations: Optional[List[str]] = None
    ) -> EvolutionNode:
        """
        Add a genome to the evolution tree.
        
        Args:
            genome: Genome to add
            parent: Parent genome (if any)
            mutations: List of mutation types applied
            
        Returns:
            Created EvolutionNode
        """
        # Create node
        node = EvolutionNode(
            genome_id=genome.id,
            genome=genome,
            parent_ids=[parent.id] if parent else [],
            generation=genome.generation,
            fitness=genome.fitness,
            mutations_applied=mutations or []
        )
        
        # Record gene types
        for gene in genome.genes:
            gene_type_name = gene.gene_type.name
            node.gene_types[gene_type_name] = node.gene_types.get(gene_type_name, 0) + 1
        
        # Calculate fitness change from parent
        if parent:
            node.fitness_change = genome.fitness - parent.fitness
        
        # Store node
        self.nodes[genome.id] = node
        self.total_nodes += 1
        
        # Update parent's children
        if parent:
            parent_id = parent.id
            if parent_id in self.nodes:
                if genome.id not in self.nodes[parent_id].children_ids:
                    self.nodes[parent_id].children_ids.append(genome.id)
        
        # Update generation index
        if genome.generation not in self.generation_nodes:
            self.generation_nodes[genome.generation] = []
        self.generation_nodes[genome.generation].append(genome.id)
        
        self.total_generations = max(self.total_generations, genome.generation)
        
        return node
    
    def get_node(self, genome_id: str) -> Optional[EvolutionNode]:
        """Get a node by genome ID."""
        return self.nodes.get(genome_id)
    
    def get_lineage(self, genome_id: str) -> List[EvolutionNode]:
        """
        Get the lineage (ancestors) of a genome.
        
        Returns:
            List of ancestors from root to parent
        """
        lineage = []
        current_id = genome_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            lineage.append(node)
            current_id = node.parent_ids[0] if node.parent_ids else None
        
        return list(reversed(lineage))
    
    def get_descendants(self, genome_id: str) -> List[EvolutionNode]:
        """
        Get all descendants of a genome.
        
        Returns:
            List of all descendant nodes
        """
        descendants = []
        to_visit = [genome_id]
        visited = set()
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            node = self.nodes.get(current_id)
            if not node:
                continue
            
            for child_id in node.children_ids:
                child = self.nodes.get(child_id)
                if child:
                    descendants.append(child)
                    to_visit.append(child_id)
        
        return descendants
    
    def get_generation(self, generation: int) -> List[EvolutionNode]:
        """Get all nodes from a specific generation."""
        node_ids = self.generation_nodes.get(generation, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_best_lineage(self) -> List[EvolutionNode]:
        """
        Get the lineage of the best fitness genome.
        
        Returns:
            List of nodes from root to best genome
        """
        if not self.nodes:
            return []
        
        # Find best node
        best_node = max(self.nodes.values(), key=lambda n: n.fitness)
        return self.get_lineage(best_node.genome_id)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate tree statistics."""
        if not self.nodes:
            return {
                'total_nodes': 0,
                'total_generations': 0,
                'avg_branching_factor': 0.0,
                'max_depth': 0,
                'fitness_improvement': 0.0
            }
        
        # Calculate branching factors
        branching_factors = []
        for node in self.nodes.values():
            branching_factors.append(len(node.children_ids))
        
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        
        # Find max depth
        depths = [n.depth for n in self.nodes.values()]
        max_depth = max(depths) if depths else 0
        
        # Calculate fitness improvement
        fitnesses = [n.fitness for n in self.nodes.values()]
        best_fitness = max(fitnesses) if fitnesses else 0.0
        
        return {
            'total_nodes': self.total_nodes,
            'total_generations': self.total_generations,
            'avg_branching_factor': avg_branching,
            'max_depth': max_depth,
            'best_fitness': best_fitness,
            'avg_fitness': sum(fitnesses) / len(fitnesses)
        }
    
    def prune(self, keep_fraction: float = 0.5) -> None:
        """
        Prune the tree to reduce memory usage.
        
        Args:
            keep_fraction: Fraction of nodes to keep (0-1)
        """
        if len(self.nodes) <= 10:
            return
        
        # Keep only best lineages
        nodes_to_keep = set()
        
        # Keep best genome's lineage
        for node in self.get_best_lineage():
            nodes_to_keep.add(node.genome_id)
        
        # Keep some random nodes from each generation
        for gen, node_ids in self.generation_nodes.items():
            keep_count = max(1, int(len(node_ids) * keep_fraction * 0.3))
            keep_nodes = random.sample(node_ids, min(keep_count, len(node_ids)))
            nodes_to_keep.update(keep_nodes)
        
        # Remove nodes not in keep set
        nodes_to_remove = set(self.nodes.keys()) - nodes_to_keep
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
        
        self.total_nodes = len(self.nodes)


class EvolutionVisualizer:
    """
    Creates visualizations of the evolution process.
    
    Generates various representations of evolution data including:
    - Tree diagrams
    - Fitness graphs
    - Genetic composition charts
    - Animation frames
    """
    
    def __init__(self, tree: Optional[EvolutionTree] = None):
        """Initialize visualizer."""
        self.tree = tree or EvolutionTree()
        
        # Layout settings
        self.node_spacing_x = 100
        self.node_spacing_y = 80
        self.node_radius = 20
    
    def layout_tree(self) -> None:
        """Calculate layout positions for tree nodes."""
        if not self.tree.nodes:
            return
        
        # Group nodes by generation
        generations = sorted(self.tree.generation_nodes.keys())
        
        for gen_idx, generation in enumerate(generations):
            nodes = self.tree.get_generation(generation)
            num_nodes = len(nodes)
            
            # Calculate x positions
            total_width = num_nodes * self.node_spacing_x
            start_x = -total_width / 2
            
            for i, node in enumerate(nodes):
                node.x = start_x + i * self.node_spacing_x + self.node_spacing_x / 2
                node.y = gen_idx * self.node_spacing_y
                node.depth = gen_idx
        
        # Calculate tree depth
        self.tree.total_generations = len(generations) - 1 if generations else 0
    
    def generate_svg(self, width: int = 1200, height: int = 800) -> str:
        """
        Generate SVG representation of the evolution tree.
        
        Args:
            width: SVG width
            height: SVG height
            
        Returns:
            SVG string
        """
        self.layout_tree()
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="100%" height="100%" fill="#1a1a2e"/>',
            f'<defs>',
            f'  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            f'    <polygon points="0 0, 10 3.5, 0 7" fill="#888"/>',
            f'  </marker>',
            f'</defs>'
        ]
        
        # Draw edges
        for node in self.tree.nodes.values():
            for child_id in node.children_ids:
                child = self.tree.nodes.get(child_id)
                if child:
                    svg_parts.append(
                        f'<line x1="{node.x + width/2}" y1="{node.y + 30}" '
                        f'x2="{child.x + width/2}" y2="{child.y}" '
                        f'stroke="#444" stroke-width="1"/>'
                    )
        
        # Draw nodes
        for node in self.tree.nodes.values():
            # Node circle
            color = self._fitness_to_color(node.fitness)
            svg_parts.append(
                f'<circle cx="{node.x + width/2}" cy="{node.y}" r="{self.node_radius}" '
                f'fill="{color}" stroke="#fff" stroke-width="2"/>'
            )
            
            # Fitness label
            svg_parts.append(
                f'<text x="{node.x + width/2}" y="{node.y + 5}" '
                f'text-anchor="middle" fill="#fff" font-size="10">'
                f'{node.fitness:.2f}</text>'
            )
            
            # Generation label
            svg_parts.append(
                f'<text x="{node.x + width/2}" y="{node.y - 30}" '
                f'text-anchor="middle" fill="#888" font-size="9">'
                f'Gen {node.generation}</text>'
            )
        
        # Legend
        svg_parts.extend([
            f'<rect x="20" y="20" width="150" height="120" fill="#2a2a4e" rx="5"/>',
            f'<text x="30" y="40" fill="#fff" font-size="12">Legend</text>',
            f'<circle cx="40" cy="55" r="8" fill="#00ff00"/>',
            f'<text x="55" y="59" fill="#fff" font-size="10">High Fitness</text>',
            f'<circle cx="40" cy="75" r="8" fill="#ffff00"/>',
            f'<text x="55" y="79" fill="#fff" font-size="10">Medium Fitness</text>',
            f'<circle cx="40" cy="95" r="8" fill="#ff0000"/>',
            f'<text x="55" y="99" fill="#fff" font-size="10">Low Fitness</text>',
        ])
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _fitness_to_color(self, fitness: float) -> str:
        """Convert fitness value to color."""
        # Green (good) to Red (bad) gradient
        if fitness >= 0.8:
            return "#00ff00"
        elif fitness >= 0.6:
            ratio = (fitness - 0.6) / 0.2
            return f"#{int(255 * (1 - ratio)):02x}ff00"
        elif fitness >= 0.4:
            ratio = (fitness - 0.4) / 0.2
            return f"#ff{int(255 * ratio):02x}00"
        elif fitness >= 0.2:
            ratio = (fitness - 0.2) / 0.2
            return f"#{int(255 * (1 - ratio)):02x}{int(255 * (1 - ratio)):02x}00"
        else:
            return "#ff0000"
    
    def generate_console_tree(self, max_depth: int = 5, max_width: int = 80) -> str:
        """
        Generate ASCII art representation of the tree.
        
        Args:
            max_depth: Maximum depth to display
            max_width: Maximum width of output
            
        Returns:
            ASCII tree string
        """
        if not self.tree.nodes:
            return "Empty tree"
        
        self.layout_tree()
        
        lines = []
        
        # Group by generation
        generations = sorted(self.tree.generation_nodes.keys())
        
        for gen_idx, generation in enumerate(generations[:max_depth]):
            nodes = self.tree.get_generation(generation)
            
            if gen_idx > 0:
                lines.append("")
            
            lines.append(f"Generation {generation}:")
            
            # Display nodes
            node_strs = []
            for node in nodes[:max_width // 10]:
                fitness_str = f"{node.fitness:.2f}"
                node_strs.append(f"[{fitness_str}]")
            
            lines.append("  " + " ".join(node_strs))
            
            # Draw connections
            if gen_idx < len(generations) - 1 and gen_idx < max_depth - 1:
                next_nodes = self.tree.get_generation(generations[gen_idx + 1])
                if next_nodes:
                    lines.append("  |")
        
        return "\n".join(lines)
    
    def generate_statistics_report(self) -> str:
        """
        Generate a text report of evolution statistics.
        
        Returns:
            Report string
        """
        stats = self.tree.calculate_statistics()
        
        report = [
            "=" * 60,
            "EVOLUTION STATISTICS REPORT",
            "=" * 60,
            "",
            f"Total Nodes:       {stats['total_nodes']}",
            f"Total Generations: {stats['total_generations']}",
            f"Avg Branching:     {stats['avg_branching_factor']:.2f}",
            f"Max Depth:         {stats['max_depth']}",
            "",
            "Fitness Statistics:",
            f"  Best Fitness:     {stats['best_fitness']:.4f}",
            f"  Average Fitness:  {stats['avg_fitness']:.4f}",
            "",
        ]
        
        # Best lineage
        best_lineage = self.tree.get_best_lineage()
        if best_lineage:
            report.append("Best Fitness Lineage:")
            for node in best_lineage:
                report.append(
                    f"  Gen {node.generation}: "
                    f"Fitness = {node.fitness:.4f} "
                    f"(Change: {node.fitness_change:+.4f})"
                )
        
        # Gene composition of best
        if best_lineage:
            best_node = best_lineage[-1]
            if best_node.gene_types:
                report.append("")
                report.append("Gene Composition (Best):")
                for gene_type, count in sorted(best_node.gene_types.items()):
                    report.append(f"  {gene_type}: {count}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def generate_json_export(self) -> str:
        """
        Generate JSON export of the evolution tree.
        
        Returns:
            JSON string
        """
        data = {
            'statistics': self.tree.calculate_statistics(),
            'generations': {},
            'best_lineage': []
        }
        
        # Add generation data
        for gen, node_ids in self.tree.generation_nodes.items():
            gen_nodes = []
            for nid in node_ids:
                node = self.tree.nodes.get(nid)
                if node:
                    gen_nodes.append({
                        'id': node.genome_id,
                        'fitness': node.fitness,
                        'gene_types': node.gene_types,
                        'num_children': len(node.children_ids)
                    })
            data['generations'][gen] = gen_nodes
        
        # Add best lineage
        for node in self.tree.get_best_lineage():
            data['best_lineage'].append({
                'generation': node.generation,
                'fitness': node.fitness,
                'fitness_change': node.fitness_change
            })
        
        return json.dumps(data, indent=2)
    
    def generate_html_dashboard(
        self,
        fitness_history: List[float],
        diversity_history: List[float],
        title: str = "Evolution Dashboard"
    ) -> str:
        """
        Generate HTML dashboard with charts.
        
        Args:
            fitness_history: List of best fitness per generation
            diversity_history: List of diversity per generation
            title: Dashboard title
            
        Returns:
            HTML string
        """
        # Calculate statistics
        stats = self.tree.calculate_statistics()
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #1a1a2e;
            color: #fff;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #2a2a4e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00ff00;
        }}
        .stat-label {{
            color: #888;
            margin-top: 5px;
        }}
        .chart-container {{
            background: #2a2a4e;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        canvas {{
            max-height: 300px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{stats['total_nodes']}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['total_generations']}</div>
                <div class="stat-label">Generations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['best_fitness']:.4f}</div>
                <div class="stat-label">Best Fitness</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['avg_branching_factor']:.2f}</div>
                <div class="stat-label">Avg Branching</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Fitness Over Time</h2>
            <canvas id="fitnessChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Diversity Over Time</h2>
            <canvas id="diversityChart"></canvas>
        </div>
    </div>
    
    <script>
        const fitnessData = {json.dumps(fitness_history)};
        const diversityData = {json.dumps(diversity_history)};
        
        new Chart(document.getElementById('fitnessChart'), {{
            type: 'line',
            data: {{
                labels: fitnessData.map((_, i) => i),
                datasets: [{{
                    label: 'Best Fitness',
                    data: fitnessData,
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ min: 0, max: 1 }}
                }}
            }}
        }});
        
        new Chart(document.getElementById('diversityChart'), {{
            type: 'line',
            data: {{
                labels: diversityData.map((_, i) => i),
                datasets: [{{
                    label: 'Diversity',
                    data: diversityData,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{ min: 0, max: 1 }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html


class AnimationFrameGenerator:
    """Generates animation frames for visualizing evolution."""
    
    def __init__(self, tree: EvolutionTree):
        self.tree = tree
        self.visualizer = EvolutionVisualizer(tree)
    
    def generate_frames(
        self,
        max_frames: int = 100
    ) -> List[str]:
        """
        Generate SVG frames for animation.
        
        Args:
            max_frames: Maximum number of frames to generate
            
        Returns:
            List of SVG strings
        """
        frames = []
        generations = sorted(self.tree.generation_nodes.keys())
        
        # Limit to max_frames
        if len(generations) > max_frames:
            step = len(generations) / max_frames
            gen_indices = [int(i * step) for i in range(max_frames)]
            generations = [generations[i] for i in gen_indices]
        
        for generation in generations:
            # Only show nodes up to this generation
            temp_tree = EvolutionTree()
            
            for gen in generations:
                if gen <= generation:
                    for node in self.tree.get_generation(gen):
                        temp_tree.add_genome(
                            node.genome,
                            parent=temp_tree.nodes.get(node.parent_ids[0]).genome 
                                   if node.parent_ids and node.parent_ids[0] in temp_tree.nodes 
                                   else None
                        )
            
            # Generate frame
            self.visualizer.tree = temp_tree
            svg = self.visualizer.generate_svg()
            frames.append(svg)
        
        return frames
    
    def export_gif(self, filepath: str, fps: int = 2) -> None:
        """
        Export animation as GIF.
        
        Note: Requires PIL/Pillow library.
        
        Args:
            filepath: Output file path
            fps: Frames per second
        """
        try:
            from PIL import Image
            from io import BytesIO
        except ImportError:
            raise ImportError("PIL/Pillow is required for GIF export. Install with: pip install Pillow")
        
        frames = self.generate_frames()
        
        # Convert SVGs to images
        images = []
        for svg in frames:
            # In practice, would need to render SVG to image
            # This is a placeholder - actual implementation would use cairosvg
            pass
        
        # Save GIF
        # images[0].save(filepath, save_all=True, append_images=images[1:], duration=1000//fps)
