"""
Enhanced Nesting Algorithm with Differential Evolution Support
Extends the original NestingAlgorithm with DE optimization capabilities
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Optional, Dict
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
import matplotlib.patches as mpatches

from shappack import Shape, Placement, NestingAlgorithm
from de_optimizer import DEOptimizer, OptimizationMetrics


class EnhancedNestingAlgorithm(NestingAlgorithm):
    """Extended nesting algorithm with differential evolution optimization"""

    def __init__(self, sheet_width: float, sheet_height: float, spacing: float = 5.0):
        """
        Initialize enhanced nesting algorithm.

        Args:
            sheet_width: Width of the sheet in mm
            sheet_height: Height of the sheet in mm
            spacing: Minimum spacing between shapes in mm
        """
        super().__init__(sheet_width, sheet_height, spacing)
        self.de_optimizer = None
        self.optimization_metrics = []

    def nest_shapes_differential_evolution(
        self,
        shapes: List[Shape],
        max_count: int = 100,
        generations: int = 100,
        population_size: int = 50,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        rotation_increment: float = 15.0,
        seed: Optional[int] = None,
        workers: int = 1,
        verbose: bool = True,
    ) -> Tuple[List[Placement], List[OptimizationMetrics]]:
        """
        Nest shapes using differential evolution optimization.

        Args:
            shapes: List of shapes to nest
            max_count: Maximum number of each shape to place
            generations: Number of DE generations to run
            population_size: Size of DE population
            mutation: DE mutation factor (F parameter)
            recombination: DE crossover probability (CR parameter)
            strategy: DE strategy to use
            rotation_increment: Rotation angle increment in degrees
            seed: Random seed for reproducibility
            workers: Number of parallel workers
            verbose: Whether to print progress

        Returns:
            Tuple of (placements, metrics_history)
        """
        # Initialize DE optimizer
        self.de_optimizer = DEOptimizer(
            sheet_width=self.sheet_width,
            sheet_height=self.sheet_height,
            spacing=self.spacing,
            rotation_increment=rotation_increment,
        )

        if verbose:
            print(f"\n=== Starting Differential Evolution Optimization ===")
            print(f"Sheet size: {self.sheet_width} x {self.sheet_height} mm")
            print(f"Number of unique shapes: {len(shapes)}")
            print(f"Max copies per shape: {max_count}")
            print(
                f"Rotation increment: {rotation_increment}° (angles: {self.de_optimizer.get_rotation_angles()})"
            )
            print(f"Population size: {population_size}")
            print(f"Generations: {generations}")
            print(f"Strategy: {strategy}")
            print(f"Mutation: {mutation}")
            print(f"Recombination: {recombination}")
            print("=" * 50)

        # Run optimization
        placements, metrics = self.de_optimizer.optimize(
            shapes=shapes,
            max_copies_per_shape=max_count,
            generations=generations,
            population_size=population_size,
            mutation=mutation,
            recombination=recombination,
            strategy=strategy,
            seed=seed,
            workers=workers,
            verbose=verbose,
        )

        # Store metrics for later analysis
        self.optimization_metrics = metrics

        return placements, metrics

    def visualize_optimization_progress(
        self,
        metrics_history: List[OptimizationMetrics],
        output_file: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 10),
    ):
        """
        Plot optimization metrics over generations.

        Args:
            metrics_history: List of optimization metrics
            output_file: Optional file path to save the plot
            figsize: Figure size tuple
        """
        if not metrics_history:
            print("No metrics history to visualize")
            return

        generations = [m.generation for m in metrics_history]
        utilizations = [m.best_utilization * 100 for m in metrics_history]
        shapes_placed = [m.shapes_placed for m in metrics_history]
        computation_times = [m.computation_time for m in metrics_history]

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Plot 1: Utilization over generations
        ax1 = axes[0]
        ax1.plot(
            generations,
            utilizations,
            "b-",
            linewidth=2,
            marker="o",
            markersize=3,
            markevery=max(1, len(generations) // 20),
        )
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Sheet Utilization (%)")
        ax1.set_title("Optimization Progress: Sheet Utilization")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # Add best utilization annotation
        best_gen = np.argmax(utilizations)
        ax1.annotate(
            f"Best: {utilizations[best_gen]:.2f}%",
            xy=(generations[best_gen], utilizations[best_gen]),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        # Plot 2: Shapes placed over generations
        ax2 = axes[1]
        ax2.plot(
            generations,
            shapes_placed,
            "g-",
            linewidth=2,
            marker="s",
            markersize=3,
            markevery=max(1, len(generations) // 20),
        )
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Number of Shapes Placed")
        ax2.set_title("Optimization Progress: Shapes Placed")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        # Add max shapes annotation
        max_shapes = metrics_history[0].total_shapes_attempted
        ax2.axhline(
            y=max_shapes,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Max possible: {max_shapes}",
        )
        ax2.legend()

        # Plot 3: Computation time
        ax3 = axes[2]
        ax3.plot(generations, computation_times, "r-", linewidth=2)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Cumulative Time (seconds)")
        ax3.set_title("Computation Time")
        ax3.grid(True, alpha=0.3)

        # Add generation rate annotation
        if len(generations) > 1:
            gen_rate = generations[-1] / computation_times[-1]
            ax3.text(
                0.95,
                0.05,
                f"Avg rate: {gen_rate:.2f} gen/sec",
                transform=ax3.transAxes,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )

        plt.suptitle(
            "Differential Evolution Optimization Progress",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Progress plot saved to: {output_file}")

        plt.show()

    def visualize_nesting_with_stats(
        self,
        placements: List[Placement],
        metrics: Optional[List[OptimizationMetrics]] = None,
        title: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        """
        Enhanced visualization showing placement and statistics.

        Args:
            placements: List of shape placements
            metrics: Optional optimization metrics
            title: Optional plot title
            output_file: Optional file to save plot
        """
        fig, (ax_main, ax_stats) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
        )

        # Main nesting visualization
        ax_main.set_xlim(0, self.sheet_width)
        ax_main.set_ylim(0, self.sheet_height)
        ax_main.set_aspect("equal")
        ax_main.set_xlabel("Width (mm)")
        ax_main.set_ylabel("Height (mm)")

        # Draw sheet boundary
        sheet_rect = plt.Rectangle(
            (0, 0),
            self.sheet_width,
            self.sheet_height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        ax_main.add_patch(sheet_rect)

        # Color map for different shapes
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        # Draw placed shapes
        shape_counts = {}
        total_area = 0

        for i, placement in enumerate(placements):
            # Get color based on original shape index
            shape_idx = placement.shape.original_idx
            if shape_idx not in shape_counts:
                shape_counts[shape_idx] = 0
            shape_counts[shape_idx] += 1

            color = colors[shape_idx % len(colors)]

            # Transform and draw shape
            poly = placement.shape.to_shapely_polygon()
            poly = rotate(poly, placement.rotation, origin="centroid")
            poly = translate(poly, xoff=placement.x_offset, yoff=placement.y_offset)

            coords = list(poly.exterior.coords)
            mpl_poly = MPLPolygon(
                coords, alpha=0.7, facecolor=color, edgecolor="black", linewidth=0.5
            )
            ax_main.add_patch(mpl_poly)

            total_area += poly.area

            # Add rotation indicator for non-zero rotations
            if placement.rotation != 0:
                cx, cy = poly.centroid.x, poly.centroid.y
                ax_main.text(
                    cx,
                    cy,
                    f"{placement.rotation:.0f}°",
                    ha="center",
                    va="center",
                    fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )

        # Title
        if title:
            ax_main.set_title(title)
        else:
            ax_main.set_title(f"Nesting Result: {len(placements)} shapes placed")

        # Statistics panel
        ax_stats.axis("off")
        stats_text = []

        # Calculate utilization
        sheet_area = self.sheet_width * self.sheet_height
        utilization = (total_area / sheet_area) * 100

        stats_text.append("=== NESTING STATISTICS ===\n")
        stats_text.append(
            f"Sheet Size: {self.sheet_width:.1f} x {self.sheet_height:.1f} mm"
        )
        stats_text.append(f"Sheet Area: {sheet_area:.0f} mm²\n")

        stats_text.append(f"Total Shapes Placed: {len(placements)}")
        stats_text.append(f"Unique Shapes: {len(shape_counts)}")
        stats_text.append(f"Total Area Used: {total_area:.0f} mm²")
        stats_text.append(f"Utilization: {utilization:.2f}%\n")

        # Shape counts by type
        stats_text.append("Shapes by Type:")
        for shape_idx, count in sorted(shape_counts.items()):
            stats_text.append(f"  Shape {shape_idx}: {count} copies")

        # Rotation statistics
        rotation_counts = {}
        for p in placements:
            rot = p.rotation
            if rot not in rotation_counts:
                rotation_counts[rot] = 0
            rotation_counts[rot] += 1

        stats_text.append("\nRotation Distribution:")
        for rot, count in sorted(rotation_counts.items()):
            stats_text.append(f"  {rot:.0f}°: {count} shapes")

        # Optimization metrics if available
        if metrics and len(metrics) > 0:
            final_metric = metrics[-1]
            stats_text.append("\n=== OPTIMIZATION METRICS ===")
            stats_text.append(f"Generations Run: {final_metric.generation}")
            stats_text.append(f"Computation Time: {final_metric.computation_time:.1f}s")
            stats_text.append(
                f"Time per Generation: {final_metric.computation_time / final_metric.generation:.2f}s"
            )

            # Find improvement over generations
            if len(metrics) > 1:
                initial_util = metrics[0].best_utilization * 100
                improvement = utilization - initial_util
                stats_text.append(f"Utilization Improvement: {improvement:.2f}%")

        # Display statistics
        stats_str = "\n".join(stats_text)
        ax_stats.text(
            0.05,
            0.95,
            stats_str,
            transform=ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
        )

        plt.suptitle(
            "CNC Shape Nesting with Differential Evolution",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Nesting visualization saved to: {output_file}")

        plt.show()

    def compare_methods(
        self,
        shapes: List[Shape],
        max_count: int = 100,
        de_generations: int = 50,
        de_population: int = 30,
        output_file: Optional[str] = None,
    ):
        """
        Compare grid-based and DE optimization methods.

        Args:
            shapes: List of shapes to nest
            max_count: Maximum copies per shape
            de_generations: Generations for DE
            de_population: Population size for DE
            output_file: Optional output file for comparison plot
        """
        import time

        print("\n=== Method Comparison: Grid vs Differential Evolution ===\n")

        # Run grid-based method
        print("Running grid-based nesting...")
        start_time = time.time()
        grid_placements = self.nest_shapes_optimal(shapes, max_count)
        grid_time = time.time() - start_time

        # Calculate grid utilization
        grid_area = sum(p.polygon.area for p in grid_placements)
        grid_utilization = (grid_area / (self.sheet_width * self.sheet_height)) * 100

        print(
            f"Grid-based: {len(grid_placements)} shapes, {grid_utilization:.2f}% utilization, {grid_time:.1f}s"
        )

        # Run DE method
        print("\nRunning differential evolution nesting...")
        de_placements, de_metrics = self.nest_shapes_differential_evolution(
            shapes,
            max_count,
            generations=de_generations,
            population_size=de_population,
            verbose=False,
        )

        # Calculate DE utilization
        de_area = sum(p.polygon.area for p in de_placements)
        de_utilization = (de_area / (self.sheet_width * self.sheet_height)) * 100
        de_time = de_metrics[-1].computation_time if de_metrics else 0

        print(
            f"DE: {len(de_placements)} shapes, {de_utilization:.2f}% utilization, {de_time:.1f}s"
        )

        # Print comparison
        print("\n=== COMPARISON RESULTS ===")
        print(f"Utilization Improvement: {de_utilization - grid_utilization:+.2f}%")
        print(
            f"Additional Shapes Placed: {len(de_placements) - len(grid_placements):+d}"
        )
        print(f"Time Difference: {de_time - grid_time:+.1f}s")

        if de_utilization > grid_utilization:
            print(
                f"\n✓ Differential Evolution performed better by {de_utilization - grid_utilization:.2f}%"
            )
        else:
            print(
                f"\n✗ Grid-based method performed better by {grid_utilization - de_utilization:.2f}%"
            )

        # Create comparison visualization if requested
        if output_file:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            for ax, placements, method, util, time_taken in [
                (axes[0], grid_placements, "Grid-Based", grid_utilization, grid_time),
                (
                    axes[1],
                    de_placements,
                    "Differential Evolution",
                    de_utilization,
                    de_time,
                ),
            ]:
                ax.set_xlim(0, self.sheet_width)
                ax.set_ylim(0, self.sheet_height)
                ax.set_aspect("equal")
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Height (mm)")
                ax.set_title(
                    f"{method}\n{len(placements)} shapes, {util:.2f}% utilization, {time_taken:.1f}s"
                )

                # Draw sheet
                sheet_rect = plt.Rectangle(
                    (0, 0),
                    self.sheet_width,
                    self.sheet_height,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(sheet_rect)

                # Draw shapes
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
                for placement in placements:
                    shape_idx = placement.shape.original_idx
                    color = colors[shape_idx % len(colors)]

                    coords = list(placement.polygon.exterior.coords)
                    mpl_poly = MPLPolygon(
                        coords,
                        alpha=0.7,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.add_patch(mpl_poly)

            plt.suptitle(
                "Method Comparison: Grid-Based vs Differential Evolution",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"\nComparison plot saved to: {output_file}")
            plt.show()

        return {
            "grid": {
                "placements": grid_placements,
                "utilization": grid_utilization,
                "time": grid_time,
                "count": len(grid_placements),
            },
            "de": {
                "placements": de_placements,
                "utilization": de_utilization,
                "time": de_time,
                "count": len(de_placements),
                "metrics": de_metrics,
            },
        }
