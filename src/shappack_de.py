#!/usr/bin/env python3
"""
Differential Evolution optimization extension for shappack.py
This module adds DE optimization capabilities to the existing shappack nesting system.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import math
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
from scipy.optimize import differential_evolution

# Import from the existing shappack module
from shappack import (
    Point,
    Shape,
    Placement,
    GCodeParser,
    GCodeGenerator,
    NestingAlgorithm,
)


@dataclass
class DEOptimizationMetrics:
    """Track optimization progress across generations"""

    generation: int
    best_utilization: float
    shapes_placed: int
    total_shapes_attempted: int
    best_vector: np.ndarray
    computation_time: float
    mean_utilization: float = 0.0
    std_utilization: float = 0.0


class DENetworkingAlgorithm(NestingAlgorithm):
    """
    Enhanced nesting algorithm that adds differential evolution optimization
    to the existing NestingAlgorithm class.
    """

    def __init__(self, sheet_width: float, sheet_height: float, spacing: float = 5.0):
        """Initialize the DE-enhanced nesting algorithm."""
        super().__init__(sheet_width, sheet_height, spacing)
        self.de_metrics_history = []
        self.de_best_placement = None
        self.de_current_generation = 0
        self.de_start_time = None
        self.rotation_increment = 15.0  # Default to 15-degree increments
        self.valid_rotations = []
        self.max_rotation = 90  # Only test 0-90 degrees for symmetrical shapes

    def _initialize_de_parameters(
        self, rotation_increment: float = 15.0, max_rotation: float = 90.0
    ):
        """Initialize DE-specific parameters."""
        self.rotation_increment = rotation_increment
        self.max_rotation = max_rotation
        # Only generate rotations from 0 to max_rotation (default 90 for symmetrical shapes)
        self.valid_rotations = list(
            range(0, int(max_rotation) + 1, int(rotation_increment))
        )

    def _create_de_bounds(
        self, shapes: List[Shape], max_copies_per_shape: int
    ) -> List[Tuple[float, float]]:
        """
        Create parameter bounds for differential evolution.
        Each shape instance has 3 parameters: x, y, rotation_index
        """
        total_parts = len(shapes) * max_copies_per_shape
        bounds = []

        for i in range(total_parts):
            # X position bounds
            bounds.append((0, self.sheet_width))
            # Y position bounds
            bounds.append((0, self.sheet_height))
            # Rotation index bounds
            bounds.append((0, len(self.valid_rotations) - 1))

        return bounds

    def _decode_de_vector(
        self, vector: np.ndarray, shapes: List[Shape], max_copies_per_shape: int
    ) -> Tuple[List[Placement], float, int]:
        """
        Decode DE solution vector into shape placements.
        Returns: (placements, utilization, shapes_placed_count)
        """
        placements = []
        placed_polygons = []
        total_area_used = 0

        # Process each potential shape placement
        for i in range(0, len(vector), 3):
            part_index = i // 3
            shape_index = part_index % len(shapes)
            copy_number = part_index // len(shapes)

            if copy_number >= max_copies_per_shape:
                continue

            x = vector[i]
            y = vector[i + 1]
            rotation_index = int(round(vector[i + 2]))
            rotation_index = max(0, min(rotation_index, len(self.valid_rotations) - 1))
            rotation_angle = self.valid_rotations[rotation_index]

            shape = shapes[shape_index]

            # Transform shape polygon
            poly = shape.to_shapely_polygon()

            # Rotate around centroid
            poly = rotate(poly, rotation_angle, origin="centroid")

            # Translate to position
            poly = translate(poly, xoff=x, yoff=y)

            # Check if placement is valid
            if self._is_valid_de_placement(poly, placed_polygons):
                # Create buffered polygon for spacing
                buffered_poly = poly.buffer(self.spacing / 2)
                placed_polygons.append(buffered_poly)
                total_area_used += poly.area

                # Store placement info
                placement = Placement(
                    shape=shape,
                    x_offset=x,
                    y_offset=y,
                    rotation=rotation_angle,
                    polygon=poly,  # Store the transformed polygon
                )
                placements.append(placement)

        sheet_area = self.sheet_width * self.sheet_height
        utilization = total_area_used / sheet_area if sheet_area > 0 else 0

        return placements, utilization, len(placements)

    def _is_valid_de_placement(
        self, polygon: Polygon, placed_polygons: List[Polygon]
    ) -> bool:
        """Check if a polygon placement is valid."""
        # Check if polygon is within sheet bounds
        if not self.sheet_polygon.contains(polygon):
            return False

        # Check for overlaps with existing placements
        for placed in placed_polygons:
            if polygon.intersects(placed):
                return False

        return True

    def _create_de_objective(
        self, shapes: List[Shape], max_copies_per_shape: int
    ) -> Callable:
        """Create the objective function for DE optimization."""

        def objective(vector):
            """Evaluate placement vector, returns negative utilization for minimization."""
            _, utilization, _ = self._decode_de_vector(
                vector, shapes, max_copies_per_shape
            )
            return -utilization

        return objective

    def _create_de_callback(
        self, shapes: List[Shape], max_copies_per_shape: int
    ) -> Callable:
        """Create callback function for tracking optimization progress."""

        def callback(xk, convergence):
            """Called after each generation."""
            self.de_current_generation += 1

            # Decode current best solution
            placements, utilization, shapes_placed = self._decode_de_vector(
                xk, shapes, max_copies_per_shape
            )

            # Calculate time elapsed
            elapsed_time = time.time() - self.de_start_time if self.de_start_time else 0

            # Create metrics entry
            metric = DEOptimizationMetrics(
                generation=self.de_current_generation,
                best_utilization=utilization,
                shapes_placed=shapes_placed,
                total_shapes_attempted=len(shapes) * max_copies_per_shape,
                best_vector=xk.copy(),
                computation_time=elapsed_time,
            )

            self.de_metrics_history.append(metric)

            # Update best placement
            if not self.de_best_placement or utilization > self.de_best_placement[1]:
                self.de_best_placement = (placements, utilization)

            # Print progress
            print(
                f"Generation {self.de_current_generation}: "
                f"Utilization = {utilization:.2%}, "
                f"Shapes placed = {shapes_placed}/{len(shapes) * max_copies_per_shape}, "
                f"Time = {elapsed_time:.1f}s"
            )

            return False  # Don't stop optimization

        return callback

    def nest_shapes_de(
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
    ) -> Tuple[List[Placement], List[DEOptimizationMetrics]]:
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
        # Initialize DE parameters
        self._initialize_de_parameters(rotation_increment)

        # Reset state
        self.de_metrics_history = []
        self.de_best_placement = None
        self.de_current_generation = 0
        self.de_start_time = time.time()

        if not shapes:
            print("Warning: No shapes to optimize")
            return [], []

        if verbose:
            print(f"\n=== Starting Differential Evolution Optimization ===")
            print(f"Sheet size: {self.sheet_width} x {self.sheet_height} mm")
            print(f"Number of unique shapes: {len(shapes)}")
            print(f"Max copies per shape: {max_count}")
            print(
                f"Rotation increment: {rotation_increment}° (testing 0-{self.max_rotation}°)"
            )
            print(f"Rotation angles: {self.valid_rotations}")
            print(f"Population size: {population_size}")
            print(f"Generations: {generations}")
            print(f"Strategy: {strategy}")
            print(f"Mutation: {mutation}")
            print(f"Recombination: {recombination}")
            print("=" * 50)

        # Create bounds for optimization
        bounds = self._create_de_bounds(shapes, max_count)

        # Create objective function
        objective = self._create_de_objective(shapes, max_count)

        # Create callback for progress tracking
        callback = self._create_de_callback(shapes, max_count) if verbose else None

        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            strategy=strategy,
            maxiter=generations,
            popsize=population_size,
            mutation=mutation,
            recombination=recombination,
            callback=callback,
            seed=seed,
            workers=workers,
            disp=verbose,
            polish=False,
            atol=0,
            tol=0,
        )

        # Get final best solution
        final_placements, final_utilization, final_count = self._decode_de_vector(
            result.x, shapes, max_count
        )

        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final utilization: {final_utilization:.2%}")
            print(f"Total shapes placed: {final_count}")
            print(f"Total time: {time.time() - self.de_start_time:.1f} seconds")

        return final_placements, self.de_metrics_history

    def visualize_de_progress(
        self,
        metrics_history: List[DEOptimizationMetrics],
        output_file: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 10),
    ):
        """Plot optimization metrics over generations."""
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


def nest_gcode_files_de(
    gcode_files: List[str],
    output_file: str,
    sheet_width: float,
    sheet_height: float,
    spacing: float = 5.0,
    max_copies: int = 100,
    generations: int = 100,
    population_size: int = 50,
    rotation_increment: float = 15.0,
    verbose: bool = True,
    show_visualization: bool = True,
    show_progress_plot: bool = True,
) -> Tuple[List[Placement], List[DEOptimizationMetrics]]:
    """
    Nest multiple G-code files using differential evolution optimization.

    Args:
        gcode_files: List of G-code file paths
        output_file: Output G-code file path
        sheet_width: Sheet width in mm
        sheet_height: Sheet height in mm
        spacing: Minimum spacing between shapes in mm
        max_copies: Maximum copies of each shape
        generations: Number of DE generations
        population_size: DE population size
        rotation_increment: Rotation angle increment in degrees
        verbose: Print progress
        show_visualization: Show nesting visualization
        show_progress_plot: Show optimization progress plot

    Returns:
        Tuple of (placements, metrics_history)
    """
    # Parse all G-code files
    parser = GCodeParser()
    all_shapes = []

    if verbose:
        print(f"\n=== Parsing {len(gcode_files)} G-code files ===")

    for idx, gcode_file in enumerate(gcode_files):
        if verbose:
            print(f"Parsing: {gcode_file}")

        with open(gcode_file, "r") as f:
            gcode_text = f.read()

        shapes = parser.parse_gcode(gcode_text)

        # Set original index for tracking
        for shape in shapes:
            shape.original_idx = idx

        all_shapes.extend(shapes)
        if verbose:
            print(f"  Found {len(shapes)} shapes")

    if verbose:
        print(f"Total shapes to nest: {len(all_shapes)}")

    # Create DE nesting algorithm
    nesting_algo = DENetworkingAlgorithm(
        sheet_width=sheet_width, sheet_height=sheet_height, spacing=spacing
    )

    # Run DE optimization
    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=max_copies,
        generations=generations,
        population_size=population_size,
        rotation_increment=rotation_increment,
        verbose=verbose,
    )

    # Generate visualization if requested
    if show_visualization:
        viz_file = output_file.replace(".nc", "_de_nesting.png")
        nesting_algo.visualize_nesting(placements, show=True, output_file=viz_file)
        if verbose:
            print(f"Nesting visualization saved to: {viz_file}")

    # Show progress plot if requested
    if show_progress_plot and metrics:
        progress_file = output_file.replace(".nc", "_de_progress.png")
        nesting_algo.visualize_de_progress(metrics, output_file=progress_file)

    # Generate output G-code
    gcode_generator = GCodeGenerator()
    output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
        placements, parser
    )

    # Write output file
    with open(output_file, "w") as f:
        f.write(output_gcode)

    if verbose:
        print(f"\n✓ Output G-code written to: {output_file}")

        # Print final statistics
        if metrics:
            final_metric = metrics[-1]
            sheet_area = sheet_width * sheet_height
            total_area = sum(p.polygon.area for p in placements)
            utilization = (total_area / sheet_area) * 100

            print(f"\n=== FINAL RESULTS ===")
            print(f"Shapes placed: {len(placements)}")
            print(f"Sheet utilization: {utilization:.2f}%")
            print(
                f"Total optimization time: {final_metric.computation_time:.1f} seconds"
            )
            print(f"Generations completed: {final_metric.generation}")

    return placements, metrics


def main():
    """Example usage of DE-enhanced nesting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Nest G-code shapes using differential evolution optimization"
    )

    parser.add_argument("input_files", nargs="+", help="Input G-code file(s) to nest")
    parser.add_argument("-o", "--output", required=True, help="Output G-code file")
    parser.add_argument(
        "-w", "--width", type=float, required=True, help="Sheet width in mm"
    )
    parser.add_argument(
        "-h", "--height", type=float, required=True, help="Sheet height in mm"
    )
    parser.add_argument(
        "-s",
        "--spacing",
        type=float,
        default=5.0,
        help="Minimum spacing between shapes",
    )
    parser.add_argument(
        "-c", "--max-copies", type=int, default=100, help="Maximum copies of each shape"
    )
    parser.add_argument(
        "-g", "--generations", type=int, default=100, help="Number of DE generations"
    )
    parser.add_argument(
        "-p", "--population", type=int, default=50, help="DE population size"
    )
    parser.add_argument(
        "--rotation-increment",
        type=float,
        default=15.0,
        help="Rotation angle increment",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization output"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress plot"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Run DE nesting
    nest_gcode_files_de(
        gcode_files=args.input_files,
        output_file=args.output,
        sheet_width=args.width,
        sheet_height=args.height,
        spacing=args.spacing,
        max_copies=args.max_copies,
        generations=args.generations,
        population_size=args.population,
        rotation_increment=args.rotation_increment,
        verbose=not args.quiet,
        show_visualization=not args.no_viz,
        show_progress_plot=not args.no_progress,
    )


if __name__ == "__main__":
    main()
