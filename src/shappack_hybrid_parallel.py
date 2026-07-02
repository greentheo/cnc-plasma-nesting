#!/usr/bin/env python3
"""
Parallel-enabled hybrid sheet nesting algorithm that combines differential evolution with greedy bottom-left placement.

This optimized module provides a hybrid approach that:
1. Uses DE to optimize the order and rotation of parts
2. Uses a greedy bottom-left algorithm for actual part placement
3. Optimizes for both sheet utilization and compactness
4. Supports efficient parallel processing with multiple workers
"""

import time
import math
import random
import numpy as np
import multiprocessing
from functools import partial
from typing import List, Tuple, Callable, Dict, Optional, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate
import os
import gc

# Import base classes from the original shappack module
from shappack import (
    Point,
    Shape,
    Placement,
    GCodeParser,
    NestingAlgorithm,
    GCodeGenerator,
)


class OptimizationMetrics:
    """Track metrics during optimization."""

    def __init__(
        self,
        generation: int,
        best_fitness: float,
        best_utilization: float,
        best_compactness: float,
        shapes_placed: int,
        total_shapes: int,
        elapsed_time: float,
    ):
        self.generation = generation
        self.best_fitness = best_fitness
        self.best_utilization = utilization = best_utilization
        self.best_compactness = compactness = best_compactness
        self.shapes_placed = shapes_placed
        self.total_shapes = total_shapes
        self.elapsed_time = elapsed_time


# Global dictionary for shared state across processes
_SHARED_STATE = {
    "sheet_width": 0,
    "sheet_height": 0,
    "spacing": 0,
    "grid_size": 10.0,
    "valid_rotations": [0, 30, 60, 90],
    "compactness_weight": 0.3,
    "early_stopping": True,
    "original_shapes": None,
    "eval_count": 0,
}


def _get_sheet_polygon():
    """Create sheet polygon from shared state."""
    width = _SHARED_STATE["sheet_width"]
    height = _SHARED_STATE["sheet_height"]
    return box(0, 0, width, height)


def _create_rotated_polygon(shape, rotation_angle):
    """Create a rotated polygon from a shape."""
    poly = shape.to_shapely_polygon()
    cx, cy = shape.center()
    return rotate(poly, rotation_angle, origin=(cx, cy))


def _fast_check_collision(test_poly, placed_polygons, spacing):
    """
    Fast collision check using bounding box pre-filtering.
    """
    # Get test polygon bounds
    test_bounds = test_poly.bounds

    # Quick check with bounding boxes first
    for placed_poly in placed_polygons:
        placed_bounds = placed_poly.bounds

        # Check if bounding boxes overlap
        if not (
            test_bounds[2] < placed_bounds[0]
            or test_bounds[0] > placed_bounds[2]
            or test_bounds[3] < placed_bounds[1]
            or test_bounds[1] > placed_bounds[3]
        ):
            # Bounding boxes overlap, check actual polygons
            if test_poly.intersects(placed_poly):
                return True

    return False


def _adaptive_greedy_placement(ordered_shapes, rotations):
    """
    Perform adaptive multi-resolution greedy bottom-left placement.
    This is a parallel-friendly version that uses shared state.
    """
    sheet_width = _SHARED_STATE["sheet_width"]
    sheet_height = _SHARED_STATE["sheet_height"]
    spacing = _SHARED_STATE["spacing"]
    grid_size = _SHARED_STATE["grid_size"]
    early_stopping = _SHARED_STATE["early_stopping"]
    sheet_polygon = _get_sheet_polygon()

    placements = []
    placed_polygons = []
    shapes_placed = 0

    # Multi-resolution grid search parameters
    coarse_grid = grid_size * 4  # Very coarse initial grid
    medium_grid = grid_size * 2  # Medium grid for refinement
    fine_grid = grid_size  # Fine grid for final refinement

    total_positions_tested = 0

    # For tracking edge utilization
    edge_usage_left = 0
    edge_usage_bottom = 0

    # Try to place each shape in order
    for shape_idx, (shape, rotation) in enumerate(zip(ordered_shapes, rotations)):
        placed = False
        best_placement = None
        best_position_metric = float("inf")  # Lower is better

        # Create rotated polygon
        rotated_poly = _create_rotated_polygon(shape, rotation)

        # Get bounds of rotated shape
        minx, miny, maxx, maxy = rotated_poly.bounds
        shape_width = maxx - minx
        shape_height = maxy - miny

        # Start with very coarse grid to find general area
        for grid_step in [coarse_grid, medium_grid, fine_grid]:
            # Skip finer grids if we already found a really good position
            if placed and best_position_metric < (sheet_width + sheet_height) * 0.05:
                break

            # Do a sweep from bottom-left to top-right
            for y in np.arange(
                spacing,
                sheet_height - shape_height - spacing,
                grid_step,
            ):
                for x in np.arange(
                    spacing,
                    sheet_width - shape_width - spacing,
                    grid_step,
                ):
                    total_positions_tested += 1

                    # Calculate offset to move shape to this position
                    x_offset = x - minx
                    y_offset = y - miny

                    # Transform polygon to test position
                    test_poly = translate(rotated_poly, xoff=x_offset, yoff=y_offset)

                    # Add spacing buffer
                    buffered_poly = test_poly.buffer(spacing / 2)

                    # Check if it fits in sheet
                    if not sheet_polygon.contains(buffered_poly):
                        continue

                    # Check collision with existing placements
                    if _fast_check_collision(buffered_poly, placed_polygons, spacing):
                        continue

                    # Calculate position metric (prefer bottom-left and edge alignment)
                    # Lower is better
                    position_metric = x + y

                    # Prefer positions that align with existing shapes (edge utilization)
                    edge_alignment_bonus = 0
                    for placed_poly in placed_polygons:
                        if buffered_poly.distance(placed_poly) < spacing * 1.5:
                            edge_alignment_bonus += (
                                10  # Bonus for being close to other parts
                            )

                    # Apply edge alignment bonus
                    position_metric -= edge_alignment_bonus

                    if position_metric < best_position_metric:
                        best_position_metric = position_metric
                        best_placement = Placement(
                            shape=shape,
                            x_offset=x_offset,
                            y_offset=y_offset,
                            rotation=rotation,
                            polygon=test_poly,
                        )
                        placed = True

                        # Early stopping if we find a position in the bottom-left corner
                        if (
                            early_stopping
                            and x < sheet_width * 0.15
                            and y < sheet_height * 0.15
                        ):
                            break

                # Break out of y-loop if we found a really good position
                if placed and early_stopping and x < sheet_width * 0.1:
                    break

            # If we found a placement on this grid level and it's good enough, stop
            if (
                placed
                and early_stopping
                and best_position_metric < (sheet_width + sheet_height) * 0.1
            ):
                break

        # Add best placement if found
        if best_placement:
            placements.append(best_placement)
            placed_polygons.append(best_placement.polygon.buffer(spacing / 2))
            shapes_placed += 1

            # Track edge usage for sheet
            minx, miny, _, _ = best_placement.polygon.bounds
            if minx < sheet_width * 0.1:
                edge_usage_left += 1
            if miny < sheet_height * 0.1:
                edge_usage_bottom += 1

    return placements


def _calculate_metrics(placements):
    """
    Calculate optimization metrics for a set of placements.
    Returns: (utilization, compactness, fitness)
    """
    sheet_width = _SHARED_STATE["sheet_width"]
    sheet_height = _SHARED_STATE["sheet_height"]
    compactness_weight = _SHARED_STATE["compactness_weight"]

    if not placements:
        return 0.0, 0.0, 0.0

    # Calculate sheet utilization
    sheet_area = sheet_width * sheet_height
    total_area_used = sum(p.polygon.area for p in placements)
    utilization = total_area_used / sheet_area if sheet_area > 0 else 0

    # Calculate compactness (inverse of bounding box efficiency)
    try:
        min_x = min(
            min(p.polygon.exterior.coords, key=lambda c: c[0])[0] for p in placements
        )
        min_y = min(
            min(p.polygon.exterior.coords, key=lambda c: c[1])[1] for p in placements
        )
        max_x = max(
            max(p.polygon.exterior.coords, key=lambda c: c[0])[0] for p in placements
        )
        max_y = max(
            max(p.polygon.exterior.coords, key=lambda c: c[1])[1] for p in placements
        )

        bounding_box_area = (max_x - min_x) * (max_y - min_y)
        compactness = (
            total_area_used / bounding_box_area if bounding_box_area > 0 else 0
        )
    except (ValueError, AttributeError):
        compactness = 0

    # Combined fitness function with weighted components
    fitness = (1 - compactness_weight) * utilization + compactness_weight * compactness

    return utilization, compactness, fitness


def _decode_order_vector(vector, shapes):
    """
    Decode the DE solution vector to determine shape order and rotation.
    Returns: (order_rotation_pairs, ordered_shapes)
    """
    valid_rotations = _SHARED_STATE["valid_rotations"]

    # Create pairs of (order_value, shape_idx, rotation_idx)
    placement_data = []
    for i in range(0, len(vector), 2):
        shape_idx = i // 2
        order_value = vector[i]  # Normalized order value (0-1)
        rotation_idx = int(round(vector[i + 1]))
        rotation_idx = max(0, min(rotation_idx, len(valid_rotations) - 1))

        placement_data.append((order_value, shape_idx, rotation_idx))

    # Sort by order value to determine placement order
    placement_data.sort(key=lambda x: x[0])

    # Extract order and rotation information
    order_rotation_pairs = []
    ordered_shapes = []

    for _, shape_idx, rotation_idx in placement_data:
        rotation_angle = valid_rotations[rotation_idx]
        order_rotation_pairs.append((shape_idx, rotation_angle))
        ordered_shapes.append(shapes[shape_idx])

    return order_rotation_pairs, ordered_shapes


def _evaluate_solution_worker(vector):
    """
    Worker function to evaluate a solution vector.
    This function is separate for parallel processing.
    """
    shapes = _SHARED_STATE["original_shapes"]

    # Decode order and rotation
    order_rotation_pairs, _ = _decode_order_vector(vector, shapes)

    # Extract shape indices and rotations in order
    shape_indices = [pair[0] for pair in order_rotation_pairs]
    rotations = [pair[1] for pair in order_rotation_pairs]

    # Get shapes in the optimized order
    ordered_shapes = [shapes[idx] for idx in shape_indices]

    # Perform adaptive greedy placement
    placements = _adaptive_greedy_placement(ordered_shapes, rotations)

    # Calculate metrics
    utilization, compactness, fitness = _calculate_metrics(placements)

    # Return information for reporting
    result = {
        "fitness": fitness,
        "utilization": utilization,
        "compactness": compactness,
        "placements": placements,
        "shape_indices": shape_indices,
        "rotations": rotations,
        "vector": vector,
    }

    return result


class ParallelHybridNestingAlgorithm(NestingAlgorithm):
    """
    Parallel-enabled hybrid nesting algorithm that combines differential evolution with greedy placement.

    This approach:
    1. Uses DE to determine placement order and rotation of shapes
    2. Uses a multi-resolution greedy bottom-left algorithm for actual placement
    3. Considers both sheet utilization and compactness in fitness function
    4. Supports efficient parallel processing with multiple workers
    """

    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        spacing: float = 5.0,
        grid_size: float = 10.0,
        compactness_weight: float = 0.3,
        early_stopping: bool = True,
    ):
        """
        Initialize the hybrid nesting algorithm.

        Args:
            sheet_width: Width of the sheet in mm
            sheet_height: Height of the sheet in mm
            spacing: Minimum spacing between parts in mm
            grid_size: Grid resolution for placement in mm
            compactness_weight: Weight of compactness in fitness function (0-1)
            early_stopping: Whether to use early stopping for placement search
        """
        super().__init__(sheet_width, sheet_height, spacing)
        self.grid_size = grid_size
        self.compactness_weight = compactness_weight
        self.early_stopping = early_stopping
        self.metrics_history = []
        self.best_placements = None
        self.best_utilization = 0
        self.best_compactness = 0
        self.best_fitness = 0
        self.current_generation = 0
        self.start_time = time.time()
        self.eval_count = 0

        # Default rotation increments (can be configured)
        self.valid_rotations = [0, 30, 60, 90]
        self.max_rotation = 90  # Default max rotation angle

        # Update shared state
        global _SHARED_STATE
        _SHARED_STATE["sheet_width"] = sheet_width
        _SHARED_STATE["sheet_height"] = sheet_height
        _SHARED_STATE["spacing"] = spacing
        _SHARED_STATE["grid_size"] = grid_size
        _SHARED_STATE["compactness_weight"] = compactness_weight
        _SHARED_STATE["early_stopping"] = early_stopping
        _SHARED_STATE["valid_rotations"] = self.valid_rotations

    def _initialize_parameters(
        self, rotation_increment: float = 30.0, max_rotation: float = 90.0
    ):
        """
        Initialize optimization parameters.

        Args:
            rotation_increment: Angle increment in degrees for testing rotations
            max_rotation: Maximum rotation angle to test (shapes often symmetric)
        """
        # Store max rotation
        self.max_rotation = max_rotation

        # Generate rotation angles (0 to max_rotation)
        if rotation_increment > 0:
            # For symmetric shapes, use standard angles for better performance
            self.valid_rotations = [0, 30, 60, 90]
            self.valid_rotations = [
                r for r in self.valid_rotations if r <= max_rotation
            ]
        else:
            # Default to standard rotations within the max range
            self.valid_rotations = [
                angle for angle in [0, 30, 60, 90] if angle <= max_rotation
            ]

        # Ensure we have at least [0, max_rotation] if increment doesn't divide evenly
        if max_rotation not in self.valid_rotations:
            self.valid_rotations.append(int(max_rotation))
            self.valid_rotations.sort()

        # Update shared state
        global _SHARED_STATE
        _SHARED_STATE["valid_rotations"] = self.valid_rotations

    def _create_bounds(self, shapes: List[Shape]) -> List[Tuple[float, float]]:
        """
        Create parameter bounds for differential evolution.
        Each shape has 2 parameters: order_index, rotation_index

        Args:
            shapes: List of shapes to optimize

        Returns:
            List of (min, max) bounds for each parameter
        """
        total_parts = len(shapes)
        bounds = []

        for _ in range(total_parts):
            # Order index bounds (normalized 0-1 values)
            bounds.append((0, 1))
            # Rotation index bounds
            bounds.append((0, len(self.valid_rotations) - 1))

        return bounds

    def _handle_worker_result(self, result):
        """
        Process result from a worker.
        """
        self.eval_count += 1

        # Extract metrics
        fitness = result["fitness"]
        utilization = result["utilization"]
        compactness = result["compactness"]
        placements = result["placements"]
        shape_indices = result["shape_indices"]
        rotations = result["rotations"]

        # Log progress periodically
        if self.eval_count % 10 == 0 or fitness > 0.5:
            elapsed = time.time() - self.start_time
            print(
                f"  Solution #{self.eval_count}: "
                f"Fitness={fitness:.4f}, "
                f"Util={utilization:.2%}, "
                f"Compact={compactness:.2%}, "
                f"Parts={len(placements)}/{len(self.original_shapes)}"
            )

        # Track best solution
        if (
            not self.best_placements
            or fitness > self.best_fitness
            or (
                fitness == self.best_fitness
                and len(placements) > len(self.best_placements)
            )
        ):
            self.best_placements = placements
            self.best_utilization = utilization
            self.best_compactness = compactness
            self.best_fitness = fitness
            self.best_shape_indices = shape_indices
            self.best_rotations = rotations

            # Flag for callback to log
            self.new_best_found = True

        return -fitness  # Return negative fitness for minimization

    def _create_objective_function(self, shapes: List[Shape]) -> Callable:
        """
        Create the objective function for optimization.

        Args:
            shapes: List of shapes to optimize

        Returns:
            Objective function that takes a solution vector and returns a score
        """
        # Store shapes in shared state
        global _SHARED_STATE
        _SHARED_STATE["original_shapes"] = shapes
        self.original_shapes = shapes

        # Return a function that evaluates the solution directly
        return lambda vector: self._handle_worker_result(
            _evaluate_solution_worker(vector)
        )

    def _create_callback(self, shapes: List[Shape]) -> Callable:
        """
        Create a callback function to track optimization progress.

        Args:
            shapes: List of shapes to optimize

        Returns:
            Callback function for the optimizer
        """

        def callback(xk, convergence=None):
            """
            Callback function for optimization progress tracking.
            """
            # Update generation counter
            self.current_generation += 1

            # Flag to track if a new best solution was found
            self.new_best_found = False

            # Always collect detailed metrics for better visualization
            # This will make the optimization a bit slower but provide better progress charts

            print(f"\nGeneration {self.current_generation} placement starting...")

            # Evaluate current best solution
            result = _evaluate_solution_worker(xk)

            # Extract metrics
            fitness = result["fitness"]
            utilization = result["utilization"]
            compactness = result["compactness"]
            placements = result["placements"]
            shape_indices = result["shape_indices"]
            rotations = result["rotations"]

            print(f"  Shape order: {shape_indices}")
            print(f"  Rotations: {rotations}")

            # Record metrics
            elapsed = time.time() - self.start_time
            metrics = OptimizationMetrics(
                generation=self.current_generation,
                best_fitness=fitness,
                best_utilization=utilization,
                best_compactness=compactness,
                shapes_placed=len(placements),
                total_shapes=len(shapes),
                elapsed_time=elapsed,
            )
            self.metrics_history.append(metrics)

            # Print progress
            print(
                f"Gen {self.current_generation}: "
                f"Parts={len(placements)}/{len(shapes)}, "
                f"Util={utilization:.2%}, "
                f"Compact={compactness:.2%}, "
                f"Fitness={fitness:.4f}, "
                f"Time={elapsed:.1f}s"
            )

            # Note if we found a new best solution during this generation
            if self.new_best_found:
                print(f"  ★ NEW BEST SOLUTION FOUND THIS GENERATION ★")

            # Save intermediate visualization at milestones
            if self.current_generation == 1 or self.current_generation % 20 == 0:
                viz_dir = "hybrid_progress"
                os.makedirs(viz_dir, exist_ok=True)
                viz_file = f"{viz_dir}/hybrid_gen_{self.current_generation:03d}.png"
                self.visualize_nesting(placements, filename=viz_file)
                print(f"  Saved visualization to {viz_file}")

            # Force garbage collection periodically
            if self.current_generation % 10 == 0:
                gc.collect()

            return False  # Continue optimization

        return callback

    def nest_shapes_hybrid(
        self,
        shapes: List[Shape],
        generations: int = 25,
        population_size: int = 50,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        rotation_increment: float = 30.0,
        max_rotation: float = 90.0,
        seed: Optional[int] = None,
        workers: int = 1,
        verbose: bool = True,
    ) -> Tuple[List[Placement], List[OptimizationMetrics]]:
        """
        Nest shapes using hybrid differential evolution + adaptive greedy placement.

        Args:
            shapes: List of shapes to nest
            generations: Number of generations to run
            population_size: Size of population
            mutation: DE mutation factor (F parameter)
            recombination: DE crossover probability (CR parameter)
            strategy: DE strategy to use
            rotation_increment: Rotation angle increment in degrees
            max_rotation: Maximum rotation to test (usually 90 for symmetric shapes)
            seed: Random seed for reproducibility
            workers: Number of parallel workers
            verbose: Whether to print progress

        Returns:
            Tuple of (placements, metrics_history)
        """
        # Initialize parameters
        self._initialize_parameters(rotation_increment, max_rotation)

        # Reset state
        self.metrics_history = []
        self.best_placements = None
        self.best_utilization = 0
        self.best_compactness = 0
        self.best_fitness = 0
        self.current_generation = 0
        self.start_time = time.time()
        self.eval_count = 0
        self.new_best_found = False

        # Global counter for parallel workers
        global _SHARED_STATE
        _SHARED_STATE["eval_count"] = 0

        if not shapes:
            print("Warning: No shapes to optimize")
            return [], []

        if verbose:
            print(f"\n=== Starting Parallel Hybrid Optimization ===")
            print(f"Sheet size: {self.sheet_width} x {self.sheet_height} mm")
            print(f"Number of shapes: {len(shapes)}")
            print(
                f"Rotation increment: {rotation_increment}° (testing 0-{max_rotation}°)"
            )
            print(f"Valid rotations: {self.valid_rotations}")
            print(f"Population size: {population_size}")
            print(f"Generations: {generations}")
            print(f"Strategy: {strategy}")
            print(f"Mutation: {mutation}")
            print(f"Recombination: {recombination}")
            print(f"Compactness weight: {self.compactness_weight}")
            print(f"Grid size: {self.grid_size} mm")
            print(f"Early stopping: {'Enabled' if self.early_stopping else 'Disabled'}")
            print(f"Parallel workers: {workers}")
            print("=" * 50)

        # Create output directory for visualizations
        os.makedirs("hybrid_progress", exist_ok=True)

        # Create bounds for optimization
        bounds = self._create_bounds(shapes)

        # Create objective function
        objective = self._create_objective_function(shapes)

        # Create callback for progress tracking
        callback = self._create_callback(shapes) if verbose else None

        # Log start of optimization
        print(f"Starting optimization with {len(shapes)} shapes...")
        print(
            f"DNA vector length: {len(bounds)} parameters ({len(bounds) // 2} shapes × 2 parameters)"
        )
        print(f"Valid rotation angles: {self.valid_rotations} degrees")
        print(f"Using {workers} worker(s)")
        print(f"Placement grid size: {self.grid_size}mm")
        print(f"Sheet dimensions: {self.sheet_width:.0f}mm × {self.sheet_height:.0f}mm")
        print(f"Optimization running... progress will be logged regularly")

        try:
            # Run differential evolution with the specified number of workers
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
                workers=workers,  # Use the specified number of workers
                disp=verbose,
                polish=False,
                updating="deferred",  # Required for parallel processing
                atol=0,
                tol=0,
            )

            # Use the best result we found
            if self.best_placements is None:
                # Evaluate final result if no best was recorded
                result = _evaluate_solution_worker(result.x)
                self.best_placements = result["placements"]
                self.best_utilization = result["utilization"]
                self.best_compactness = result["compactness"]
                self.best_fitness = result["fitness"]

            # Report final results
            if verbose:
                print(f"\nOptimization complete!")
                print(f"Final utilization: {self.best_utilization:.2%}")
                print(f"Final compactness: {self.best_compactness:.2%}")
                print(f"Total shapes placed: {len(self.best_placements)}/{len(shapes)}")
                print(f"Total time: {time.time() - self.start_time:.1f} seconds")

            return self.best_placements, self.metrics_history

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            if self.best_placements:
                print(f"Returning best solution found so far...")
                return self.best_placements, self.metrics_history
            return [], []

    def visualize_progress(
        self, metrics_history: List[OptimizationMetrics], filename: Optional[str] = None
    ):
        """
        Visualize the optimization progress over generations.

        Args:
            metrics_history: List of metrics from optimization
            filename: Optional filename to save the plot
        """
        if not metrics_history:
            print("No metrics to visualize")
            return

        generations = [m.generation for m in metrics_history]
        utilizations = [m.best_utilization for m in metrics_history]
        compactness = [m.best_compactness for m in metrics_history]
        fitness = [m.best_fitness for m in metrics_history]
        parts_placed = [m.shapes_placed for m in metrics_history]
        total_parts = metrics_history[0].total_shapes

        # Create figure with multiple subplots - now with 4 plots including fitness
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        # Plot fitness (objective function)
        ax1.plot(generations, fitness, "m-", label="Objective Function")
        ax1.set_ylabel("Fitness Value")
        ax1.set_title("Optimization Progress")
        ax1.grid(True)
        ax1.legend()

        # Plot utilization
        ax2.plot(generations, utilizations, "b-", label="Utilization")
        ax2.set_ylabel("Sheet Utilization (%)")
        ax2.grid(True)
        ax2.legend()

        # Plot compactness
        ax3.plot(generations, compactness, "g-", label="Compactness")
        ax3.set_ylabel("Compactness")
        ax3.grid(True)
        ax3.legend()

        # Plot parts placed with target line
        ax4.plot(generations, parts_placed, "r-", label="Parts Placed")
        ax4.axhline(
            y=total_parts,
            color="gray",
            linestyle="--",
            label=f"Total Parts ({total_parts})",
        )
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Number of Parts")
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Progress visualization saved to {filename}")
        else:
            plt.show()


def nest_gcode_files_parallel(
    files: List[Tuple[str, int]],
    sheet_width_mm: float,
    sheet_height_mm: float,
    spacing_mm: float = 5.0,
    output_file: str = "nested_output_parallel.nc",
    optimization_params: Dict[str, Any] = None,
    visualize: bool = True,
    workers: int = 4,
) -> Tuple[str, float, int]:
    """
    Nest multiple G-code files using parallel hybrid optimization and generate output.

    Args:
        files: List of (filename, count) tuples
        sheet_width_mm: Sheet width in mm
        sheet_height_mm: Sheet height in mm
        spacing_mm: Minimum spacing between parts in mm
        output_file: Output G-code filename
        optimization_params: Parameters for optimization
        visualize: Whether to visualize results
        workers: Number of parallel workers to use

    Returns:
        Tuple of (output_file, utilization, parts_placed)
    """
    print(f"\n{'=' * 70}")
    print(f"PARALLEL HYBRID OPTIMIZATION NESTING")
    print(f"{'=' * 70}")

    print(f"\nSheet: {sheet_width_mm:.0f} x {sheet_height_mm:.0f} mm")
    print(f"Spacing: {spacing_mm} mm")
    print(f"Workers: {workers}")

    # Set default optimization parameters
    if optimization_params is None:
        optimization_params = {
            "generations": 25,
            "population_size": 50,
            "rotation_increment": 30.0,
            "max_rotation": 90.0,
            "compactness_weight": 0.3,
            "grid_size": 10.0,
            "early_stopping": True,
        }

    # Parse files
    parser = GCode
