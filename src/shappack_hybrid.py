#!/usr/bin/env python3
"""
Hybrid sheet nesting algorithm that combines differential evolution with greedy bottom-left placement.

This module provides a hybrid approach that:
1. Uses DE to optimize the order and rotation of parts
2. Uses a greedy bottom-left algorithm for actual part placement
3. Optimizes for both sheet utilization and compactness
"""

import time
import math
import random
import numpy as np
from typing import List, Tuple, Callable, Dict, Optional, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, box
from shapely.affinity import rotate, translate
import os

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
        elapsed_time: float,
    ):
        self.generation = generation
        self.best_fitness = best_fitness
        self.best_utilization = best_utilization
        self.best_compactness = best_compactness
        self.shapes_placed = shapes_placed
        self.elapsed_time = elapsed_time


class HybridNestingAlgorithm(NestingAlgorithm):
    """
    Hybrid nesting algorithm that combines differential evolution with greedy placement.

    This approach:
    1. Uses DE to determine placement order and rotation of shapes
    2. Uses a greedy bottom-left algorithm for actual placement
    3. Considers both sheet utilization and compactness in fitness function
    """

    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        spacing: float = 5.0,
        grid_size: float = 5.0,
        compactness_weight: float = 0.3,
        external_objective: Callable = None,
    ):
        """
        Initialize the hybrid nesting algorithm.

        Args:
            sheet_width: Width of the sheet in mm
            sheet_height: Height of the sheet in mm
            spacing: Minimum spacing between parts in mm
            grid_size: Grid resolution for placement in mm
            compactness_weight: Weight of compactness in fitness function (0-1)
            external_objective: External objective function for parallel processing
        """
        """
        Initialize the hybrid nesting algorithm.

        Args:
            sheet_width: Width of the sheet in mm
            sheet_height: Height of the sheet in mm
            spacing: Minimum spacing between parts in mm
            grid_size: Grid resolution for placement in mm
            compactness_weight: Weight of compactness in fitness function (0-1)
        """
        super().__init__(sheet_width, sheet_height, spacing)
        self.grid_size = grid_size
        self.compactness_weight = compactness_weight
        self.metrics_history = []
        self.best_placements = None
        self.current_generation = 0
        self.start_time = time.time()
        self.eval_count = 0  # Counter for evaluations

        # Performance optimization
        self.placement_cache = {}  # Cache for placement results

        # Default rotation increments (can be configured)
        self.valid_rotations = [0, 15, 30, 45, 60, 75, 90]
        self.max_rotation = 90  # Default max rotation angle
        self.external_objective = external_objective

    def _initialize_parameters(
        self, rotation_increment: float = 15.0, max_rotation: float = 90.0
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
            # This gives better rotation coverage with fewer angles
            self.valid_rotations = [0, 30, 60, 90]
            self.valid_rotations = [
                r for r in self.valid_rotations if r <= max_rotation
            ]
        else:
            # Default to standard rotations within the max range
            self.valid_rotations = [
                angle for angle in [0, 15, 30, 45, 60, 75, 90] if angle <= max_rotation
            ]

        # Ensure we have at least [0, max_rotation] if increment doesn't divide evenly
        if max_rotation not in self.valid_rotations:
            self.valid_rotations.append(int(max_rotation))
            self.valid_rotations.sort()

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

    def _decode_order_vector(
        self, vector: np.ndarray, shapes: List[Shape]
    ) -> Tuple[List[Tuple[int, int]], List[Shape]]:
        """
        Decode the DE solution vector to determine shape order and rotation.

        Args:
            vector: DE solution vector
            shapes: List of input shapes

        Returns:
            Tuple of (order_rotation_pairs, ordered_shapes)
                order_rotation_pairs: [(shape_idx, rotation_angle), ...]
                ordered_shapes: List of shapes in placement order
        """
        # Create pairs of (order_value, shape_idx, rotation_idx)
        placement_data = []
        for i in range(0, len(vector), 2):
            shape_idx = i // 2
            order_value = vector[i]  # Normalized order value (0-1)
            rotation_idx = int(round(vector[i + 1]))
            rotation_idx = max(0, min(rotation_idx, len(self.valid_rotations) - 1))

            placement_data.append((order_value, shape_idx, rotation_idx))

        # Sort by order value to determine placement order
        placement_data.sort(key=lambda x: x[0])

        # Extract order and rotation information
        order_rotation_pairs = []
        ordered_shapes = []

        for _, shape_idx, rotation_idx in placement_data:
            rotation_angle = self.valid_rotations[rotation_idx]
            order_rotation_pairs.append((shape_idx, rotation_angle))
            ordered_shapes.append(shapes[shape_idx])

        return order_rotation_pairs, ordered_shapes

    def _greedy_bottom_left_placement(
        self, shapes: List[Shape], rotations: List[int], log_step: int = 5
    ) -> List[Placement]:
        """
        Perform greedy bottom-left placement of shapes with specified rotations.

        Args:
            shapes: List of shapes in placement order
            rotations: List of rotation angles for each shape
            log_step: Log progress after this many shapes

        Returns:
            List of final placements
        """
        placements = []
        placed_polygons = []

        # Calculate grid search parameters
        step_size = self.grid_size  # mm - search grid resolution

        # Create a cache key for this set of shapes and rotations
        # Use string representation to avoid hashing issues with multiprocessing
        cache_key = str([(i, r) for i, r in enumerate(rotations)])
        if cache_key in self.placement_cache:
            return self.placement_cache[cache_key]

        # Log start of placement process once
        elapsed = time.time() - self.start_time
        # print(
        #     f"  Placing {len(shapes)} shapes with {len(self.valid_rotations)} possible rotations..."
        # )

        total_positions_tested = 0
        shapes_placed = 0

        for shape_idx, (shape, rotation) in enumerate(zip(shapes, rotations)):
            placed = False
            best_placement = None
            best_position_metric = float("inf")

            # Create rotated polygon
            poly = shape.to_shapely_polygon()
            cx, cy = shape.center()
            rotated_poly = rotate(poly, rotation, origin=(cx, cy))

            # Get bounds of rotated shape
            minx, miny, maxx, maxy = rotated_poly.bounds
            shape_width = maxx - minx
            shape_height = maxy - miny

            # Try positions on a grid (bottom-left search pattern with adaptive step size)
            positions_tested = 0

            # Use larger steps for initial placement attempts
            initial_step = step_size * 3

            # First try with a coarser grid
            for y in np.arange(
                self.spacing,
                self.sheet_height - shape_height - self.spacing,
                initial_step,
            ):
                for x in np.arange(
                    self.spacing,
                    self.sheet_width - shape_width - self.spacing,
                    initial_step,
                ):
                    positions_tested += 1

                    # Calculate offset to move shape to this position
                    x_offset = x - minx
                    y_offset = y - miny

                    # Transform polygon to test position
                    test_poly = translate(rotated_poly, xoff=x_offset, yoff=y_offset)

                    # Add spacing buffer
                    buffered_poly = test_poly.buffer(self.spacing / 2)

                    # Check if it fits in sheet
                    if not self.sheet_polygon.contains(buffered_poly):
                        continue

                    # Check collision with existing placements
                    collision = False
                    for placed_poly in placed_polygons:
                        if buffered_poly.intersects(placed_poly):
                            collision = True
                            break

                    if not collision:
                        # Calculate position metric (prefer bottom-left)
                        # Lower values are better
                        position_metric = x + y

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

                            # Early stopping if we find a good bottom-left position
                            if (
                                x < self.sheet_width * 0.2
                                and y < self.sheet_height * 0.2
                            ):
                                # Skip further grid search
                                y = self.sheet_height  # Force outer loop exit
                                break

                if placed:
                    break

            # If not placed with coarse grid, try again with finer grid
            if not placed:
                for y in np.arange(
                    self.spacing,
                    min(
                        self.sheet_height * 0.5,
                        self.sheet_height - shape_height - self.spacing,
                    ),
                    step_size,
                ):
                    for x in np.arange(
                        self.spacing,
                        min(
                            self.sheet_width * 0.5,
                            self.sheet_width - shape_width - self.spacing,
                        ),
                        step_size,
                    ):
                        positions_tested += 1

                        # Calculate offset to move shape to this position
                        x_offset = x - minx
                        y_offset = y - miny

                        # Transform polygon to test position
                        test_poly = translate(
                            rotated_poly, xoff=x_offset, yoff=y_offset
                        )

                        # Add spacing buffer
                        buffered_poly = test_poly.buffer(self.spacing / 2)

                        # Check if it fits in sheet
                        if not self.sheet_polygon.contains(buffered_poly):
                            continue

                        # Check collision with existing placements
                        collision = False
                        for placed_poly in placed_polygons:
                            if buffered_poly.intersects(placed_poly):
                                collision = True
                                break

                        if not collision:
                            # Calculate position metric (prefer bottom-left)
                            # Lower values are better
                            position_metric = x + y

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

                                # Early stopping if we find a good position
                                if (
                                    x < self.sheet_width * 0.3
                                    and y < self.sheet_height * 0.3
                                ):
                                    break

                    if placed:
                        break

            # Count positions tested but don't log individual placements
            total_positions_tested += positions_tested

            # Add best placement if found
            if best_placement:
                placements.append(best_placement)
                placed_polygons.append(best_placement.polygon.buffer(self.spacing / 2))
                shapes_placed += 1

        # Log summary of placement process
        elapsed = time.time() - self.start_time
        # print(
        #     f"  Placed {shapes_placed}/{len(shapes)} shapes, tested {total_positions_tested} positions, time: {elapsed:.1f}s"
        # )

        # Cache the result before returning
        self.placement_cache[cache_key] = placements

        return placements

    def _calculate_metrics(
        self, placements: List[Placement]
    ) -> Tuple[float, float, float]:
        """
        Calculate optimization metrics for a set of placements.

        Args:
            placements: List of placements

        Returns:
            Tuple of (utilization, compactness, fitness)
        """
        if not placements:
            return 0.0, 0.0, 0.0

        # Calculate sheet utilization
        sheet_area = self.sheet_width * self.sheet_height
        total_area_used = sum(p.polygon.area for p in placements)
        utilization = total_area_used / sheet_area if sheet_area > 0 else 0

        # Calculate compactness (inverse of bounding box efficiency)
        # Get the bounding box of all placements
        if placements:
            all_points = []
            for p in placements:
                all_points.extend(list(p.polygon.exterior.coords))

            if all_points:
                min_x = min(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_x = max(p[0] for p in all_points)
                max_y = max(p[1] for p in all_points)

                bounding_box_area = (max_x - min_x) * (max_y - min_y)
                compactness = (
                    total_area_used / bounding_box_area if bounding_box_area > 0 else 0
                )
            else:
                compactness = 0
        else:
            compactness = 0

        # Combined fitness function with weighted components
        # Higher is better for both metrics
        fitness = (
            1 - self.compactness_weight
        ) * utilization + self.compactness_weight * compactness

        return utilization, compactness, fitness

    def _evaluate_solution(self, vector, shapes):
        """
        Evaluate a solution vector.
        This function is separate to avoid pickling issues with multiprocessing.

        Args:
            vector: Solution vector to evaluate
            shapes: List of shapes to optimize

        Returns:
            Negative fitness value (for minimization)
        """
        # Increment evaluation counter
        self.eval_count += 1

        # Log evaluation information periodically
        if self.eval_count % 10 == 0:
            elapsed = time.time() - self.start_time
            print(f"  Evaluation #{self.eval_count}, time: {elapsed:.1f}s")

        # Decode order and rotation
        order_rotation_pairs, _ = self._decode_order_vector(vector, shapes)

        # Extract shape indices and rotations in order
        shape_indices = [pair[0] for pair in order_rotation_pairs]
        rotations = [pair[1] for pair in order_rotation_pairs]

        # Get shapes in the optimized order
        ordered_shapes = [shapes[idx] for idx in shape_indices]

        # Perform greedy placement
        placements = self._greedy_bottom_left_placement(
            ordered_shapes, rotations, log_step=len(shapes)
        )

        # Calculate metrics
        utilization, compactness, fitness = self._calculate_metrics(placements)

        # Log all solutions with meaningful information
        if self.eval_count % 5 == 0 or fitness > 0.4:
            print(
                f"  Solution #{self.eval_count}: Fitness={fitness:.4f}, Util={utilization:.2%}, "
                f"Compact={compactness:.2%}, Parts={len(placements)}/{len(shapes)}"
            )

        # Return negative fitness for minimization
        return -fitness

    def _create_objective_function(self, shapes: List[Shape]) -> Callable:
        """
        Create the objective function for optimization.

        Args:
            shapes: List of shapes to optimize

        Returns:
            Objective function that takes a solution vector and returns a score
        """
        # Store shapes as instance variable to avoid pickle issues
        self._current_shapes = shapes

        # Use external objective function if provided (for parallel processing)
        if self.external_objective:
            return self.external_objective

        # Return a function that just calls our evaluate method
        return lambda vector: self._evaluate_solution(vector, shapes)

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

            Args:
                xk: Current best solution vector
                convergence: Convergence value
            """
            # Update generation counter
            self.current_generation += 1

            # Always collect metrics for visualization purposes
            print(f"\nGeneration {self.current_generation} placement starting...")

            # Decode order and rotation
            order_rotation_pairs, _ = self._decode_order_vector(xk, shapes)

            # Extract shape indices and rotations in order
            shape_indices = [pair[0] for pair in order_rotation_pairs]
            rotations = [pair[1] for pair in order_rotation_pairs]

            # Get shapes in the optimized order
            ordered_shapes = [shapes[idx] for idx in shape_indices]

            # Show more detailed shape order info for better understanding
            print(f"  Shape order: {shape_indices}")
            print(f"  Rotations: {rotations}")

            # Reset evaluation counter for this generation
            self.eval_count = 0

            # Perform greedy placement
            placements = self._greedy_bottom_left_placement(ordered_shapes, rotations)

            # Calculate metrics
            utilization, compactness, fitness = self._calculate_metrics(placements)

            # Track best placements
            if (
                not self.best_placements
                or utilization > self.best_utilization
                or (
                    utilization == self.best_utilization
                    and compactness > self.best_compactness
                )
            ):
                self.best_placements = placements
                self.best_utilization = utilization
                self.best_compactness = compactness
                self.best_fitness = fitness

                # Log that we found a new best
                print(f"  ★ NEW BEST SOLUTION ★")

            # Record metrics
            elapsed = time.time() - self.start_time
            metrics = OptimizationMetrics(
                generation=self.current_generation,
                best_fitness=fitness,
                best_utilization=utilization,
                best_compactness=compactness,
                shapes_placed=len(placements),
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

            # Save intermediate visualization only for milestone generations
            if self.current_generation == 1 or self.current_generation % 20 == 0:
                viz_file = f"hybrid_gen_{self.current_generation:03d}.png"
                self.visualize_nesting(placements, filename=viz_file)
                print(f"  Saved visualization to {viz_file}")

            # Clear the placement cache periodically to avoid memory issues
            if len(self.placement_cache) > 50:
                print(
                    f"  Clearing placement cache ({len(self.placement_cache)} entries), memory freed: {len(self.placement_cache) * 0.5:.1f}MB"
                )
                self.placement_cache.clear()

            return False  # Continue optimization

        return callback

    def nest_shapes_hybrid(
        self,
        shapes: List[Shape],
        generations: int = 100,
        population_size: int = 50,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        rotation_increment: float = 15.0,
        max_rotation: float = 90.0,  # Maximum rotation angle (default 90 for symmetric shapes)
        seed: Optional[int] = None,
        workers: int = 1,  # Default to 1 worker to avoid pickling issues
        verbose: bool = True,
    ) -> Tuple[List[Placement], List[OptimizationMetrics]]:
        """
        Nest shapes using hybrid differential evolution + greedy bottom-left placement.

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

        if not shapes:
            print("Warning: No shapes to optimize")
            return [], []

        if verbose:
            print(f"\n=== Starting Hybrid Optimization ===")
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
                updating="deferred",  # Use deferred updating for multiprocessing
                atol=0,
                tol=0,
            )
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            if self.best_placements:
                print("Returning best solution found so far...")
                return self.best_placements, self.metrics_history
            return [], []

        # Extract best solution if no callback (or callback didn't save anything)
        if not self.best_placements:
            order_rotation_pairs, _ = self._decode_order_vector(result.x, shapes)

            # Extract shape indices and rotations in order
            shape_indices = [pair[0] for pair in order_rotation_pairs]
            rotations = [pair[1] for pair in order_rotation_pairs]

            # Get shapes in the optimized order
            ordered_shapes = [shapes[idx] for idx in shape_indices]

            # Perform greedy placement with optimal order and rotation
            print("Generating final placement...")
            final_placements = self._greedy_bottom_left_placement(
                ordered_shapes, rotations
            )

            # Calculate final metrics
            utilization, compactness, fitness = self._calculate_metrics(
                final_placements
            )
        else:
            # Use the best placements found during optimization
            final_placements = self.best_placements
            utilization = self.best_utilization
            compactness = self.best_compactness
            fitness = self.best_fitness

        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final utilization: {utilization:.2%}")
            print(f"Final compactness: {compactness:.2%}")
            print(f"Total shapes placed: {len(final_placements)}/{len(shapes)}")
            print(f"Total time: {time.time() - self.start_time:.1f} seconds")

        return final_placements, self.metrics_history

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

        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        # Plot overall fitness/objective function
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

        # Plot parts placed
        ax4.plot(generations, parts_placed, "r-", label="Parts Placed")
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


def nest_gcode_files_hybrid(
    files: List[Tuple[str, int]],
    sheet_width_mm: float,
    sheet_height_mm: float,
    spacing_mm: float = 5.0,
    output_file: str = "nested_output_hybrid.nc",
    optimization_params: Dict[str, Any] = None,
    visualize: bool = True,
) -> Tuple[str, float, int]:
    """
    Nest multiple G-code files using hybrid optimization and generate output.

    Args:
        files: List of (filename, count) tuples
        sheet_width_mm: Sheet width in mm
        sheet_height_mm: Sheet height in mm
        spacing_mm: Minimum spacing between parts in mm
        output_file: Output G-code filename
        optimization_params: Parameters for optimization
        visualize: Whether to visualize results

    Returns:
        Tuple of (output_file, utilization, parts_placed)
    """
    print(f"\n{'=' * 70}")
    print(f"HYBRID OPTIMIZATION NESTING")
    print(f"{'=' * 70}")

    print(f"\nSheet: {sheet_width_mm:.0f} x {sheet_height_mm:.0f} mm")
    print(f"Spacing: {spacing_mm} mm")

    # Set default optimization parameters
    if optimization_params is None:
        optimization_params = {
            "generations": 100,
            "population_size": 50,
            "rotation_increment": 15.0,
            "max_rotation": 90.0,
            "compactness_weight": 0.3,
            "grid_size": 5.0,
        }

    # Parse files
    parser = GCodeParser()
    all_shapes = []
    original_gcodes = []

    print("\nParsing G-code files...")
    for idx, (file_path, max_count) in enumerate(files):
        if not file_path:
            print(f"⚠️ Invalid file path")
            continue

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        print(f"✓ Found: {file_path}")
        with open(file_path, "r") as f:
            gcode_text = f.read()

        original_gcodes.append(gcode_text)
        shapes = parser.parse_gcode(gcode_text)
        print(f"  Parsed {len(shapes)} shape(s), requesting {max_count} copies")

        # Add shapes with original index for tracking
        for shape in shapes:
            for _ in range(max_count):
                shape_copy = type(shape)(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=idx,
                )
                all_shapes.append(shape_copy)

    if not all_shapes:
        print("No shapes found!")
        return output_file, 0.0, 0

    print(f"\nTotal shape instances to place: {len(all_shapes)}")

    # Create hybrid nesting algorithm
    nesting_algo = HybridNestingAlgorithm(
        sheet_width=sheet_width_mm,
        sheet_height=sheet_height_mm,
        spacing=spacing_mm,
        grid_size=optimization_params.get("grid_size", 5.0),
        compactness_weight=optimization_params.get("compactness_weight", 0.3),
    )

    # Run hybrid optimization
    print("\n" + "=" * 70)
    print("RUNNING HYBRID OPTIMIZATION")
    print("=" * 70)

    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_hybrid(
        shapes=all_shapes,
        generations=optimization_params.get("generations", 100),
        population_size=optimization_params.get("population_size", 50),
        rotation_increment=optimization_params.get("rotation_increment", 15.0),
        max_rotation=optimization_params.get("max_rotation", 90.0),
        workers=1,  # Use single worker to avoid pickling issues
        verbose=True,
    )

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if placements:
        sheet_area = sheet_width_mm * sheet_height_mm
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        print(f"Shapes placed: {len(placements)} / {len(all_shapes)}")
        print(f"Utilization: {utilization:.2f}%")
        print(f"Time: {elapsed:.1f} seconds")

        # Save visualization
        if visualize:
            viz_file = "nested_output_hybrid.png"
            nesting_algo.visualize_nesting(placements, filename=viz_file)
            print(f"\n✓ Visualization saved to: {viz_file}")

            # Also save progress visualization
            progress_file = "hybrid_optimization_progress.png"
            nesting_algo.visualize_progress(metrics, filename=progress_file)
            print(f"✓ Progress visualization saved to: {progress_file}")

        # Generate G-code output
        print("\nGenerating G-code output...")
        gcode_generator = GCodeGenerator()

        try:
            # Generate the nested G-code with optimal travel
            output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                original_gcodes, placements, parser
            )

            # Save the G-code to file
            with open(output_file, "w") as f:
                f.write(output_gcode)

            print(f"✓ G-code saved to: {output_file}")

            # Show some statistics about the output
            lines = output_gcode.split("\n")
            print(f"  Total G-code lines: {len(lines)}")
            print(f"  Total shapes in output: {len(placements)}")

            return output_file, utilization, len(placements)

        except Exception as e:
            print(f"⚠️ G-code generation encountered an issue: {str(e)}")
            print("  Note: The shapes were still successfully nested and visualized.")

    else:
        print("No shapes could be placed!")

    return output_file, 0.0, 0


def main():
    """Example of using hybrid nesting algorithm."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Hybrid nesting optimization for G-code files"
    )
    parser.add_argument("files", nargs="*", help="G-code files to nest")
    parser.add_argument(
        "--copies", type=int, default=1, help="Number of copies per file"
    )
    parser.add_argument(
        "--width", type=float, default=48 * 25.4, help="Sheet width in mm"
    )
    parser.add_argument(
        "--height", type=float, default=96 * 25.4, help="Sheet height in mm"
    )
    parser.add_argument(
        "--spacing", type=float, default=5, help="Spacing between parts in mm"
    )
    parser.add_argument(
        "--generations", type=int, default=100, help="Number of generations"
    )
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument(
        "--rot-step", type=float, default=15, help="Rotation step in degrees"
    )
    parser.add_argument(
        "--max-rotation",
        type=float,
        default=90,
        help="Maximum rotation angle (degrees)",
    )
    parser.add_argument(
        "--compactness", type=float, default=0.3, help="Compactness weight (0-1)"
    )
    parser.add_argument(
        "--grid", type=float, default=5, help="Grid size for placement in mm"
    )
    parser.add_argument(
        "--output", type=str, default="nested_output_hybrid.nc", help="Output file"
    )

    args = parser.parse_args()

    # If no files provided, show example
    if not args.files:
        print("No files provided. Please specify G-code files to nest.")
        print("Example: python shappack_hybrid.py file1.nc file2.nc --copies 5")
        return

    # Check if files exist
    files = [(f, args.copies) for f in args.files if os.path.exists(f)]

    if not files:
        print("No valid files found!")
        return

    # Run nesting with parameters from command line
    optimization_params = {
        "generations": args.generations,
        "population_size": args.population,
        "rotation_increment": args.rot_step,
        "max_rotation": args.max_rotation,
        "compactness_weight": args.compactness,
        "grid_size": args.grid,
    }

    output_file, utilization, parts_placed = nest_gcode_files_hybrid(
        files,
        args.width,
        args.height,
        args.spacing,
        args.output,
        optimization_params=optimization_params,
    )

    if parts_placed > 0:
        print(
            f"\nNesting complete! {parts_placed} parts placed with {utilization:.2f}% utilization."
        )
        print(f"Output saved to: {output_file}")
    else:
        print("\nNesting failed! No parts could be placed.")


# Standalone functions for parallel processing support
def _decode_order_vector(vector, shapes, valid_rotations=None):
    """
    Standalone version of decode_order_vector for parallel processing.
    Returns: (order_rotation_pairs, ordered_shapes)
    """
    if valid_rotations is None:
        valid_rotations = [0, 30, 60, 90]

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


def _adaptive_greedy_placement(
    shapes, rotations, sheet_width, sheet_height, spacing, grid_size
):
    """
    Standalone version of greedy placement for parallel processing.
    """
    # Create sheet polygon
    from shapely.geometry import box

    sheet_polygon = box(0, 0, sheet_width, sheet_height)

    placements = []
    placed_polygons = []

    # Multi-resolution grid search parameters
    coarse_grid = grid_size * 4  # Very coarse initial grid
    medium_grid = grid_size * 2  # Medium grid for refinement
    fine_grid = grid_size  # Fine grid for final refinement

    # Try to place each shape in order
    for shape_idx, (shape, rotation) in enumerate(zip(shapes, rotations)):
        placed = False
        best_placement = None
        best_position_metric = float("inf")  # Lower is better

        # Create rotated polygon
        poly = shape.to_shapely_polygon()
        cx, cy = shape.center()
        from shapely.affinity import rotate, translate

        rotated_poly = rotate(poly, rotation, origin=(cx, cy))

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
            import numpy as np

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
                    collision = False
                    for placed_poly in placed_polygons:
                        if buffered_poly.intersects(placed_poly):
                            collision = True
                            break

                    if not collision:
                        # Calculate position metric (prefer bottom-left)
                        # Lower values are better
                        position_metric = x + y

                        if position_metric < best_position_metric:
                            best_position_metric = position_metric
                            from shappack import Placement

                            best_placement = Placement(
                                shape=shape,
                                x_offset=x_offset,
                                y_offset=y_offset,
                                rotation=rotation,
                                polygon=test_poly,
                            )
                            placed = True

                            # Early stopping if we find a good position
                            if x < sheet_width * 0.15 and y < sheet_height * 0.15:
                                break

                if placed and x < sheet_width * 0.1:
                    break

        # Add best placement if found
        if best_placement:
            placements.append(best_placement)
            placed_polygons.append(best_placement.polygon.buffer(spacing / 2))

    return placements


def _calculate_metrics(placements, sheet_width, sheet_height, compactness_weight=0.3):
    """
    Standalone version of calculate_metrics for parallel processing.
    Returns: (utilization, compactness, fitness)
    """
    if not placements:
        return 0.0, 0.0, 0.0

    # Calculate sheet utilization
    sheet_area = sheet_width * sheet_height
    total_area_used = sum(p.polygon.area for p in placements)
    utilization = total_area_used / sheet_area if sheet_area > 0 else 0

    # Calculate compactness
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


if __name__ == "__main__":
    main()
