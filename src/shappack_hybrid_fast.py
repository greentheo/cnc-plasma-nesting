#!/usr/bin/env python3
"""
High-performance hybrid sheet nesting algorithm that combines differential evolution with greedy bottom-left placement.

This optimized module provides a hybrid approach that:
1. Uses DE to optimize the order and rotation of parts
2. Uses a greedy bottom-left algorithm for actual part placement
3. Optimizes for both sheet utilization and compactness
4. Includes performance optimizations for faster placement

Performance improvements include:
- Multi-resolution grid search (coarse-to-fine)
- Smart caching of placements
- Efficient collision detection
- Adaptive evaluation frequency
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
        self.best_utilization = best_utilization
        self.best_compactness = best_compactness
        self.shapes_placed = shapes_placed
        self.total_shapes = total_shapes
        self.elapsed_time = elapsed_time


class HybridNestingAlgorithm(NestingAlgorithm):
    """
    High-performance hybrid nesting algorithm that combines differential evolution with greedy placement.

    This approach:
    1. Uses DE to determine placement order and rotation of shapes
    2. Uses a multi-resolution greedy bottom-left algorithm for actual placement
    3. Considers both sheet utilization and compactness in fitness function
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
        self.current_generation = 0
        self.start_time = time.time()
        self.eval_count = 0  # Counter for evaluations
        self.total_time_placing = 0.0  # Time spent in placement function
        self.total_time_collision = 0.0  # Time spent checking collisions

        # Performance optimization
        self.placement_cache = {}  # Cache for placement results
        self.best_shape_positions = {}  # Cache for best positions per shape

        # Default rotation increments (can be configured)
        self.valid_rotations = [0, 30, 60, 90]
        self.max_rotation = 90  # Default max rotation angle

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

    def _fast_check_collision(self, test_poly, placed_polygons):
        """
        Fast collision check using bounding box pre-filtering.

        Args:
            test_poly: Polygon to test
            placed_polygons: List of already placed polygons

        Returns:
            bool: True if collision detected, False otherwise
        """
        # Get test polygon bounds
        test_bounds = test_poly.bounds

        # Quick check with bounding boxes first
        for placed_poly in placed_polygons:
            placed_bounds = placed_poly.bounds

            # Check if bounding boxes overlap
            if not (test_bounds[2] < placed_bounds[0] or  # test right < placed left
                    test_bounds[0] > placed_bounds[2] or  # test left > placed right
                    test_bounds[3] < placed_bounds[1] or  # test top < placed bottom
                    test_bounds[1] > placed_bounds[3]):   # test bottom > placed top

                # Bounding boxes overlap, check actual polygons
                if test_poly.intersects(placed_poly):
                    return True

        return False

    def _adaptive_greedy_placement(
        self, shapes: List[Shape], rotations: List[int]
    ) -> List[Placement]:
        """
        Perform adaptive multi-resolution greedy bottom-left placement.
        This is a high-performance version that uses:
        1. Multi-resolution grid search (coarse-to-fine)
        2. Bounding box collision pre-filtering
        3. Shape position caching
        4. Early stopping for good positions

        Args:
            shapes: List of shapes in placement order
            rotations: List of rotation angles for each shape

        Returns:
            List of final placements
        """
        placement_start_time = time.time()

        # Create a cache key for this set of shapes and rotations
        # Use string representation to avoid hashing issues with multiprocessing
        shape_ids = [id(s) for s in shapes]
        cache_key = str(list(zip(shape_ids, rotations)))

        # Check if we already have a result for this exact combination
        if cache_key in self.placement_cache:
            return self.placement_cache[cache_key]

        placements = []
        placed_polygons = []
        shapes_placed = 0

        # Log start of placement process once
        print(f"  Placing {len(shapes)} shapes with {len(self.valid_rotations)} possible rotations...")

        # Multi-resolution grid search parameters
        coarse_grid = self.grid_size * 4  # Very coarse initial grid
        medium_grid = self.grid_size * 2  # Medium grid for refinement
        fine_grid = self.grid_size        # Fine grid for final refinement

        total_positions_tested = 0

        # For tracking best edge utilization
        edge_usage_left = 0
        edge_usage_bottom = 0

        # Try to place each shape in order
        for shape_idx, (shape, rotation) in enumerate(zip(shapes, rotations)):
            placed = False
            best_placement = None
            best_position_metric = float("inf")  # Lower is better

            # Try to use cached best position for this shape and rotation
            shape_key = (id(shape), rotation)
            if shape_key in self.best_shape_positions:
                cached_x, cached_y = self.best_shape_positions[shape_key]

                # Create rotated polygon
                poly = shape.to_shapely_polygon()
                cx, cy = shape.center()
                rotated_poly = rotate(poly, rotation, origin=(cx, cy))

                # Get bounds of rotated shape
                minx, miny, maxx, maxy = rotated_poly.bounds

                # Calculate offset to move shape to this position
                x_offset = cached_x - minx
                y_offset = cached_y - miny

                # Transform polygon to test position
                test_poly = translate(rotated_poly, xoff=x_offset, yoff=y_offset)

                # Add spacing buffer
                buffered_poly = test_poly.buffer(self.spacing / 2)

                # Check if it fits and doesn't collide
                if (self.sheet_polygon.contains(buffered_poly) and
                    not self._fast_check_collision(buffered_poly, placed_polygons)):

                    # Use the cached position
                    best_placement = Placement(
                        shape=shape,
                        x_offset=x_offset,
                        y_offset=y_offset,
                        rotation=rotation,
                        polygon=test_poly,
                    )
                    placed = True
                    total_positions_tested += 1

            # If cached position didn't work, perform grid search
            if not placed:
                # Create rotated polygon
                poly = shape.to_shapely_polygon()
                cx, cy = shape.center()
                rotated_poly = rotate(poly, rotation, origin=(cx, cy))

                # Get bounds of rotated shape
                minx, miny, maxx, maxy = rotated_poly.bounds
                shape_width = maxx - minx
                shape_height = maxy - miny

                # Start with very coarse grid to find general area
                for grid_size in [coarse_grid, medium_grid, fine_grid]:
                    # Skip finer grids if we already found a really good position
                    if placed and best_position_metric < (self.sheet_width + self.sheet_height) * 0.05:
                        break

                    # Do a sweep from bottom-left to top-right
                    for y in np.arange(
                        self.spacing,
                        self.sheet_height - shape_height - self.spacing,
                        grid_size,
                    ):
                        for x in np.arange(
                            self.spacing,
                            self.sheet_width - shape_width - self.spacing,
                            grid_size,
                        ):
                            total_positions_tested += 1

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
                            if self._fast_check_collision(buffered_poly, placed_polygons):
                                continue

                            # Calculate position metric (prefer bottom-left and edge alignment)
                            # Lower is better
                            position_metric = x + y

                            # Prefer positions that align with existing shapes (edge utilization)
                            # This creates a more compact packing
                            edge_alignment_bonus = 0
                            for placed_poly in placed_polygons:
                                if buffered_poly.distance(placed_poly) < self.spacing * 1.5:
                                    edge_alignment_bonus += 10  # Bonus for being close to other parts

                            # Apply edge alignment bonus (subtracting from metric makes it better)
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
                                if self.early_stopping and x < self.sheet_width * 0.15 and y < self.sheet_height * 0.15:
                                    break

                        # Break out of y-loop if we found a really good position
                        if placed and self.early_stopping and x < self.sheet_width * 0.1:
                            break

                    # If we found a placement on this grid level and it's good enough, stop
                    if placed and self.early_stopping and best_position_metric < (self.sheet_width + self.sheet_height) * 0.1:
                        break

            # Add best placement if found
            if best_placement:
                placements.append(best_placement)
                placed_polygons.append(best_placement.polygon.buffer(self.spacing / 2))
                shapes_placed += 1

                # Cache the good position for this shape and rotation
                if best_placement:
                    shape_key = (id(shape), rotation)
                    # Store the absolute position
                    minx, miny, _, _ = best_placement.polygon.bounds
                    self.best_shape_positions[shape_key] = (minx, miny)

                # Track edge usage for sheet
                minx, miny, _, _ = best_placement.polygon.bounds
                if minx < self.sheet_width * 0.1:
                    edge_usage_left += 1
                if miny < self.sheet_height * 0.1:
                    edge_usage_bottom += 1

        # Log summary of placement process
        elapsed = time.time() - placement_start_time
        self.total_time_placing += elapsed
        print(
            f"  Placed {shapes_placed}/{len(shapes)} shapes, tested {total_positions_tested} positions, time: {elapsed:.1f}s"
        )
        if shapes_placed > 0:
            print(f"  Edge utilization - Left: {edge_usage_left}/{shapes_placed}, Bottom: {edge_usage_bottom}/{shapes_placed}")

        # Cache the result before returning
        self.placement_cache[cache_key] = placements

        # If cache is getting too large, remove older entries
        if len(self.placement_cache) > 100:
            # Remove half of the cache entries
            keys_to_remove = list(self.placement_cache.keys())[:len(self.placement_cache)//2]
            for k in keys_to_remove:
                del self.placement_cache[k]

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
            try:
                min_x = min(min(p.polygon.exterior.coords, key=lambda c: c[0])[0] for p in placements)
                min_y = min(min(p.polygon.exterior.coords, key=lambda c: c[1])[1] for p in placements)
                max_x = max(max(p.polygon.exterior.coords, key=lambda c: c[0])[0] for p in placements)
                max_y = max(max(p.polygon.exterior.coords, key=lambda c: c[1])[1] for p in placements)

                bounding_box_area = (max_x - min_x) * (max_y - min_y)
                compactness = (
                    total_area_used / bounding_box_area if bounding_box_area > 0 else 0
                )
            except Exception:
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

        # Perform adaptive greedy placement
        placements = self._adaptive_greedy_placement(ordered_shapes, rotations)

        # Calculate metrics
        utilization, compactness, fitness = self._calculate_metrics(placements)

        # Log good solutions
        if self.eval_count % 10 == 0 or fitness > 0.5:
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

            # Only do detailed evaluation every 5 generations to save time
            do_detailed = (self.current_generation % 5 == 0) or (
                self.current_generation == 1
            )

            if not do_detailed:
                print(f"\nGeneration {self.current_generation} (quick check)...")
                return False

            print(f"\nGeneration {self.current_generation} placement starting...")

            # Decode order and rotation
            order_rotation_pairs, _ = self._decode_order_vector(xk, shapes)

            # Extract shape indices and rotations in order
            shape_indices = [pair[0] for pair in order_rotation_pairs]
            rotations = [pair[1] for pair in order_rotation_pairs]

            # Get shapes in the optimized order
            ordered_shapes = [shapes[idx] for idx in shape_indices]

            print(f"  Shape order: {shape_indices}")
            print(f"  Rotations: {rotations}")

            # Reset evaluation counter for this generation
            self.eval_count = 0

            # Perform greedy placement
            placements = self._adaptive_greedy_placement(ordered_shapes, rotations)

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
                or (
                    utilization == self.best_utilization
                    and compactness == self.best_compactness
                    and len(placements) > len(self.best_placements)
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

            # Save intermediate visualization only for milestone generations
            if self.current_generation == 1 or self.current_generation % 20 == 0:
                viz_dir = "hybrid_progress"
                os.makedirs(viz_dir, exist_ok=True)
                viz_file = f"{viz_dir}/hybrid_gen_{self.current_generation:03d}.png"
                self.visualize_nesting(placements, filename=viz_file)
                print(f"  Saved visualization to {viz_file}")

            # Clear the placement cache periodically to avoid memory issues
            if len(self.placement_cache) > 200:
                cache_mb = len(self.placement_cache) * 0.5
                print(
                    f"  Clearing placement cache ({len(self.placement_cache)} entries), memory freed: {cache_mb:.1f}MB"
                )
                self.placement_cache.clear()

                # Force garbage collection
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
        max_rotation: float = 90.0,  # Maximum rotation angle (default 90 for symmetric shapes)
        seed: Optional[int] = None,
        workers: int = 1,  # Default to 1 worker to avoid pickling issues
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
        self.total_time_placing = 0.0
        self.total_time_collision = 0.0

        # Clear caches
        self.placement_cache = {}
        self.best_shape_positions = {}

        if not shapes:
            print("Warning: No shapes to optimize")
            return [], []

        if verbose:
            print(f"\n=== Starting High-Performance Hybrid Optimization ===")
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
                final_placements = self._adaptive_greedy_placement(
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
                print
