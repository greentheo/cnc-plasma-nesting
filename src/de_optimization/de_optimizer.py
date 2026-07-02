"""
Differential Evolution Optimizer for CNC Shape Nesting
Optimizes shape placement and rotation using scipy's differential evolution algorithm
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
import time


@dataclass
class OptimizationMetrics:
    """Track optimization progress across generations"""

    generation: int
    best_utilization: float
    shapes_placed: int
    total_shapes_attempted: int
    best_vector: np.ndarray
    computation_time: float
    mean_utilization: float = 0.0
    std_utilization: float = 0.0


class DEOptimizer:
    """Differential Evolution optimizer for shape nesting"""

    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        spacing: float = 5.0,
        rotation_increment: float = 15.0,
    ):
        """
        Initialize the DE optimizer.

        Args:
            sheet_width: Width of the sheet in mm
            sheet_height: Height of the sheet in mm
            spacing: Minimum spacing between shapes in mm
            rotation_increment: Rotation angle increment in degrees (e.g., 15 for 0, 15, 30, ...)
        """
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.spacing = spacing
        self.rotation_increment = rotation_increment
        self.sheet_area = sheet_width * sheet_height
        self.sheet_polygon = Polygon(
            [(0, 0), (sheet_width, 0), (sheet_width, sheet_height), (0, sheet_height)]
        )
        self.metrics_history = []
        self.best_placement = None
        self.current_generation = 0
        self.start_time = None

        # Calculate valid rotation angles based on increment
        self.valid_rotations = list(range(0, 360, int(rotation_increment)))

    def create_bounds(
        self, shapes: List, max_copies_per_shape: int
    ) -> List[Tuple[float, float]]:
        """
        Create parameter bounds for differential evolution.

        Each shape instance has 3 parameters: x, y, rotation_index

        Args:
            shapes: List of Shape objects to place
            max_copies_per_shape: Maximum copies of each shape to place

        Returns:
            List of (min, max) tuples for each parameter
        """
        total_parts = len(shapes) * max_copies_per_shape
        bounds = []

        for i in range(total_parts):
            # X position bounds
            bounds.append((0, self.sheet_width))
            # Y position bounds
            bounds.append((0, self.sheet_height))
            # Rotation index bounds (index into valid_rotations array)
            bounds.append((0, len(self.valid_rotations) - 1))

        return bounds

    def decode_vector(
        self, vector: np.ndarray, shapes: List, max_copies_per_shape: int
    ) -> Tuple[List, float, int]:
        """
        Decode DE vector into shape placements.

        Args:
            vector: DE solution vector
            shapes: List of Shape objects
            max_copies_per_shape: Maximum copies per shape

        Returns:
            Tuple of (placements, utilization, shapes_placed_count)
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
            if self._is_valid_placement(poly, placed_polygons):
                # Create buffered polygon for spacing
                buffered_poly = poly.buffer(self.spacing / 2)
                placed_polygons.append(buffered_poly)
                total_area_used += poly.area

                # Store placement info
                # Create a simple placement object to avoid import issues
                placement = type(
                    "Placement",
                    (),
                    {
                        "shape": shape,
                        "x_offset": x,
                        "y_offset": y,
                        "rotation": rotation_angle,
                        "polygon": poly,
                    },
                )()
                placements.append(placement)

        utilization = total_area_used / self.sheet_area if self.sheet_area > 0 else 0

        return placements, utilization, len(placements)

    def _is_valid_placement(
        self, polygon: Polygon, placed_polygons: List[Polygon]
    ) -> bool:
        """
        Check if a polygon placement is valid.

        Args:
            polygon: Polygon to check
            placed_polygons: List of already placed polygons (with buffer)

        Returns:
            True if placement is valid, False otherwise
        """
        # Check if polygon is within sheet bounds
        if not self.sheet_polygon.contains(polygon):
            return False

        # Check for overlaps with existing placements
        for placed in placed_polygons:
            if polygon.intersects(placed):
                return False

        return True

    def create_objective_function(
        self, shapes: List, max_copies_per_shape: int
    ) -> Callable:
        """
        Create the objective function for DE optimization.

        Args:
            shapes: List of Shape objects
            max_copies_per_shape: Maximum copies per shape

        Returns:
            Objective function that returns negative utilization (for minimization)
        """

        def objective(vector):
            """Evaluate placement vector, returns negative utilization."""
            _, utilization, _ = self.decode_vector(vector, shapes, max_copies_per_shape)

            # Return negative utilization for minimization
            # Add small penalty for invalid placements
            return -utilization

        return objective

    def create_callback(self, shapes: List, max_copies_per_shape: int) -> Callable:
        """
        Create callback function for tracking optimization progress.

        Args:
            shapes: List of Shape objects
            max_copies_per_shape: Maximum copies per shape

        Returns:
            Callback function for DE optimization
        """

        def callback(xk, convergence):
            """Called after each generation."""
            self.current_generation += 1

            # Decode current best solution
            placements, utilization, shapes_placed = self.decode_vector(
                xk, shapes, max_copies_per_shape
            )

            # Calculate time elapsed
            elapsed_time = time.time() - self.start_time if self.start_time else 0

            # Create metrics entry
            metric = OptimizationMetrics(
                generation=self.current_generation,
                best_utilization=utilization,
                shapes_placed=shapes_placed,
                total_shapes_attempted=len(shapes) * max_copies_per_shape,
                best_vector=xk.copy(),
                computation_time=elapsed_time,
            )

            self.metrics_history.append(metric)

            # Update best placement
            if not self.best_placement or utilization > self.best_placement[1]:
                self.best_placement = (placements, utilization)

            # Print progress
            print(
                f"Generation {self.current_generation}: "
                f"Utilization = {utilization:.2%}, "
                f"Shapes placed = {shapes_placed}/{len(shapes) * max_copies_per_shape}, "
                f"Time = {elapsed_time:.1f}s"
            )

            return False  # Don't stop optimization

        return callback

    def optimize(
        self,
        shapes: List,
        max_copies_per_shape: int = 1,
        generations: int = 100,
        population_size: int = 50,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        seed: Optional[int] = None,
        workers: int = 1,
        verbose: bool = True,
    ) -> Tuple[List, List[OptimizationMetrics]]:
        """
        Run differential evolution optimization.

        Args:
            shapes: List of Shape objects to place
            max_copies_per_shape: Maximum copies of each shape
            generations: Number of generations to run
            population_size: Size of population (number of candidate solutions)
            mutation: Mutation constant (F parameter), can be float or (min, max) tuple
            recombination: Crossover probability (CR parameter)
            strategy: DE strategy to use
            seed: Random seed for reproducibility
            workers: Number of parallel workers (-1 for all CPUs)
            verbose: Whether to print progress

        Returns:
            Tuple of (best_placements, metrics_history)
        """
        # Reset state
        self.metrics_history = []
        self.best_placement = None
        self.current_generation = 0
        self.start_time = time.time()

        # Create bounds for optimization
        bounds = self.create_bounds(shapes, max_copies_per_shape)

        # Check if we have valid bounds
        if not bounds:
            print("Warning: No shapes to optimize")
            return [], []

        # Create objective function
        objective = self.create_objective_function(shapes, max_copies_per_shape)

        # Create callback for progress tracking
        callback = (
            self.create_callback(shapes, max_copies_per_shape) if verbose else None
        )

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
            polish=False,  # Don't use local optimization at the end
            atol=0,  # Don't stop based on tolerance
            tol=0,
        )

        # Get final best solution
        final_placements, final_utilization, final_count = self.decode_vector(
            result.x, shapes, max_copies_per_shape
        )

        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final utilization: {final_utilization:.2%}")
            print(f"Total shapes placed: {final_count}")
            print(f"Total time: {time.time() - self.start_time:.1f} seconds")

        return final_placements, self.metrics_history

    def get_rotation_angles(self) -> List[float]:
        """Get list of valid rotation angles based on increment."""
        return self.valid_rotations.copy()
