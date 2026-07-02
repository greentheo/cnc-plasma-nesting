#!/usr/bin/env python3
"""
An extension of the HybridNestingAlgorithm that supports multi-objective optimization.
This adds support for three objectives: parts placed, compactness, and utilization.
"""

import time
from typing import List, Tuple, Callable, Optional, Any, Dict, Union

from shappack import Shape, Placement
from shappack_hybrid import HybridNestingAlgorithm, OptimizationMetrics


class MultiObjectiveHybridNestingAlgorithm(HybridNestingAlgorithm):
    """
    Extension of HybridNestingAlgorithm with multi-objective optimization.
    This class adds a third objective (parts placed) to the fitness function.
    """

    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        spacing: float = 5.0,
        grid_size: float = 10.0,
        parts_placed_weight: float = 0.6,
        compactness_weight: float = 0.2,
        utilization_weight: float = 0.2,
        external_objective: Optional[Callable[..., float]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the multi-objective hybrid nesting algorithm.

        Args:
            sheet_width: Width of the sheet in mm
            sheet_height: Height of the sheet in mm
            spacing: Minimum spacing between shapes in mm
            grid_size: Size of grid cells for placement search in mm
            parts_placed_weight: Weight for the parts placed objective (0.0-1.0)
            compactness_weight: Weight for the compactness objective (0.0-1.0)
            utilization_weight: Weight for the utilization objective (0.0-1.0)
            external_objective: Optional external objective function for parallel processing
            verbose: Whether to print verbose output
        """
        # Call the parent constructor but we'll override the compactness weight handling
        super().__init__(
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            spacing=spacing,
            grid_size=grid_size,
            compactness_weight=compactness_weight,
            external_objective=external_objective,
        )

        # Set verbose flag
        self.verbose = verbose

        # Store weights for multi-objective optimization
        self.parts_placed_weight: float = parts_placed_weight
        self.compactness_weight: float = compactness_weight
        self.utilization_weight: float = utilization_weight
        self.parts_placed_ratio: float = 0.0

        # Validate weights
        if (
            abs(parts_placed_weight + compactness_weight + utilization_weight - 1.0)
            > 1e-6
        ):
            print(
                f"Warning: Weights sum to {parts_placed_weight + compactness_weight + utilization_weight}, "
                f"not 1.0. This is allowed but may produce unexpected results."
            )

    def _calculate_metrics(
        self, placements: List[Placement], shapes: Optional[List[Shape]] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate optimization metrics for a set of placements.

        Args:
            placements: List of placements
            shapes: Optional list of all shapes to calculate parts placed ratio

        Returns:
            Tuple of (utilization, compactness, fitness)
        """
        if not placements:
            return 0.0, 0.0, 0.0

        # Calculate sheet utilization
        sheet_area = self.sheet_width * self.sheet_height
        total_area_used = sum(p.polygon.area for p in placements)
        utilization = total_area_used / sheet_area if sheet_area > 0 else 0

        # Calculate compactness (area efficiency within bounding box)
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

        # Calculate parts placed ratio and store it as instance attribute
        if shapes is not None:
            self.parts_placed_ratio = len(placements) / len(shapes) if shapes else 0
        else:
            # If shapes not provided, assume it's 1.0 (all parts placed)
            self.parts_placed_ratio = 1.0

        # Combined fitness function with weighted components is calculated later
        # to avoid duplication with evaluate_solution
        fitness = 0.0  # Placeholder, calculated before returning

        # Calculate final multi-objective fitness
        fitness = (
            self.parts_placed_weight * self.parts_placed_ratio
            + self.compactness_weight * compactness
            + self.utilization_weight * utilization
        )

        # Return 3 values to match parent class signature
        return utilization, compactness, fitness

    def _evaluate_solution(self, vector: Any, shapes: List[Shape]) -> float:
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
        if not hasattr(self, "eval_count"):
            self.eval_count = 0
        self.eval_count += 1

        # Log evaluation information periodically
        if self.eval_count % 10 == 0 and self.verbose:
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
            ordered_shapes, rotations, log_step=len(shapes) if self.verbose else 0
        )

        # Calculate basic metrics - our _calculate_metrics already includes the multi-objective fitness
        utilization, compactness, fitness = self._calculate_metrics(placements, shapes)

        # Get the parts placed ratio we stored in the calculate_metrics method
        parts_placed_ratio = self.parts_placed_ratio

        # Log all solutions with meaningful information
        if (self.eval_count % 5 == 0 or fitness > 0.4) and self.verbose:
            print(
                f"  Solution #{self.eval_count}: Fitness={fitness:.4f}, Util={utilization:.2%}, "
                f"Compact={compactness:.2%}, Parts={len(placements)}/{len(shapes)} ({parts_placed_ratio:.2%})"
            )

        # Return negative fitness for minimization
        return -fitness

    def nest_shapes_hybrid(
        self,
        shapes: List[Shape],
        generations: int = 10,
        population_size: int = 30,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        rotation_increment: float = 45.0,
        max_rotation: float = 360.0,
        seed: Optional[int] = None,
        workers: int = 1,
        verbose: bool = False,
    ) -> Tuple[List[Placement], List[OptimizationMetrics]]:
        """
        Optimize shape nesting using a hybrid genetic-differential evolution algorithm.
        This version uses multi-objective optimization.

        Args:
            shapes: List of shapes to nest
            generations: Number of generations to run
            population_size: Size of the population
            rotation_increment: Increments to use for rotation (e.g., 45 means 0, 45, 90...)
            max_rotation: Maximum rotation angle to consider
            strategy: Differential evolution strategy
            mutation: Mutation parameters
            recombination: Recombination parameter
            workers: Number of worker processes for parallel evaluation
            seed: Random seed for reproducibility
            verbose: Whether to print verbose output

        Returns:
            Tuple of (placements, optimization_metrics)
        """
        self.verbose = verbose
        return super().nest_shapes_hybrid(
            shapes=shapes,
            generations=generations,
            population_size=population_size,
            mutation=mutation,
            recombination=recombination,
            strategy=strategy,
            rotation_increment=rotation_increment,
            max_rotation=max_rotation,
            seed=seed,
            workers=workers,
            verbose=verbose,
        )


# Example usage
if __name__ == "__main__":
    print("MultiObjectiveHybridNestingAlgorithm module loaded.")
    print("Use this class to optimize with multiple objectives:")
    print("  - Parts placed ratio")
    print("  - Compactness")
    print("  - Utilization")
