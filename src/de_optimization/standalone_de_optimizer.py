#!/usr/bin/env python3
"""
Standalone Differential Evolution Optimizer for CNC Shape Nesting
This version includes all necessary classes to avoid import issues.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point as ShapelyPoint
from shapely.affinity import translate, rotate
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
import matplotlib.patches as mpatches


@dataclass
class Point:
    """Simple point class"""

    x: float
    y: float


@dataclass
class Shape:
    """Shape representation"""

    points: List[Point]
    gcode_lines: List[str]
    original_idx: int = -1

    def to_shapely_polygon(self) -> Polygon:
        """Convert to Shapely polygon for collision detection"""
        coords = [(p.x, p.y) for p in self.points]
        if len(coords) < 3:
            return ShapelyPoint(coords[0] if coords else (0, 0)).buffer(0.1)
        return Polygon(coords)

    def bounds(self):
        """Get bounding box of shape"""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def width(self):
        minx, _, maxx, _ = self.bounds()
        return maxx - minx

    def height(self):
        _, miny, _, maxy = self.bounds()
        return maxy - miny

    def center(self):
        minx, miny, maxx, maxy = self.bounds()
        return (minx + maxx) / 2, (miny + maxy) / 2


@dataclass
class Placement:
    """Placement information for a shape"""

    shape: Shape
    x_offset: float
    y_offset: float
    rotation: float
    polygon: Polygon


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


class StandaloneDEOptimizer:
    """Complete DE optimizer for shape nesting without external dependencies"""

    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        spacing: float = 5.0,
        rotation_increment: float = 15.0,
    ):
        """Initialize the optimizer"""
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

        # Calculate valid rotation angles
        self.valid_rotations = list(range(0, 360, int(rotation_increment)))

    def create_bounds(
        self, shapes: List[Shape], max_copies_per_shape: int
    ) -> List[Tuple[float, float]]:
        """Create parameter bounds for DE"""
        total_parts = len(shapes) * max_copies_per_shape
        bounds = []

        for i in range(total_parts):
            bounds.append((0, self.sheet_width))  # X position
            bounds.append((0, self.sheet_height))  # Y position
            bounds.append((0, len(self.valid_rotations) - 1))  # Rotation index

        return bounds

    def decode_vector(
        self, vector: np.ndarray, shapes: List[Shape], max_copies_per_shape: int
    ) -> Tuple[List[Placement], float, int]:
        """Decode DE vector into shape placements"""
        placements = []
        placed_polygons = []
        total_area_used = 0

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

            # Transform shape
            poly = shape.to_shapely_polygon()
            poly = rotate(poly, rotation_angle, origin="centroid")
            poly = translate(poly, xoff=x, yoff=y)

            # Check validity
            if self._is_valid_placement(poly, placed_polygons):
                buffered_poly = poly.buffer(self.spacing / 2)
                placed_polygons.append(buffered_poly)
                total_area_used += poly.area

                placement = Placement(
                    shape=shape,
                    x_offset=x,
                    y_offset=y,
                    rotation=rotation_angle,
                    polygon=poly,
                )
                placements.append(placement)

        utilization = total_area_used / self.sheet_area if self.sheet_area > 0 else 0
        return placements, utilization, len(placements)

    def _is_valid_placement(
        self, polygon: Polygon, placed_polygons: List[Polygon]
    ) -> bool:
        """Check if placement is valid"""
        if not self.sheet_polygon.contains(polygon):
            return False

        for placed in placed_polygons:
            if polygon.intersects(placed):
                return False

        return True

    def create_objective_function(
        self, shapes: List[Shape], max_copies_per_shape: int
    ) -> Callable:
        """Create objective function for DE"""

        def objective(vector):
            _, utilization, _ = self.decode_vector(vector, shapes, max_copies_per_shape)
            return -utilization  # Negative for minimization

        return objective

    def create_callback(
        self, shapes: List[Shape], max_copies_per_shape: int
    ) -> Callable:
        """Create callback for progress tracking"""

        def callback(xk, convergence):
            self.current_generation += 1

            placements, utilization, shapes_placed = self.decode_vector(
                xk, shapes, max_copies_per_shape
            )

            elapsed_time = time.time() - self.start_time if self.start_time else 0

            metric = OptimizationMetrics(
                generation=self.current_generation,
                best_utilization=utilization,
                shapes_placed=shapes_placed,
                total_shapes_attempted=len(shapes) * max_copies_per_shape,
                best_vector=xk.copy(),
                computation_time=elapsed_time,
            )

            self.metrics_history.append(metric)

            if not self.best_placement or utilization > self.best_placement[1]:
                self.best_placement = (placements, utilization)

            print(
                f"Generation {self.current_generation}: "
                f"Utilization = {utilization:.2%}, "
                f"Shapes placed = {shapes_placed}/{len(shapes) * max_copies_per_shape}, "
                f"Time = {elapsed_time:.1f}s"
            )

            return False

        return callback

    def optimize(
        self,
        shapes: List[Shape],
        max_copies_per_shape: int = 1,
        generations: int = 100,
        population_size: int = 50,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = "best1bin",
        seed: Optional[int] = None,
        workers: int = 1,
        verbose: bool = True,
    ) -> Tuple[List[Placement], List[OptimizationMetrics]]:
        """Run DE optimization"""

        # Reset state
        self.metrics_history = []
        self.best_placement = None
        self.current_generation = 0
        self.start_time = time.time()

        # Check for empty input
        if not shapes:
            print("Warning: No shapes to optimize")
            return [], []

        # Create bounds
        bounds = self.create_bounds(shapes, max_copies_per_shape)

        # Create objective
        objective = self.create_objective_function(shapes, max_copies_per_shape)

        # Create callback
        callback = (
            self.create_callback(shapes, max_copies_per_shape) if verbose else None
        )

        # Run optimization
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

        # Get final solution
        final_placements, final_utilization, final_count = self.decode_vector(
            result.x, shapes, max_copies_per_shape
        )

        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final utilization: {final_utilization:.2%}")
            print(f"Total shapes placed: {final_count}")
            print(f"Total time: {time.time() - self.start_time:.1f} seconds")

        return final_placements, self.metrics_history

    def visualize_results(
        self,
        placements: List[Placement],
        metrics: Optional[List[OptimizationMetrics]] = None,
        title: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        """Visualize nesting results with statistics"""
        fig, (ax_main, ax_stats) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]}
        )

        # Main visualization
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

        # Color map
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        # Draw shapes
        shape_counts = {}
        total_area = 0

        for placement in placements:
            shape_idx = placement.shape.original_idx
            if shape_idx not in shape_counts:
                shape_counts[shape_idx] = 0
            shape_counts[shape_idx] += 1

            color = colors[shape_idx % len(colors)]

            coords = list(placement.polygon.exterior.coords)
            mpl_poly = MPLPolygon(
                coords, alpha=0.7, facecolor=color, edgecolor="black", linewidth=0.5
            )
            ax_main.add_patch(mpl_poly)

            total_area += placement.polygon.area

            # Show rotation for non-zero angles
            if placement.rotation != 0:
                cx, cy = placement.polygon.centroid.x, placement.polygon.centroid.y
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

        # Statistics
        ax_stats.axis("off")
        stats_text = []

        utilization = (total_area / self.sheet_area) * 100

        stats_text.append("=== NESTING STATISTICS ===\n")
        stats_text.append(f"Sheet: {self.sheet_width:.1f} x {self.sheet_height:.1f} mm")
        stats_text.append(f"Sheet Area: {self.sheet_area:.0f} mm²\n")
        stats_text.append(f"Total Shapes: {len(placements)}")
        stats_text.append(f"Unique Types: {len(shape_counts)}")
        stats_text.append(f"Area Used: {total_area:.0f} mm²")
        stats_text.append(f"Utilization: {utilization:.2f}%\n")

        stats_text.append("Shapes by Type:")
        for idx, count in sorted(shape_counts.items()):
            stats_text.append(f"  Type {idx}: {count} copies")

        # Rotation stats
        rotation_counts = {}
        for p in placements:
            rot = p.rotation
            if rot not in rotation_counts:
                rotation_counts[rot] = 0
            rotation_counts[rot] += 1

        stats_text.append("\nRotations Used:")
        for rot, count in sorted(rotation_counts.items()):
            stats_text.append(f"  {rot:.0f}°: {count} shapes")

        # Optimization metrics
        if metrics and len(metrics) > 0:
            final = metrics[-1]
            stats_text.append("\n=== OPTIMIZATION ===")
            stats_text.append(f"Generations: {final.generation}")
            stats_text.append(f"Time: {final.computation_time:.1f}s")
            stats_text.append(
                f"Time/Gen: {final.computation_time / final.generation:.2f}s"
            )

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

        plt.suptitle("DE Optimized CNC Shape Nesting", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to: {output_file}")

        plt.show()

    def visualize_progress(
        self, metrics: List[OptimizationMetrics], output_file: Optional[str] = None
    ):
        """Plot optimization progress"""
        if not metrics:
            print("No metrics to visualize")
            return

        generations = [m.generation for m in metrics]
        utilizations = [m.best_utilization * 100 for m in metrics]
        shapes_placed = [m.shapes_placed for m in metrics]
        fitness = [m.best_fitness for m in metrics]
        times = [m.computation_time for m in metrics]

        fig, axes = plt.subplots(4, 1, figsize=(12, 16))

        # Fitness/Objective function
        axes[0].plot(
            generations,
            fitness,
            "m-",
            linewidth=2,
            marker="o",
            markersize=3,
            markevery=max(1, len(generations) // 20),
        )
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness Value")
        axes[0].set_title("Objective Function Progress")
        axes[0].grid(True, alpha=0.3)

        # Utilization
        axes[1].plot(
            generations,
            utilizations,
            "b-",
            linewidth=2,
            marker="o",
            markersize=3,
            markevery=max(1, len(generations) // 20),
        )
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Utilization (%)")
        axes[1].set_title("Sheet Utilization Progress")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)

        # Shapes placed
        axes[2].plot(
            generations,
            shapes_placed,
            "g-",
            linewidth=2,
            marker="s",
            markersize=3,
            markevery=max(1, len(generations) // 20),
        )
        axes[2].set_xlabel("Generation")
        axes[2].set_ylabel("Shapes Placed")
        axes[2].set_title("Number of Shapes Placed")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(bottom=0)

        if metrics:
            max_shapes = metrics[0].total_shapes_attempted
            axes[2].axhline(
                y=max_shapes,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Max: {max_shapes}",
            )
            axes[2].legend()

        # Time
        axes[3].plot(generations, times, "r-", linewidth=2)
        axes[3].set_xlabel("Generation")
        axes[3].set_ylabel("Time (seconds)")
        axes[3].set_title("Computation Time")
        axes[3].grid(True, alpha=0.3)

        plt.suptitle("DE Optimization Progress", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Saved progress plot to: {output_file}")

        plt.show()


def create_sample_shapes() -> List[Shape]:
    """Create sample shapes for testing"""
    shapes = []

    # Rectangle
    rect = Shape(
        points=[Point(0, 0), Point(30, 0), Point(30, 20), Point(0, 20)],
        gcode_lines=[],
        original_idx=0,
    )
    shapes.append(rect)

    # Triangle
    tri = Shape(
        points=[Point(0, 0), Point(25, 0), Point(12.5, 25)],
        gcode_lines=[],
        original_idx=1,
    )
    shapes.append(tri)

    # L-shape
    l_shape = Shape(
        points=[
            Point(0, 0),
            Point(20, 0),
            Point(20, 10),
            Point(10, 10),
            Point(10, 30),
            Point(0, 30),
        ],
        gcode_lines=[],
        original_idx=2,
    )
    shapes.append(l_shape)

    # Pentagon
    pentagon_points = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        x = 15 + 15 * np.cos(angle)
        y = 15 + 15 * np.sin(angle)
        pentagon_points.append(Point(x, y))
    pentagon = Shape(points=pentagon_points, gcode_lines=[], original_idx=3)
    shapes.append(pentagon)

    return shapes


def main():
    """Demonstration of standalone DE optimizer"""
    print("\n" + "=" * 60)
    print("STANDALONE DIFFERENTIAL EVOLUTION OPTIMIZER DEMO")
    print("=" * 60)

    # Create sample shapes
    shapes = create_sample_shapes()
    print(f"\nCreated {len(shapes)} sample shapes")

    # Initialize optimizer
    optimizer = StandaloneDEOptimizer(
        sheet_width=300.0,
        sheet_height=200.0,
        spacing=5.0,
        rotation_increment=15.0,  # 15-degree increments
    )

    print(f"Sheet size: {optimizer.sheet_width} x {optimizer.sheet_height} mm")
    print(f"Rotation angles: {optimizer.valid_rotations}")

    # Run optimization
    print("\nStarting optimization...")
    print("-" * 40)

    placements, metrics = optimizer.optimize(
        shapes=shapes,
        max_copies_per_shape=10,
        generations=50,
        population_size=30,
        mutation=(0.5, 1.0),
        recombination=0.7,
        verbose=True,
    )

    # Show results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)

    if metrics:
        final = metrics[-1]
        print(f"Final utilization: {final.best_utilization:.2%}")
        print(f"Shapes placed: {final.shapes_placed}")
        print(f"Time taken: {final.computation_time:.1f} seconds")

    # Visualize
    print("\nGenerating visualizations...")

    optimizer.visualize_results(
        placements,
        metrics,
        title="Standalone DE Optimization Results",
        output_file="standalone_de_result.png",
    )

    optimizer.visualize_progress(metrics, output_file="standalone_de_progress.png")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
