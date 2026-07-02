#!/usr/bin/env python3
"""
Test script for hybrid nesting algorithm with parallel processing support.
This version is designed to be robust across different Python environments.

Usage:
  python hybrid_parallel_test.py [workers]

  Optional arguments:
    workers - Number of parallel workers (default: auto-detect)
"""

import sys
import os
import time
import random
import multiprocessing
from pathlib import Path

# Import the different nesting algorithms
from shappack import GCodeParser, GCodeGenerator, NestingAlgorithm
from shappack_hybrid import HybridNestingAlgorithm


# Define standalone objective function that can be pickled
def standalone_objective_function(
    vector,
    shapes,
    sheet_width,
    sheet_height,
    spacing,
    grid_size,
    valid_rotations,
    compactness_weight,
):
    """Standalone objective function for parallel processing."""
    # Import the necessary functions from shappack_hybrid
    from shappack_hybrid import _decode_order_vector, _calculate_metrics

    # Decode order and rotation
    order_rotation_pairs, _ = _decode_order_vector(
        vector, shapes, valid_rotations=valid_rotations
    )

    # Extract shape indices and rotations in order
    shape_indices = [pair[0] for pair in order_rotation_pairs]
    rotations = [pair[1] for pair in order_rotation_pairs]

    # Get shapes in the optimized order
    ordered_shapes = [shapes[idx] for idx in shape_indices]

    # Perform greedy bottom-left placement
    placements = greedy_bottom_left_placement(
        ordered_shapes,
        rotations,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        spacing=spacing,
        grid_size=grid_size,
    )

    # Calculate metrics
    utilization, compactness, fitness = _calculate_metrics(
        placements,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        compactness_weight=compactness_weight,
    )

    # Return negative fitness for minimization
    return -fitness


def greedy_bottom_left_placement(
    shapes, rotations, sheet_width, sheet_height, spacing, grid_size
):
    """
    Simplified greedy bottom-left placement for parallel processing.
    This version works with minimal dependencies for better pickling.
    """
    from shapely.geometry import box
    from shapely.affinity import rotate, translate
    import numpy as np
    from shappack import Placement

    # Create sheet polygon
    sheet_polygon = box(0, 0, sheet_width, sheet_height)

    placements = []
    placed_polygons = []

    # Grid search parameters
    step_size = grid_size

    # Try to place each shape in order
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

        # Try positions on a grid (bottom-left search pattern)
        for y in np.arange(
            spacing,
            sheet_height - shape_height - spacing,
            step_size,
        ):
            for x in np.arange(
                spacing,
                sheet_width - shape_width - spacing,
                step_size,
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
                        if x < sheet_width * 0.15 and y < sheet_height * 0.15:
                            break

            if placed and x < sheet_width * 0.1:
                break

        # Add best placement if found
        if best_placement:
            placements.append(best_placement)
            placed_polygons.append(best_placement.polygon.buffer(spacing / 2))

    return placements


def run_test_with_workers(worker_count):
    """Run nesting test with the specified number of workers."""
    # Set random seed for reproducibility
    random.seed(42)

    # Configure nesting parameters
    SHEET_WIDTH = 24 * 25.4  # mm (24 inches)
    SHEET_HEIGHT = 24 * 25.4  # mm (24 inches)
    SPACING = 3  # mm between parts

    print(f"\n{'=' * 70}")
    print(
        f"TESTING HYBRID ALGORITHM WITH {worker_count} WORKER{'S' if worker_count > 1 else ''}"
    )
    print(f"{'=' * 70}")
    print(f"\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm")
    print(f"Spacing: {SPACING} mm")

    # Test with fewer shapes for faster testing
    test_files = [
        ("geodesic_6_top.nc", 3),
        ("geodesic_6_bottom.nc", 1),
        ("geodesic_5_top.nc", 2),
        ("geodesic_5_bottom.nc", 1),
        ("geodesic_4_top.nc", 1),
        ("geodesic_4_bottom.nc", 2),
    ]

    # Parse files
    parser = GCodeParser()
    all_shapes = []
    original_gcodes = []

    print("Parsing G-code files...")
    for idx, (file_path, max_count) in enumerate(test_files):
        if not os.path.exists(file_path):
            print(f"✗ Missing: {file_path}")
            continue

        print(f"✓ Found: {file_path}")
        with open(file_path, "r") as f:
            gcode_text = f.read()
            original_gcodes.append(gcode_text)

        shapes = parser.parse_gcode(gcode_text)
        print(f"  Parsed {len(shapes)} shape(s), requesting {max_count} copies")

        # Duplicate shapes based on count
        for _ in range(max_count):
            for shape in shapes:
                shape_copy = type(shape)(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=idx,
                )
                all_shapes.append(shape_copy)

    if not all_shapes:
        print("\nNo shapes found!")
        return None

    print(f"\nTotal shape instances: {len(all_shapes)}")

    # Optimization parameters optimized for speed
    hybrid_params = {
        "generations": 3,  # Few generations for testing
        "population_size": 10,  # Small population
        "rotation_increment": 30.0,  # 30-degree increments
        "max_rotation": 90.0,  # Only test 0-90 degrees
        "compactness_weight": 0.3,
        "grid_size": 20.0,  # Larger grid for faster search
    }

    valid_rotations = [0, 30, 60, 90]  # Fixed rotation angles

    print(f"Generations: {hybrid_params['generations']}")
    print(f"Population: {hybrid_params['population_size']}")
    print(f"Rotation angles: {valid_rotations} (0-90° only)")
    print(f"Grid size: {hybrid_params['grid_size']} mm")
    print(f"Workers: {worker_count}")

    # Create the hybrid algorithm
    hybrid_algo = HybridNestingAlgorithm(
        sheet_width=SHEET_WIDTH,
        sheet_height=SHEET_HEIGHT,
        spacing=SPACING,
        grid_size=hybrid_params["grid_size"],
        compactness_weight=hybrid_params["compactness_weight"],
    )

    # Override the valid_rotations to ensure consistency
    hybrid_algo.valid_rotations = valid_rotations

    # For parallel processing, we'll use an external objective function
    # Define it as a simple lambda that calls our standalone function
    if worker_count > 1:
        hybrid_algo.external_objective = lambda vector: standalone_objective_function(
            vector,
            all_shapes,
            SHEET_WIDTH,
            SHEET_HEIGHT,
            SPACING,
            hybrid_params["grid_size"],
            valid_rotations,
            hybrid_params["compactness_weight"],
        )

    # Run the optimization
    start_time = time.time()
    try:
        hybrid_placements, hybrid_metrics = hybrid_algo.nest_shapes_hybrid(
            shapes=all_shapes,
            generations=hybrid_params["generations"],
            population_size=hybrid_params["population_size"],
            rotation_increment=hybrid_params["rotation_increment"],
            max_rotation=hybrid_params["max_rotation"],
            verbose=True,
            seed=42,
            workers=worker_count,
        )
    except Exception as e:
        print(f"\n⚠️ Error during optimization: {str(e)}")
        elapsed = time.time() - start_time
        return {"workers": worker_count, "error": str(e), "time": elapsed}

    # Calculate timing and metrics
    elapsed = time.time() - start_time

    # Calculate sheet utilization and compactness
    if hybrid_placements:
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        total_area = sum(p.polygon.area for p in hybrid_placements)
        utilization = (total_area / sheet_area) * 100

        # Calculate compactness
        try:
            min_x = min(
                min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in hybrid_placements
            )
            min_y = min(
                min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in hybrid_placements
            )
            max_x = max(
                max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in hybrid_placements
            )
            max_y = max(
                max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in hybrid_placements
            )

            bounding_box_area = (max_x - min_x) * (max_y - min_y)
            compactness = (
                (total_area / bounding_box_area) * 100 if bounding_box_area > 0 else 0
            )
        except (ValueError, AttributeError):
            compactness = 0
    else:
        utilization = 0
        compactness = 0

    print("\nResults:")
    print(f"Shapes placed: {len(hybrid_placements)} / {len(all_shapes)}")
    print(f"Utilization: {utilization:.2f}%")
    print(f"Compactness: {compactness:.2f}%")
    print(f"Time: {elapsed:.1f} seconds")

    # Save visualization
    result_file = f"hybrid_{worker_count}workers_result.png"
    hybrid_algo.visualize_nesting(hybrid_placements, filename=result_file)
    print(f"✓ Visualization saved to: {result_file}")

    # Return results for summary
    return {
        "workers": worker_count,
        "parts_placed": len(hybrid_placements),
        "utilization": utilization,
        "compactness": compactness,
        "time": elapsed,
        "placements": hybrid_placements,
    }


def main():
    """Run the parallel hybrid test."""
    # Determine available CPU cores
    available_cores = multiprocessing.cpu_count()
    recommended_workers = max(1, available_cores - 1)  # Leave one core free

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            requested_workers = int(sys.argv[1])
            worker_configs = [
                4,
                requested_workers,
            ]  # Always test with 1 worker for comparison
        except ValueError:
            print(f"Invalid worker count: {sys.argv[1]}")
            print(f"Using 1 and {min(4, recommended_workers)} workers")
            worker_configs = [4, min(4, recommended_workers)]
    else:
        # Default: test with 1 worker and auto-detected worker count
        worker_configs = [1, min(4, recommended_workers)]

    print("=" * 70)
    print("PARALLEL HYBRID NESTING ALGORITHM TEST")
    print("=" * 70)
    print(f"Available CPU cores: {available_cores}")
    print(f"Recommended workers: {recommended_workers}")
    print(f"Testing with worker counts: {worker_configs}")

    # Run tests with different worker configurations
    results = []
    base_time = None

    for workers in worker_configs:
        result = run_test_with_workers(workers)
        if result:
            # First result is the baseline for speed-up calculation
            if base_time is None and "error" not in result:
                base_time = result["time"]

            # Calculate speed-up factor
            if base_time and "error" not in result:
                result["speedup"] = base_time / result["time"]
            else:
                result["speedup"] = 0

            results.append(result)

    # Print comparison summary
    print("\n" + "=" * 70)
    print("PARALLEL PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\nWorkers | Parts Placed | Utilization | Compactness | Time (s) | Speed-up")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(
                f"{r['workers']:7} | {'ERROR':12} | {'N/A':10} | {'N/A':10} | {r['time']:7.1f} | {'N/A':7}"
            )
        else:
            print(
                f"{r['workers']:7} | {r['parts_placed']:12} | {r['utilization']:10.2f}% | "
                f"{r['compactness']:10.2f}% | {r['time']:7.1f} | {r['speedup']:7.2f}x"
            )

    # Find the best result by parts placed
    best_result = max(
        [r for r in results if "error" not in r],
        key=lambda x: (x["parts_placed"], x["utilization"]),
        default=None,
    )

    if best_result:
        best_workers = best_result["workers"]
        print(
            f"\nBest result: {best_workers} worker{'s' if best_workers > 1 else ''} "
            f"with {best_result['parts_placed']} parts at {best_result['utilization']:.2f}% utilization"
        )

        # Generate G-code for best result
        if best_result["placements"]:
            try:
                print(
                    f"\nGenerating G-code for best result ({best_workers} workers)..."
                )
                gcode_generator = GCodeGenerator()

                # Parse files again to get original G-code
                parser = GCodeParser()
                original_gcodes = []
                for file_path, _ in [
                    f
                    for f in [
                        ("geodesic_6_top.nc", 3),
                        ("geodesic_6_bottom.nc", 1),
                        ("geodesic_5_top.nc", 2),
                        ("geodesic_5_bottom.nc", 1),
                        ("geodesic_4_top.nc", 1),
                        ("geodesic_4_bottom.nc", 2),
                    ]
                    if os.path.exists(f[0])
                ]:
                    with open(file_path, "r") as f:
                        original_gcodes.append(f.read())

                output_gcode = (
                    gcode_generator.generate_nested_gcode_with_optimal_travel(
                        original_gcodes, best_result["placements"], parser
                    )
                )

                output_file = f"hybrid_{best_workers}workers_output.nc"
                with open(output_file, "w") as f:
                    f.write(output_gcode)

                print(f"✓ G-code saved to: {output_file}")
            except Exception as e:
                print(f"⚠️ G-code generation encountered an issue: {str(e)}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
