#!/usr/bin/env python3
"""
Test script for the parallel hybrid nesting algorithm.
This version is optimized to use multiple CPU cores for faster processing.
"""

import sys
import os
import time
import random
from pathlib import Path
import multiprocessing

# Import the different nesting algorithms
from shappack import GCodeParser, GCodeGenerator, NestingAlgorithm
from shappack_de import DENetworkingAlgorithm
from shappack_hybrid import HybridNestingAlgorithm


def test_parallel_hybrid_nesting():
    """Run a comparison test of the hybrid nesting algorithm with parallel processing."""
    # Set random seed for reproducibility
    random.seed(42)

    # Determine number of CPU cores available for parallel processing
    available_cores = multiprocessing.cpu_count()
    recommended_workers = max(1, available_cores - 4)  # Leave one core free

    # Configure nesting parameters
    SHEET_WIDTH = 24 * 25.4  # mm (24 inches)
    SHEET_HEIGHT = 48 * 25.4  # mm (48 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print(f"PARALLEL HYBRID NESTING ALGORITHM TEST")
    print("=" * 70)
    print(f"\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm")
    print(f"Spacing: {SPACING} mm")
    print(f"Available CPU cores: {available_cores}")
    print(f"Recommended workers: {recommended_workers}\n")

    # Test with several files and fewer copies for faster testing
    test_files = [
        ("geodesic_6_top.nc", 5),
        ("geodesic_6_bottom.nc", 5),
        ("geodesic_5_top.nc", 2),
        ("geodesic_5_bottom.nc", 2),
        ("geodesic_4_top.nc", 2),
        ("geodesic_4_bottom.nc", 4),
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
        print("\nNo shapes found! Creating sample files...")
        os.system("python create_geodesic_samples.py")
        return

    print(f"\nTotal shape instances: {len(all_shapes)}")

    # Run tests with different worker configurations
    worker_configs = [1]  # , min(4, recommended_workers)]

    results = []

    for workers in worker_configs:
        if workers > available_cores:
            print(
                f"\nSkipping {workers} workers configuration (exceeds available cores)"
            )
            continue

        print("\n" + "=" * 70)
        print(
            f"TESTING HYBRID ALGORITHM WITH {workers} WORKER{'S' if workers > 1 else ''}"
        )
        print("=" * 70)

        hybrid_params = {
            "generations": 5,  # Reduced for testing
            "population_size": 20,
            "rotation_increment": 30.0,  # 30-degree increments
            "compactness_weight": 0.3,
            "grid_size": 50.0,  # Larger grid for faster search
        }

        print(f"Generations: {hybrid_params['generations']}")
        print(f"Population: {hybrid_params['population_size']}")
        print(
            f"Rotation increment: {hybrid_params['rotation_increment']}° (0-90° only)"
        )
        print(f"Compactness weight: {hybrid_params['compactness_weight']}")
        print(f"Grid size: {hybrid_params['grid_size']} mm")
        print(f"Workers: {workers}")

        # Create a separate objective function that can be pickled
        def external_objective_function(vector):
            """Standalone objective function for parallel processing."""
            from shappack_hybrid import (
                _decode_order_vector,
                _adaptive_greedy_placement,
                _calculate_metrics,
            )

            # Decode order and rotation
            order_rotation_pairs, _ = _decode_order_vector(
                vector, all_shapes, valid_rotations=[0, 30, 60, 90]
            )

            # Extract shape indices and rotations in order
            shape_indices = [pair[0] for pair in order_rotation_pairs]
            rotations = [pair[1] for pair in order_rotation_pairs]

            # Get shapes in the optimized order
            ordered_shapes = [all_shapes[idx] for idx in shape_indices]

            # Perform greedy placement
            placements = _adaptive_greedy_placement(
                ordered_shapes,
                rotations,
                sheet_width=SHEET_WIDTH,
                sheet_height=SHEET_HEIGHT,
                spacing=SPACING,
                grid_size=hybrid_params["grid_size"],
            )

            # Calculate metrics
            utilization, compactness, fitness = _calculate_metrics(
                placements,
                sheet_width=SHEET_WIDTH,
                sheet_height=SHEET_HEIGHT,
                compactness_weight=hybrid_params["compactness_weight"],
            )

            # Return negative fitness for minimization
            return -fitness

        hybrid_algo = HybridNestingAlgorithm(
            sheet_width=SHEET_WIDTH,
            sheet_height=SHEET_HEIGHT,
            spacing=SPACING,
            grid_size=hybrid_params["grid_size"],
            compactness_weight=hybrid_params["compactness_weight"],
            external_objective=external_objective_function if workers > 1 else None,
        )

        hybrid_start_time = time.time()
        hybrid_placements, hybrid_metrics = hybrid_algo.nest_shapes_hybrid(
            shapes=all_shapes,
            generations=hybrid_params["generations"],
            population_size=hybrid_params["population_size"],
            rotation_increment=hybrid_params["rotation_increment"],
            verbose=True,
            max_rotation=90,  # Set maximum rotation to 90 degrees (symmetric shapes)
            workers=workers,  # Set worker count for parallel processing
            seed=42,  # Set seed for reproducibility
        )
        hybrid_elapsed = time.time() - hybrid_start_time

        hybrid_sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        hybrid_total_area = sum(p.polygon.area for p in hybrid_placements)
        hybrid_utilization = (hybrid_total_area / hybrid_sheet_area) * 100

        print("\nHybrid Algorithm Results:")
        print(f"Shapes placed: {len(hybrid_placements)} / {len(all_shapes)}")
        print(f"Utilization: {hybrid_utilization:.2f}%")
        print(f"Time: {hybrid_elapsed:.1f} seconds")

        # Calculate speed-up compared to single worker
        base_time = 1
        if workers == 1:
            base_time = hybrid_elapsed
            speedup = 1.0
        else:
            speedup = base_time / hybrid_elapsed if hybrid_elapsed > 0 else 0

        print(f"Speed-up factor: {speedup:.2f}x")

        # Calculate compactness
        if hybrid_placements:
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
            hybrid_compactness = (
                (hybrid_total_area / bounding_box_area) * 100
                if bounding_box_area > 0
                else 0
            )
            print(f"Compactness: {hybrid_compactness:.2f}%")
        else:
            hybrid_compactness = 0

        # Save visualization
        hybrid_viz_file = f"test_hybrid_{workers}workers_result.png"
        hybrid_algo.visualize_nesting(hybrid_placements, filename=hybrid_viz_file)
        print(f"✓ Visualization saved to: {hybrid_viz_file}")

        # Save results
        results.append(
            {
                "workers": workers,
                "placements": hybrid_placements,
                "parts_placed": len(hybrid_placements),
                "utilization": hybrid_utilization,
                "compactness": hybrid_compactness,
                "time": hybrid_elapsed,
                "speedup": speedup,
            }
        )

    # Show progress visualization for best result
    best_result = max(results, key=lambda r: r["parts_placed"])
    progress_file = f"hybrid_progress_{best_result['workers']}workers.png"
    hybrid_algo.visualize_progress(hybrid_metrics, filename=progress_file)
    print(f"\n✓ Progress visualization saved to: {progress_file}")

    # ======= Parallel Performance Summary =======
    print("\n" + "=" * 70)
    print("PARALLEL PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\nWorkers | Parts Placed | Utilization | Compactness | Time (s) | Speed-up")
    print("-" * 80)

    for r in results:
        print(
            f"{r['workers']:7} | {r['parts_placed']:12} | {r['utilization']:10.2f}% | "
            f"{r['compactness']:10.2f}% | {r['time']:7.1f} | {r['speedup']:7.2f}x"
        )

    # Generate G-code for best result
    best_algo_name = f"hybrid_{best_result['workers']}workers"
    best_placements = best_result["placements"]

    if best_placements:
        print(
            f"\nGenerating G-code for best result ({best_result['workers']} workers)..."
        )
        gcode_generator = GCodeGenerator()

        try:
            output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                original_gcodes, best_placements, parser
            )

            output_file = f"test_{best_algo_name}_output.nc"
            with open(output_file, "w") as f:
                f.write(output_gcode)

            print(f"✓ G-code saved to: {output_file}")
        except Exception as e:
            print(f"⚠ G-code generation encountered an issue: {str(e)}")

    print("\nTest complete!")


if __name__ == "__main__":
    test_parallel_hybrid_nesting()
