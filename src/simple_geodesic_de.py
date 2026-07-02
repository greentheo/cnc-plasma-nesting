#!/usr/bin/env python3
"""
Simple script to demonstrate DE optimization on geodesic shapes.
Uses fewer copies to avoid memory issues while still showing the optimization working.
"""

import sys
import os
import time
from pathlib import Path

# Import the modules
from shappack_de import DENetworkingAlgorithm
from shappack import GCodeParser


def simple_geodesic_de():
    """Run a simple DE optimization on geodesic shapes."""

    # Configure nesting parameters
    SHEET_WIDTH = 48 * 25.4  # mm (48 inches)
    SHEET_HEIGHT = 96 * 25.4  # mm (96 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("SIMPLE GEODESIC DE OPTIMIZATION DEMONSTRATION")
    print("=" * 70)
    print(
        f'\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm ({SHEET_WIDTH / 25.4:.0f}" x {SHEET_HEIGHT / 25.4:.0f}")'
    )
    print(f"Spacing: {SPACING} mm\n")

    # Use just a few shapes with limited copies for demonstration
    test_files = [
        ("geodesic_6_bottom.nc", 2),  # Just 2 copies
        ("geodesic_6_top.nc", 3),  # Just 3 copies
        ("geodesic_4_top.nc", 2),  # Just 2 copies
    ]

    # Check for files
    print("Checking for geodesic files...")
    print("-" * 40)

    available_files = []
    for file_path, count in test_files:
        if os.path.exists(file_path):
            available_files.append((file_path, count))
            print(f"✓ Found: {file_path} (will place {count} copies)")
        else:
            print(f"✗ Missing: {file_path}")

    if not available_files:
        print("\nNo files found. Creating sample files...")
        os.system("python create_geodesic_samples.py")
        print("\nPlease run this script again.")
        return

    # Parse files
    parser = GCodeParser()
    all_shapes = []
    file_info = []

    print("\n" + "=" * 70)
    print("PARSING G-CODE FILES")
    print("=" * 70)

    for idx, (file_path, max_count) in enumerate(available_files):
        with open(file_path, "r") as f:
            gcode_text = f.read()

        shapes = parser.parse_gcode(gcode_text)

        if not shapes:
            print(f"Warning: No shapes found in {file_path}")
            continue

        print(f"\n{file_path}:")
        print(f"  Parsed {len(shapes)} shape(s)")
        print(f"  Creating {max_count} copies")

        # Create copies
        for copy_num in range(max_count):
            for shape in shapes:
                shape_copy = type(shape)(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=idx,
                )
                all_shapes.append(shape_copy)

        file_info.append((file_path, max_count))

    if not all_shapes:
        print("\nNo shapes were parsed successfully.")
        return

    print(f"\nTotal shape instances: {len(all_shapes)}")

    # DE parameters - conservative for quick demo
    de_params = {
        "generations": 20,  # Quick optimization
        "population_size": 15,  # Small population
        "rotation_increment": 15.0,  # 15-degree increments
    }

    print("\n" + "=" * 70)
    print("DIFFERENTIAL EVOLUTION PARAMETERS")
    print("=" * 70)
    print(f"Generations: {de_params['generations']}")
    print(f"Population Size: {de_params['population_size']}")
    print(f"Rotation Increment: {de_params['rotation_increment']}°")
    print(f"Testing rotation range: 0-90° (shapes are symmetrical)")
    angles = list(range(0, 91, int(de_params["rotation_increment"])))
    print(f"Rotation angles to test: {angles}")

    # Create nesting algorithm
    nesting_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING OPTIMIZATION")
    print("=" * 70)
    print("This may take a moment...\n")

    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Already created copies
        generations=de_params["generations"],
        population_size=de_params["population_size"],
        rotation_increment=de_params["rotation_increment"],
        mutation=(0.5, 1.0),
        recombination=0.7,
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if placements:
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        print(f"\n✓ Optimization Complete!")
        print(f"  • Shapes placed: {len(placements)} / {len(all_shapes)}")
        print(f"  • Sheet utilization: {utilization:.2f}%")
        print(f"  • Time taken: {elapsed_time:.1f} seconds")
        print(
            f"  • Time per generation: {elapsed_time / de_params['generations']:.2f} seconds"
        )

        # Count by file
        shape_counts = {}
        for p in placements:
            idx = p.shape.original_idx
            shape_counts[idx] = shape_counts.get(idx, 0) + 1

        print("\nShapes placed by type:")
        for idx, (file_path, requested) in enumerate(file_info):
            placed = shape_counts.get(idx, 0)
            print(f"  • {file_path}: {placed}/{requested} placed")

        # Rotation analysis
        rotations = [p.rotation for p in placements]
        unique_rotations = set(rotations)
        print(f"\nRotation angles used: {len(unique_rotations)} different")
        print(f"  Angles: {sorted(unique_rotations)}° (limited to 0-90° for symmetry)")

        # Save visualization
        print("\n" + "=" * 70)
        print("SAVING OUTPUTS")
        print("=" * 70)

        viz_file = "simple_geodesic_de_layout.png"
        nesting_algo.visualize_nesting(placements, filename=viz_file)
        print(f"✓ Layout saved to: {viz_file}")

        if metrics and len(metrics) > 1:
            progress_file = "simple_geodesic_de_progress.png"
            nesting_algo.visualize_de_progress(metrics, output_file=progress_file)
            print(f"✓ Progress plot saved to: {progress_file}")

            # Show improvement
            initial_util = metrics[0].best_utilization * 100
            final_util = metrics[-1].best_utilization * 100
            improvement = final_util - initial_util

            print(f"\nOptimization improvement:")
            print(f"  • Initial utilization: {initial_util:.2f}%")
            print(f"  • Final utilization: {final_util:.2f}%")
            print(f"  • Improvement: {improvement:.2f}%")

    else:
        print("\nNo shapes could be placed.")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nThis demonstrates:")
    print("  1. DE optimization finding good placements")
    print("  2. Using 15-degree rotation increments")
    print("  3. Tracking utilization over generations")
    print("  4. Measuring how well shapes pack on the sheet")
    print("\nFor production use, increase generations and population size.")


if __name__ == "__main__":
    simple_geodesic_de()
