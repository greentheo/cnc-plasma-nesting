#!/usr/bin/env python3
"""
Script to run differential evolution optimization on geodesic shapes.
Based on the configuration from shappack.py lines 846-925.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

# Import the DE-enhanced shappack module
from shappack_de import DENetworkingAlgorithm, nest_gcode_files_de
from shappack import GCodeParser, GCodeGenerator


def run_geodesic_de_optimization():
    """Run DE optimization on geodesic shapes with specified parameters."""

    # Configure nesting parameters (from shappack.py)
    SHEET_WIDTH = 48 * 25.4  # mm (48 inches converted to mm)
    SHEET_HEIGHT = 96 * 25.4  # mm (96 inches converted to mm)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("GEODESIC SHAPES DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 70)
    print(f"\nSheet Dimensions: {SHEET_WIDTH:.1f} x {SHEET_HEIGHT:.1f} mm")
    print(f'                  ({SHEET_WIDTH / 25.4:.1f}" x {SHEET_HEIGHT / 25.4:.1f}")')
    print(f"Spacing: {SPACING} mm")
    print()

    # Define the 6 geodesic files and their quantities
    geodesic_files = [
        ("geodesic_6_bottom.nc", 10),
        ("geodesic_6_top.nc", 25),
        ("geodesic_5_top.nc", 5),
        ("geodesic_5_bottom.nc", 5),
        ("geodesic_4_top.nc", 15),
        ("geodesic_4_bottom.nc", 5),
    ]

    # Collect available files
    available_files = []
    total_requested_shapes = 0

    print("Checking for geodesic files...")
    print("-" * 40)

    for file_path, count in geodesic_files:
        if count > 0 and os.path.exists(file_path):
            available_files.append((file_path, count))
            total_requested_shapes += count
            print(f"✓ Found: {file_path} (requesting {count} copies)")
        elif count > 0:
            print(f"✗ Missing: {file_path} (would place {count} copies)")

    if not available_files:
        print("\nError: No geodesic files found!")
        print("Please ensure the geodesic NC files are in the current directory.")
        return

    print(f"\nTotal shapes requested: {total_requested_shapes}")
    print()

    # DE optimization parameters
    print("=" * 70)
    print("DIFFERENTIAL EVOLUTION PARAMETERS")
    print("=" * 70)

    # You can adjust these parameters for better optimization
    de_params = {
        "generations": 100,  # Number of generations
        "population_size": 50,  # Population size
        "mutation": (0.5, 1.0),  # Mutation factor range
        "recombination": 0.7,  # Crossover probability
        "strategy": "best1bin",  # DE strategy
        "rotation_increment": 15.0,  # 15-degree rotation increments
        "seed": None,  # Set to a number for reproducible results
        "workers": 1,  # Number of parallel workers
    }

    print(f"Generations: {de_params['generations']}")
    print(f"Population Size: {de_params['population_size']}")
    print(f"Mutation Range: {de_params['mutation']}")
    print(f"Recombination: {de_params['recombination']}")
    print(f"Strategy: {de_params['strategy']}")
    print(f"Rotation Increment: {de_params['rotation_increment']}°")
    print()

    # Parse all files first to prepare shapes
    parser = GCodeParser()
    all_shapes = []
    shape_info = []

    print("=" * 70)
    print("PARSING G-CODE FILES")
    print("=" * 70)

    for idx, (file_path, max_count) in enumerate(available_files):
        print(f"\nParsing: {file_path}")
        with open(file_path, "r") as f:
            gcode_text = f.read()

        shapes = parser.parse_gcode(gcode_text)

        # We'll need to duplicate shapes based on requested count
        for _ in range(max_count):
            for shape in shapes:
                shape_copy = type(shape)(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=idx,
                )
                all_shapes.append(shape_copy)

        shape_info.append((file_path, len(shapes), max_count))
        print(
            f"  Found {len(shapes)} unique shape(s), will place up to {max_count} copies"
        )

    print(f"\nTotal shape instances to optimize: {len(all_shapes)}")

    # Create DE nesting algorithm
    print("\n" + "=" * 70)
    print("STARTING OPTIMIZATION")
    print("=" * 70)

    nesting_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    # Run DE optimization
    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Since we've already duplicated shapes
        generations=de_params["generations"],
        population_size=de_params["population_size"],
        mutation=de_params["mutation"],
        recombination=de_params["recombination"],
        strategy=de_params["strategy"],
        rotation_increment=de_params["rotation_increment"],
        seed=de_params["seed"],
        workers=de_params["workers"],
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    # Calculate final statistics
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    if placements:
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        # Count shapes by type
        shape_counts = {}
        for p in placements:
            idx = p.shape.original_idx
            if idx not in shape_counts:
                shape_counts[idx] = 0
            shape_counts[idx] += 1

        print(f"\nTotal shapes placed: {len(placements)} / {len(all_shapes)}")
        print(f"Sheet utilization: {utilization:.2f}%")
        print(f"Total area used: {total_area:.0f} mm²")
        print(f"Optimization time: {elapsed_time:.1f} seconds")

        print("\nShapes placed by type:")
        print("-" * 40)
        for idx, (file_path, unique_shapes, requested) in enumerate(shape_info):
            placed = shape_counts.get(idx, 0)
            percentage = (placed / requested * 100) if requested > 0 else 0
            status = "✓" if placed == requested else "⚠"
            print(f"{status} {file_path}:")
            print(f"   Requested: {requested}, Placed: {placed} ({percentage:.0f}%)")

        # Generate output G-code
        print("\n" + "=" * 70)
        print("GENERATING OUTPUT")
        print("=" * 70)

        gcode_generator = GCodeGenerator()
        output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
            placements, parser
        )

        # Save outputs
        output_base = "geodesic_de_nested"

        # Save G-code
        output_nc = f"{output_base}.nc"
        with open(output_nc, "w") as f:
            f.write(output_gcode)
        print(f"\n✓ G-code saved to: {output_nc}")

        # Save visualization
        viz_file = f"{output_base}_layout.png"
        nesting_algo.visualize_nesting(placements, show=False, output_file=viz_file)
        print(f"✓ Layout visualization saved to: {viz_file}")

        # Save progress plot
        if metrics:
            progress_file = f"{output_base}_progress.png"
            nesting_algo.visualize_de_progress(metrics, output_file=progress_file)
            print(f"✓ Progress plot saved to: {progress_file}")

        # Print optimization insights
        if metrics:
            print("\n" + "=" * 70)
            print("OPTIMIZATION INSIGHTS")
            print("=" * 70)

            # Find when optimization plateaued
            utilizations = [m.best_utilization * 100 for m in metrics]
            final_util = utilizations[-1]

            # Find generation where we reached 95% of final utilization
            target_util = final_util * 0.95
            plateau_gen = next(
                (i for i, u in enumerate(utilizations) if u >= target_util),
                len(metrics),
            )

            print(f"\nUtilization progression:")
            print(f"  Generation 1: {utilizations[0]:.2f}%")
            if len(utilizations) >= 10:
                print(f"  Generation 10: {utilizations[9]:.2f}%")
            if len(utilizations) >= 25:
                print(f"  Generation 25: {utilizations[24]:.2f}%")
            if len(utilizations) >= 50:
                print(f"  Generation 50: {utilizations[49]:.2f}%")
            print(f"  Final (Gen {len(metrics)}): {final_util:.2f}%")

            print(f"\n95% of final utilization reached at generation {plateau_gen + 1}")

            improvement = final_util - utilizations[0]
            print(f"Total improvement: {improvement:.2f}%")

            if plateau_gen < len(metrics) * 0.5:
                print(
                    "\n💡 Tip: Optimization converged early. You could use fewer generations."
                )
            elif improvement < 2.0:
                print("\n💡 Tip: Limited improvement seen. Consider:")
                print("   - Increasing population size")
                print("   - Adjusting mutation parameters")
                print("   - Using smaller rotation increments")

        # Compare with grid-based method
        print("\n" + "=" * 70)
        print("COMPARISON WITH GRID-BASED METHOD")
        print("=" * 70)

        print("\nRunning grid-based nesting for comparison...")

        # Reset shapes for grid-based method
        grid_shapes = []
        for file_path, max_count in available_files:
            with open(file_path, "r") as f:
                gcode_text = f.read()
            shapes = parser.parse_gcode(gcode_text)
            for shape in shapes:
                shape.original_idx = len(grid_shapes)
            grid_shapes.extend(shapes)

        # Run grid-based method
        from shappack import NestingAlgorithm

        grid_nesting = NestingAlgorithm(SHEET_WIDTH, SHEET_HEIGHT, SPACING)

        grid_start = time.time()
        # For grid method, we need to specify max_count differently
        max_copies_for_grid = max(count for _, count in available_files)
        grid_placements = grid_nesting.nest_shapes_optimal(
            grid_shapes, max_copies_for_grid
        )
        grid_time = time.time() - grid_start

        if grid_placements:
            grid_area = sum(p.polygon.area for p in grid_placements)
            grid_utilization = (grid_area / sheet_area) * 100

            print(f"\nGrid-based results:")
            print(f"  Shapes placed: {len(grid_placements)}")
            print(f"  Utilization: {grid_utilization:.2f}%")
            print(f"  Time: {grid_time:.1f}s")

            print(f"\nDE optimization results:")
            print(f"  Shapes placed: {len(placements)}")
            print(f"  Utilization: {utilization:.2f}%")
            print(f"  Time: {elapsed_time:.1f}s")

            print(f"\nImprovement:")
            util_improvement = utilization - grid_utilization
            shapes_improvement = len(placements) - len(grid_placements)

            if util_improvement > 0:
                print(f"  ✓ Utilization: +{util_improvement:.2f}%")
            else:
                print(f"  ✗ Utilization: {util_improvement:.2f}%")

            if shapes_improvement > 0:
                print(f"  ✓ Shapes placed: +{shapes_improvement}")
            else:
                print(f"  ✗ Shapes placed: {shapes_improvement}")

            time_factor = elapsed_time / grid_time if grid_time > 0 else 0
            print(f"  ⚠ Time factor: {time_factor:.1f}x")

            if util_improvement > 5:
                print("\n🎉 Significant improvement with DE optimization!")
            elif util_improvement > 0:
                print("\n✓ DE optimization provided modest improvements.")
            else:
                print("\n⚠ Grid-based method performed better for this case.")

    else:
        print(
            "\nNo shapes could be placed. Check your sheet size and shape dimensions."
        )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_geodesic_de_optimization()
