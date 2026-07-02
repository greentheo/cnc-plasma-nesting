#!/usr/bin/env python3
"""
Complete script to run differential evolution optimization on all 6 geodesic shapes.
Based on the configuration from shappack.py lines 846-925 with DE optimization.
"""

import sys
import os
import time
from pathlib import Path

# Import the DE-enhanced shappack module
from shappack_de import DENetworkingAlgorithm
from shappack import GCodeParser


def run_all_geodesic_de():
    """Run DE optimization on all geodesic shapes with specified parameters."""

    # Configure nesting parameters (from shappack.py)
    SHEET_WIDTH = 48 * 25.4  # mm (48 inches converted to mm)
    SHEET_HEIGHT = 96 * 25.4  # mm (96 inches converted to mm)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("ALL GEODESIC SHAPES - DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 70)
    print(f"\nSheet Dimensions: {SHEET_WIDTH:.1f} x {SHEET_HEIGHT:.1f} mm")
    print(f'                  ({SHEET_WIDTH / 25.4:.1f}" x {SHEET_HEIGHT / 25.4:.1f}")')
    print(f"Spacing: {SPACING} mm\n")

    # Define all 6 geodesic files with their quantities (from shappack.py)
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
        print("Please run: python create_geodesic_samples.py")
        return

    print(f"\nTotal shapes requested: {total_requested_shapes}")

    # DE optimization parameters - balanced for good results without taking too long
    de_params = {
        "generations": 50,  # Moderate number of generations
        "population_size": 30,  # Reasonable population size
        "mutation": (0.5, 1.0),  # Standard mutation range
        "recombination": 0.7,  # Standard crossover
        "strategy": "best1bin",  # Reliable strategy
        "rotation_increment": 15.0,  # 15-degree increments for good flexibility
    }

    print("\n" + "=" * 70)
    print("DIFFERENTIAL EVOLUTION PARAMETERS")
    print("=" * 70)
    print(f"Generations: {de_params['generations']}")
    print(f"Population Size: {de_params['population_size']}")
    print(f"Rotation Increment: {de_params['rotation_increment']}°")
    print(f"Strategy: {de_params['strategy']}")

    # Parse all files
    parser = GCodeParser()
    all_shapes = []
    shape_info = []

    print("\n" + "=" * 70)
    print("PARSING G-CODE FILES")
    print("=" * 70)

    for idx, (file_path, max_count) in enumerate(available_files):
        print(f"\nParsing: {file_path}")
        with open(file_path, "r") as f:
            gcode_text = f.read()

        shapes = parser.parse_gcode(gcode_text)

        # Create requested number of copies
        for copy_num in range(max_count):
            for shape in shapes:
                shape_copy = type(shape)(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=idx,
                )
                all_shapes.append(shape_copy)

        shape_info.append((file_path, len(shapes), max_count))
        print(f"  Found {len(shapes)} unique shape(s), creating {max_count} copies")

    print(f"\nTotal shape instances to optimize: {len(all_shapes)}")

    # Create DE nesting algorithm
    nesting_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    # Run DE optimization
    print("\n" + "=" * 70)
    print("STARTING DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 70)

    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Already duplicated shapes above
        generations=de_params["generations"],
        population_size=de_params["population_size"],
        mutation=de_params["mutation"],
        recombination=de_params["recombination"],
        strategy=de_params["strategy"],
        rotation_increment=de_params["rotation_increment"],
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    # Calculate and display results
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
        print(
            f"Time per generation: {elapsed_time / de_params['generations']:.2f} seconds"
        )

        print("\nShapes placed by type:")
        print("-" * 40)
        for idx, (file_path, unique_shapes, requested) in enumerate(shape_info):
            placed = shape_counts.get(idx, 0)
            percentage = (placed / requested * 100) if requested > 0 else 0
            status = "✓" if placed == requested else "⚠"
            print(f"{status} {file_path}:")
            print(f"   Requested: {requested}, Placed: {placed} ({percentage:.0f}%)")

        # Save visualizations
        print("\n" + "=" * 70)
        print("SAVING OUTPUTS")
        print("=" * 70)

        # Save nesting layout
        viz_file = "all_geodesic_de_layout.png"
        nesting_algo.visualize_nesting(placements, filename=viz_file)
        print(f"✓ Layout visualization saved to: {viz_file}")

        # Save progress plot
        if metrics:
            progress_file = "all_geodesic_de_progress.png"
            nesting_algo.visualize_de_progress(metrics, output_file=progress_file)
            print(f"✓ Progress plot saved to: {progress_file}")

        # Optimization insights
        if metrics:
            print("\n" + "=" * 70)
            print("OPTIMIZATION INSIGHTS")
            print("=" * 70)

            utilizations = [m.best_utilization * 100 for m in metrics]
            final_util = utilizations[-1]

            print(f"\nUtilization progression:")
            print(f"  Generation 1: {utilizations[0]:.2f}%")
            if len(utilizations) >= 10:
                print(f"  Generation 10: {utilizations[9]:.2f}%")
            if len(utilizations) >= 25:
                print(f"  Generation 25: {utilizations[24]:.2f}%")
            if len(utilizations) >= de_params["generations"]:
                print(f"  Final (Gen {de_params['generations']}): {final_util:.2f}%")

            improvement = final_util - utilizations[0]
            print(f"\nTotal improvement: {improvement:.2f}%")

            # Find when optimization plateaued
            target_util = final_util * 0.95
            plateau_gen = next(
                (i for i, u in enumerate(utilizations) if u >= target_util),
                len(metrics),
            )
            print(f"95% of final utilization reached at generation {plateau_gen + 1}")

            if plateau_gen < len(metrics) * 0.5:
                print(
                    "\n💡 Optimization converged early - could use fewer generations next time"
                )
            elif improvement < 2.0:
                print(
                    "\n💡 Limited improvement - consider adjusting parameters or rotation increment"
                )
            else:
                print("\n✓ Good optimization performance achieved")

        # Summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        success_rate = (len(placements) / len(all_shapes)) * 100
        print(f"\n📊 Final Statistics:")
        print(
            f'   • Sheet Size: {SHEET_WIDTH / 25.4:.0f}" x {SHEET_HEIGHT / 25.4:.0f}"'
        )
        print(f"   • Utilization: {utilization:.2f}%")
        print(
            f"   • Success Rate: {success_rate:.1f}% ({len(placements)}/{len(all_shapes)} shapes)"
        )
        print(f"   • Computation Time: {elapsed_time:.1f} seconds")
        print(
            f"   • Rotation Angles Used: {len(set(p.rotation for p in placements))} different"
        )

        if utilization > 80:
            print("\n🎉 Excellent utilization achieved!")
        elif utilization > 60:
            print("\n✓ Good utilization achieved")
        elif utilization > 40:
            print("\n⚠ Moderate utilization - consider parameter tuning")
        else:
            print("\n❗ Low utilization - shapes may be too large for sheet")

    else:
        print("\n❗ No shapes could be placed. Check sheet size and shape dimensions.")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print("  • all_geodesic_de_layout.png - Visual layout of nested shapes")
    print("  • all_geodesic_de_progress.png - Optimization progress over generations")


if __name__ == "__main__":
    # First ensure sample files exist
    if not os.path.exists("geodesic_6_bottom.nc"):
        print("Creating sample geodesic files first...")
        os.system("python create_geodesic_samples.py")
        print()

    # Run the optimization
    run_all_geodesic_de()
