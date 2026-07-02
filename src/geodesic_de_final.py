#!/usr/bin/env python3
"""
Final Differential Evolution Optimizer for Geodesic Shape Nesting
Optimizes placement and rotation (0-90 degrees) for CNC cutting
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

# Import the modules
from shappack_de import DENetworkingAlgorithm
from shappack import GCodeParser, GCodeGenerator


def run_geodesic_de_optimization():
    """Run DE optimization on geodesic shapes with 0-90 degree rotation."""

    # Configure nesting parameters
    SHEET_WIDTH = 48 * 25.4  # mm (48 inches)
    SHEET_HEIGHT = 96 * 25.4  # mm (96 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("GEODESIC SHAPE DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 70)
    print(f"\nSheet Dimensions: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm")
    print(f'                  ({SHEET_WIDTH / 25.4:.0f}" x {SHEET_HEIGHT / 25.4:.0f}")')
    print(f"Spacing: {SPACING} mm\n")

    # Define the 6 geodesic files with quantities (from shappack.py lines 846-925)
    geodesic_files = [
        ("geodesic_6_bottom.nc", 10),
        ("geodesic_6_top.nc", 25),
        ("geodesic_5_top.nc", 5),
        ("geodesic_5_bottom.nc", 5),
        ("geodesic_4_top.nc", 15),
        ("geodesic_4_bottom.nc", 5),
    ]

    # Check for available files
    print("Checking for geodesic files...")
    print("-" * 40)

    available_files = []
    missing_files = []

    for file_path, count in geodesic_files:
        if os.path.exists(file_path):
            available_files.append((file_path, count))
            print(f"✓ Found: {file_path} (requesting {count} copies)")
        else:
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")

    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} files missing")
        if not available_files:
            print(
                "\n❌ No files found! Please ensure NC files are in current directory."
            )
            return

    # Parse files and create shape instances
    parser = GCodeParser()
    all_shapes = []
    total_requested = 0

    print("\n" + "=" * 70)
    print("PARSING G-CODE FILES")
    print("=" * 70)

    for file_idx, (file_path, max_count) in enumerate(available_files):
        with open(file_path, "r") as f:
            gcode_text = f.read()

        shapes = parser.parse_gcode(gcode_text)

        if not shapes:
            print(f"⚠ Warning: No shapes found in {file_path}")
            continue

        print(f"\n{file_path}:")
        print(f"  Parsed {len(shapes)} unique shape(s)")
        print(f"  Creating {max_count} copies for optimization")

        # Create requested number of copies
        for copy_num in range(max_count):
            for shape in shapes:
                shape_copy = type(shape)(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=file_idx,
                )
                all_shapes.append(shape_copy)

        total_requested += max_count

    if not all_shapes:
        print("\n❌ No shapes were parsed successfully!")
        return

    print(f"\nTotal shape instances to optimize: {len(all_shapes)}")

    # DE optimization parameters
    print("\n" + "=" * 70)
    print("DIFFERENTIAL EVOLUTION PARAMETERS")
    print("=" * 70)

    # Optimized parameters for production use
    de_params = {
        "generations": 100,  # Good balance of quality vs time
        "population_size": 50,  # Reasonable population size
        "rotation_increment": 15.0,  # 15-degree increments (0, 15, 30, 45, 60, 75, 90)
        "mutation": (0.5, 1.0),  # Standard mutation range
        "recombination": 0.7,  # Standard crossover probability
        "strategy": "best1bin",  # Reliable strategy
    }

    print(f"Generations: {de_params['generations']}")
    print(f"Population Size: {de_params['population_size']}")
    print(f"Rotation Increment: {de_params['rotation_increment']}°")
    print(f"Rotation Range: 0-90° (exploiting shape symmetry)")

    rotation_angles = list(range(0, 91, int(de_params["rotation_increment"])))
    print(f"Rotation Angles: {rotation_angles}°")
    print(f"Strategy: {de_params['strategy']}")

    # Create nesting algorithm
    nesting_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    # Run DE optimization
    print("\n" + "=" * 70)
    print("RUNNING DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 70)
    print("Optimizing placement and rotation for maximum sheet utilization...")
    print("(This may take a few minutes depending on the number of shapes)\n")

    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Already duplicated shapes above
        generations=de_params["generations"],
        population_size=de_params["population_size"],
        rotation_increment=de_params["rotation_increment"],
        mutation=de_params["mutation"],
        recombination=de_params["recombination"],
        strategy=de_params["strategy"],
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    # Process and display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    if placements:
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        print(f"\n✓ Optimization Complete!")
        print(f"  • Shapes placed: {len(placements)} / {len(all_shapes)}")
        print(f"  • Sheet utilization: {utilization:.2f}%")
        print(f"  • Optimization time: {elapsed_time:.1f} seconds")
        print(
            f"  • Time per generation: {elapsed_time / de_params['generations']:.2f} seconds"
        )

        # Count shapes by type
        shape_counts = {}
        for p in placements:
            idx = p.shape.original_idx
            shape_counts[idx] = shape_counts.get(idx, 0) + 1

        print("\nShapes placed by file:")
        for idx, (file_path, requested) in enumerate(available_files):
            placed = shape_counts.get(idx, 0)
            percentage = (placed / requested * 100) if requested > 0 else 0
            status = "✓" if placed == requested else "⚠"
            print(f"  {status} {file_path}: {placed}/{requested} ({percentage:.0f}%)")

        # Analyze rotations used
        rotations = [p.rotation for p in placements]
        unique_rotations = sorted(set(rotations))
        print(f"\nRotation angles used: {unique_rotations}°")
        print(f"  ({len(unique_rotations)} different angles, all ≤ 90°)")

        # Save visualization
        print("\n" + "=" * 70)
        print("SAVING OUTPUT FILES")
        print("=" * 70)

        # Save nesting layout visualization
        viz_file = "geodesic_nested_layout.png"
        nesting_algo.visualize_nesting(placements, filename=viz_file)
        print(f"✓ Layout visualization saved to: {viz_file}")

        # Save progress plot if we have metrics
        if metrics and len(metrics) > 1:
            progress_file = "geodesic_optimization_progress.png"
            nesting_algo.visualize_de_progress(metrics, output_file=progress_file)
            print(f"✓ Progress plot saved to: {progress_file}")

            # Analyze optimization progress
            initial_util = metrics[0].best_utilization * 100
            final_util = metrics[-1].best_utilization * 100
            improvement = final_util - initial_util

            print(f"\nOptimization improvement:")
            print(f"  • Initial utilization: {initial_util:.2f}%")
            print(f"  • Final utilization: {final_util:.2f}%")
            print(f"  • Total improvement: {improvement:.2f}%")

            # Find convergence point
            target_util = final_util * 0.95
            for i, m in enumerate(metrics):
                if m.best_utilization * 100 >= target_util:
                    print(f"  • 95% of final utilization reached at generation {i + 1}")
                    break

        # Generate G-code output
        print("\nGenerating G-code output...")
        gcode_generator = GCodeGenerator()

        try:
            # Collect original G-code texts
            original_gcodes = []
            for file_path, _ in available_files:
                with open(file_path, "r") as f:
                    original_gcodes.append(f.read())

            # Generate nested G-code with optimal travel
            output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                original_gcodes, placements, parser
            )

            # Save G-code file
            output_file = "geodesic_nested_output.nc"
            with open(output_file, "w") as f:
                f.write(output_gcode)

            print(f"✓ G-code saved to: {output_file}")

            # Show G-code statistics
            lines = output_gcode.split("\n")
            print(f"  • Total G-code lines: {len(lines)}")
            print(f"  • Total shapes in output: {len(placements)}")

        except Exception as e:
            print(f"⚠ G-code generation issue: {str(e)}")
            print("  Shapes were still successfully nested and visualized.")

        # Performance assessment
        print("\n" + "=" * 70)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 70)

        if utilization > 80:
            print("🎉 Excellent utilization achieved!")
            print("   Your sheets are being used very efficiently.")
        elif utilization > 60:
            print("✓ Good utilization achieved")
            print("   This is a solid nesting result.")
        elif utilization > 40:
            print("⚠ Moderate utilization")
            print("   Consider:")
            print("   • Running more generations")
            print("   • Using smaller rotation increment (e.g., 5°)")
            print("   • Adjusting sheet size if possible")
        else:
            print("❗ Low utilization")
            print("   Possible causes:")
            print("   • Shapes may be too large for the sheet")
            print("   • Too few shapes to efficiently fill the sheet")
            print("   • Consider using a different sheet size")

        success_rate = (len(placements) / len(all_shapes) * 100) if all_shapes else 0
        if success_rate < 100:
            print(f"\n⚠ Only {success_rate:.0f}% of requested shapes were placed")
            print("   Consider using a larger sheet or requesting fewer copies")

        # Final summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n📊 Final Statistics:")
        print(
            f'   • Sheet size: {SHEET_WIDTH / 25.4:.0f}" x {SHEET_HEIGHT / 25.4:.0f}"'
        )
        print(f"   • Utilization: {utilization:.2f}%")
        print(f"   • Shapes placed: {len(placements)}/{len(all_shapes)}")
        print(f"   • Unique rotation angles used: {len(unique_rotations)}")
        print(f"   • Total optimization time: {elapsed_time:.1f} seconds")

        print("\n📁 Output Files Created:")
        print("   • geodesic_nested_layout.png - Visual layout of nested shapes")
        print("   • geodesic_optimization_progress.png - Optimization progress chart")
        print("   • geodesic_nested_output.nc - G-code for CNC cutting")

    else:
        print("\n❌ No shapes could be placed!")
        print("   Check that your shapes fit within the sheet dimensions.")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Ensure sample files exist
    if not any(Path(".").glob("geodesic_*.nc")):
        print("No geodesic NC files found. Creating sample files...")
        os.system("python create_geodesic_samples.py")
        print()

    # Run the optimization
    run_geodesic_de_optimization()
