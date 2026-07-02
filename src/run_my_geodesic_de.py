#!/usr/bin/env python3
"""
Script to run differential evolution optimization on YOUR actual geodesic NC files.
This will work with any NC files you have, not just samples.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Import the modules
from shappack_de import DENetworkingAlgorithm
from shappack import GCodeParser, Shape, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
import numpy as np


def analyze_nc_file(file_path: str, parser: GCodeParser) -> Dict:
    """Analyze an NC file and return shape information."""
    with open(file_path, "r") as f:
        gcode_text = f.read()

    shapes = parser.parse_gcode(gcode_text)

    info = {
        "file_path": file_path,
        "shapes": shapes,
        "shape_count": len(shapes),
        "total_points": sum(len(s.points) for s in shapes),
        "shape_types": [],
    }

    for shape in shapes:
        num_points = len(shape.points)
        if num_points == 3:
            shape_type = "Triangle"
        elif num_points == 4:
            shape_type = "Quadrilateral"
        elif num_points == 5:
            shape_type = "Pentagon"
        elif num_points == 6:
            shape_type = "Hexagon"
        else:
            shape_type = f"{num_points}-sided"

        xs = [p.x for p in shape.points]
        ys = [p.y for p in shape.points]

        info["shape_types"].append(
            {
                "type": shape_type,
                "points": num_points,
                "width": max(xs) - min(xs) if xs else 0,
                "height": max(ys) - min(ys) if ys else 0,
                "area_estimate": ((max(xs) - min(xs)) * (max(ys) - min(ys)))
                if xs and ys
                else 0,
            }
        )

    return info


def visualize_input_shapes(
    file_infos: List[Dict], output_file: str = "input_shapes_analysis.png"
):
    """Create visualization of all input shapes before optimization."""
    num_files = len(file_infos)
    cols = min(3, num_files)
    rows = (num_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if num_files == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, info in enumerate(file_infos):
        if idx >= len(axes):
            break

        ax = axes[idx] if num_files > 1 else axes[0]

        # Draw each shape in the file
        colors = plt.cm.Set3(np.linspace(0, 1, 12))

        for shape_idx, shape in enumerate(info["shapes"]):
            if shape.points:
                coords = [(p.x, p.y) for p in shape.points]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                patch = MPLPolygon(
                    coords,
                    alpha=0.6,
                    facecolor=colors[shape_idx % len(colors)],
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(patch)

                # Add shape type label
                xs = [p.x for p in shape.points]
                ys = [p.y for p in shape.points]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2

                shape_info = info["shape_types"][shape_idx]
                ax.text(
                    cx,
                    cy,
                    shape_info["type"],
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

                ax.set_xlim(min(xs) - 10, max(xs) + 10)
                ax.set_ylim(min(ys) - 10, max(ys) + 10)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{Path(info['file_path']).name}\n({info['shape_count']} shape(s))"
        )

    # Hide unused subplots
    for idx in range(num_files, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Input Shapes Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Input shapes visualization saved to: {output_file}")
    plt.show()


def run_my_geodesic_de():
    """Run DE optimization on your actual geodesic NC files."""

    # Configure nesting parameters
    SHEET_WIDTH = 48 * 25.4  # mm (48 inches)
    SHEET_HEIGHT = 96 * 25.4  # mm (96 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("DIFFERENTIAL EVOLUTION OPTIMIZATION FOR YOUR NC FILES")
    print("=" * 70)
    print(
        f'\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm ({SHEET_WIDTH / 25.4:.0f}" x {SHEET_HEIGHT / 25.4:.0f}")'
    )
    print(f"Spacing: {SPACING} mm\n")

    # Find all NC files in current directory
    nc_files = list(Path(".").glob("*.nc"))

    if not nc_files:
        print("❌ No .nc files found in current directory!")
        print("\nPlease ensure your NC files are in the same directory as this script.")
        return

    print("Found NC files:")
    print("-" * 40)
    for nc_file in nc_files:
        print(f"  • {nc_file}")

    # Ask user which files to use and how many copies
    print("\n" + "=" * 70)
    print("FILE SELECTION")
    print("=" * 70)

    selected_files = []

    print("\nFor each file, enter the number of copies you want (0 to skip):")
    for nc_file in nc_files:
        try:
            count = input(f"  {nc_file} - Number of copies (default 1): ").strip()
            count = int(count) if count else 1
            if count > 0:
                selected_files.append((str(nc_file), count))
        except ValueError:
            print(f"    Invalid input, skipping {nc_file}")

    if not selected_files:
        print("\n❌ No files selected!")
        return

    # Analyze selected files
    print("\n" + "=" * 70)
    print("ANALYZING INPUT FILES")
    print("=" * 70)

    parser = GCodeParser()
    file_infos = []
    all_shapes = []

    for file_path, max_count in selected_files:
        info = analyze_nc_file(file_path, parser)
        file_infos.append(info)

        print(f"\n{file_path}:")
        print(f"  • Shapes found: {info['shape_count']}")

        for shape_info in info["shape_types"]:
            print(
                f"    - {shape_info['type']}: {shape_info['width']:.1f} x {shape_info['height']:.1f} mm"
            )

        print(f"  • Copies requested: {max_count}")

        # Create copies for optimization
        for copy_num in range(max_count):
            for shape_idx, shape in enumerate(info["shapes"]):
                shape_copy = Shape(
                    points=shape.points.copy(),
                    gcode_lines=shape.gcode_lines.copy(),
                    original_idx=len(file_infos) - 1,
                )
                all_shapes.append(shape_copy)

    total_shapes = len(all_shapes)
    print(f"\nTotal shape instances to place: {total_shapes}")

    # Visualize input shapes
    print("\n" + "=" * 70)
    print("VISUALIZING INPUT SHAPES")
    print("=" * 70)
    visualize_input_shapes(file_infos, "input_shapes_analysis.png")

    # Get optimization parameters from user
    print("\n" + "=" * 70)
    print("OPTIMIZATION PARAMETERS")
    print("=" * 70)

    print("\nDefault parameters (press Enter to use defaults):")

    try:
        gens = input("  Generations (default 50): ").strip()
        generations = int(gens) if gens else 50
    except:
        generations = 50

    try:
        pop = input("  Population size (default 30): ").strip()
        population_size = int(pop) if pop else 30
    except:
        population_size = 30

    try:
        rot = input("  Rotation increment in degrees (default 15): ").strip()
        rotation_increment = float(rot) if rot else 15.0
    except:
        rotation_increment = 15.0

    print(f"\nUsing parameters:")
    print(f"  • Generations: {generations}")
    print(f"  • Population size: {population_size}")
    print(f"  • Rotation increment: {rotation_increment}°")
    print(
        f"  • Possible rotation angles: {list(range(0, 360, int(rotation_increment)))}"
    )

    # Create nesting algorithm
    nesting_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    # Run optimization
    print("\n" + "=" * 70)
    print("RUNNING DIFFERENTIAL EVOLUTION OPTIMIZATION")
    print("=" * 70)
    print("This may take a few moments...\n")

    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Already created copies
        generations=generations,
        population_size=population_size,
        rotation_increment=rotation_increment,
        mutation=(0.5, 1.0),
        recombination=0.7,
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    if placements:
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        print(f"\n✓ Optimization Complete!")
        print(f"  • Shapes placed: {len(placements)} / {total_shapes}")
        print(f"  • Sheet utilization: {utilization:.2f}%")
        print(f"  • Time taken: {elapsed_time:.1f} seconds")
        print(f"  • Time per generation: {elapsed_time / generations:.2f} seconds")

        # Count shapes by file
        shape_counts = {}
        for p in placements:
            idx = p.shape.original_idx
            shape_counts[idx] = shape_counts.get(idx, 0) + 1

        print("\nShapes placed by file:")
        for idx, (file_path, requested) in enumerate(selected_files):
            placed = shape_counts.get(idx, 0)
            percentage = (placed / requested * 100) if requested > 0 else 0
            print(f"  • {file_path}: {placed}/{requested} ({percentage:.0f}%)")

        # Rotation analysis
        rotations = [p.rotation for p in placements]
        unique_rotations = sorted(set(rotations))
        print(f"\nRotation angles used: {len(unique_rotations)} different")
        print(f"  Angles: {unique_rotations}")

        # Save outputs
        print("\n" + "=" * 70)
        print("SAVING OUTPUTS")
        print("=" * 70)

        # Save nesting layout
        viz_file = "optimized_nesting_layout.png"
        nesting_algo.visualize_nesting(placements, filename=viz_file)
        print(f"✓ Optimized layout saved to: {viz_file}")

        # Save progress plot
        if metrics and len(metrics) > 1:
            progress_file = "optimization_progress.png"
            nesting_algo.visualize_de_progress(metrics, output_file=progress_file)
            print(f"✓ Progress plot saved to: {progress_file}")

            # Show improvement
            initial_util = metrics[0].best_utilization * 100
            final_util = metrics[-1].best_utilization * 100
            improvement = final_util - initial_util

            print(f"\nOptimization improvement:")
            print(f"  • Initial: {initial_util:.2f}%")
            print(f"  • Final: {final_util:.2f}%")
            print(f"  • Improvement: {improvement:.2f}%")

        # Performance assessment
        print("\n" + "=" * 70)
        print("PERFORMANCE ASSESSMENT")
        print("=" * 70)

        if utilization > 80:
            print("🎉 Excellent utilization achieved!")
        elif utilization > 60:
            print("✓ Good utilization achieved")
        elif utilization > 40:
            print("⚠ Moderate utilization - consider:")
            print("  • Running more generations")
            print("  • Increasing population size")
            print("  • Using smaller rotation increment")
        else:
            print("❗ Low utilization - possible causes:")
            print("  • Shapes may be too large for sheet")
            print("  • Too few shapes to fill sheet")
            print("  • Consider different sheet size")

        success_rate = (len(placements) / total_shapes * 100) if total_shapes > 0 else 0
        if success_rate < 100:
            print(f"\n⚠ Only {success_rate:.0f}% of shapes were placed")
            print("  Consider using a larger sheet or fewer copies")

    else:
        print("\n❌ No shapes could be placed!")
        print("Check that your shapes fit within the sheet dimensions.")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print("  • input_shapes_analysis.png - Analysis of input shapes")
    print("  • optimized_nesting_layout.png - Final optimized layout")
    print("  • optimization_progress.png - Optimization progress over generations")


if __name__ == "__main__":
    run_my_geodesic_de()
