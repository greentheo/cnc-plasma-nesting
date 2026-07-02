#!/usr/bin/env python3
"""
Quick test script for DE optimization on geodesic shapes.
Uses fewer shapes and generations for faster testing.
"""

import sys
import os
import time
from pathlib import Path

# Import the DE-enhanced shappack module
from shappack_de import DENetworkingAlgorithm
from shappack import GCodeParser, GCodeGenerator


def test_geodesic_de():
    """Run a quick DE optimization test on geodesic shapes."""

    # Configure nesting parameters
    SHEET_WIDTH = 48 * 25.4  # mm (48 inches)
    SHEET_HEIGHT = 96 * 25.4  # mm (96 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("GEODESIC DE OPTIMIZATION - QUICK TEST")
    print("=" * 70)
    print(f"\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm")
    print(f"Spacing: {SPACING} mm\n")

    # Test with just 2 files and fewer copies
    test_files = [
        ("geodesic_6_top.nc", 25),
        ("geodesic_6_bottom.nc", 25),  # Only 3 copies
        ("geodesic_5_top.nc", 5),
        ("geodesic_5_bottom.nc", 5),
        ("geodesic_4_top.nc", 15),  # Only 3 copies
        ("geodesic_4_bottom.nc", 15),
    ]

    # Parse files
    parser = GCodeParser()
    all_shapes = []

    print("Parsing G-code files...")
    for idx, (file_path, max_count) in enumerate(test_files):
        if not os.path.exists(file_path):
            print(f"✗ Missing: {file_path}")
            continue

        print(f"✓ Found: {file_path}")
        with open(file_path, "r") as f:
            gcode_text = f.read()

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

    # Create DE nesting algorithm
    nesting_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    # Run DE optimization with minimal parameters
    print("\n" + "=" * 70)
    print("RUNNING DE OPTIMIZATION")
    print("=" * 70)

    de_params = {
        "generations": 100,  # Only 10 generations for quick test
        "population_size": 100,  # Smaller population
        "rotation_increment": 15.0,  # Larger increment for speed
    }

    print(f"Generations: {de_params['generations']}")
    print(f"Population: {de_params['population_size']}")
    print(f"Rotation increment: {de_params['rotation_increment']}°")
    print(f"Testing rotation range: 0-90° (shapes are symmetrical)")
    print(f"Rotation angles to test: 0°, 45°, 90°")
    print()

    start_time = time.time()

    placements, metrics = nesting_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Already duplicated
        generations=de_params["generations"],
        population_size=de_params["population_size"],
        rotation_increment=de_params["rotation_increment"],
        verbose=True,
    )

    elapsed = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if placements:
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        print(f"Shapes placed: {len(placements)} / {len(all_shapes)}")
        print(f"Utilization: {utilization:.2f}%")
        print(f"Time: {elapsed:.1f} seconds")

        # Save visualization
        viz_file = "test_geodesic_de.png"
        nesting_algo.visualize_nesting(placements, filename=viz_file)
        print(f"\n✓ Visualization saved to: {viz_file}")

        # Generate G-code output
        print("\nGenerating G-code output...")
        gcode_generator = GCodeGenerator()

        # Generate the nested G-code with optimal travel
        try:
            # Collect original G-code texts for the generator
            original_gcodes = []
            for idx, (file_path, _) in enumerate(test_files):
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        original_gcodes.append(f.read())

            # The generator needs original gcodes as first parameter
            output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                original_gcodes, placements, parser
            )

            # Save the G-code to file
            output_file = "test_geodesic_de_output.nc"
            with open(output_file, "w") as f:
                f.write(output_gcode)

            print(f"✓ G-code saved to: {output_file}")

            # Show some statistics about the output
            lines = output_gcode.split("\n")
            print(f"  Total G-code lines: {len(lines)}")
            print(f"  Total shapes in output: {len(placements)}")

        except Exception as e:
            print(f"⚠ G-code generation encountered an issue: {str(e)}")
            print("  Note: The shapes were still successfully nested and visualized.")

    else:
        print("No shapes could be placed!")

    print("\nTest complete!")


if __name__ == "__main__":
    test_geodesic_de()
