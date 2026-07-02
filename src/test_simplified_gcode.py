#!/usr/bin/env python3
"""
Test script for simplified G-code generation.

This script loads a G-code file, transforms it using the simplified G-code generator,
and saves the result to a new file.
"""

import os
import sys
import argparse
from shappack import GCodeParser, GCodeGenerator, NestingAlgorithm, Shape, Placement


def main():
    parser = argparse.ArgumentParser(description="Test simplified G-code generation")
    parser.add_argument("--input", "-i", required=True, help="Input G-code file")
    parser.add_argument(
        "--output", "-o", default="simplified_output.nc", help="Output G-code file"
    )
    parser.add_argument(
        "--width", "-w", type=float, default=24, help="Sheet width in inches"
    )
    parser.add_argument(
        "--height", "-H", type=float, default=48, help="Sheet height in inches"
    )
    parser.add_argument(
        "--spacing", "-s", type=float, default=3, help="Spacing between parts in mm"
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Generate a visualization of the layout",
    )
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    # Convert sheet dimensions from inches to mm
    sheet_width = args.width * 25.4
    sheet_height = args.height * 25.4
    spacing = args.spacing

    # Read input file
    with open(args.input, "r") as f:
        original_gcode = f.read()

    # Initialize parser
    parser = GCodeParser()

    # Parse G-code
    shapes = parser.parse_gcode(original_gcode)
    if not shapes:
        print("Error: No shapes found in the input file.")
        sys.exit(1)

    print(f"Found {len(shapes)} shapes in the input file.")

    # Create algorithm
    algorithm = NestingAlgorithm(sheet_width, sheet_height, spacing)

    # Nest shapes
    print(f'Nesting shapes on {args.width}" x {args.height}" sheet...')
    placements = algorithm.nest_shapes_optimal(shapes)

    if not placements:
        print("Error: Failed to nest shapes.")
        sys.exit(1)

    print(f"Successfully placed {len(placements)} shapes.")

    # Generate simplified G-code
    print("Generating simplified G-code...")
    generator = GCodeGenerator()
    simplified_gcode = generator.generate_simplified_gcode(
        [original_gcode],  # List of original G-codes
        placements,  # List of placements
        parser,  # Parser with torch parameters
    )

    # Save output
    with open(args.output, "w") as f:
        f.write(simplified_gcode)

    print(f"Simplified G-code saved to {args.output}")

    # Generate visualization if requested
    if args.visualize:
        viz_file = os.path.splitext(args.output)[0] + ".png"
        print(f"Generating visualization to {viz_file}...")
        algorithm.visualize_nesting(placements, viz_file)

    # Test a simple manual placement
    print("\nTesting manual placement...")
    # Extract the first shape from the input
    shape = shapes[0]

    # Create a manual placement with rotation
    # First transform the shape to create a placement
    manual_polygon = shape.to_shapely_polygon()
    manual_placement = Placement(
        shape=shape, polygon=manual_polygon, x_offset=50, y_offset=50, rotation=45
    )  # x_offset=50mm, y_offset=50mm, rotation=45°

    # Generate G-code for just this placement
    manual_output = os.path.splitext(args.output)[0] + "_manual.nc"
    manual_gcode = generator.generate_simplified_gcode(
        [original_gcode],  # List of original G-codes
        [manual_placement],  # List with a single placement
        parser,  # Parser with torch parameters
    )

    # Save manual output
    with open(manual_output, "w") as f:
        f.write(manual_gcode)

    print(f"Manual placement G-code saved to {manual_output}")

    # Generate visualization for manual placement
    if args.visualize:
        manual_viz = os.path.splitext(manual_output)[0] + ".png"
        print(f"Generating visualization to {manual_viz}...")

        # Create a temporary algorithm with a small sheet size
        temp_algo = NestingAlgorithm(sheet_width / 2, sheet_height / 2, spacing)
        temp_algo.visualize_nesting([manual_placement], manual_viz)


if __name__ == "__main__":
    main()
