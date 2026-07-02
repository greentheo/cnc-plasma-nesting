#!/usr/bin/env python3
"""
A small test script for the simplified G-code generator.
This script tests the G-code generation with a focus on problematic cases
like coordinate order and negative values.
"""

import os
import sys
import math
import re
from typing import List, Tuple, Optional

# Import the necessary classes from shappack
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from shappack import (
        GCodeParser,
        GCodeGenerator,
        NestingAlgorithm,
        Shape,
        Placement,
    )
except ImportError:
    print(
        "Error: Could not import from shappack.py. Make sure it's in the same directory."
    )
    sys.exit(1)


def main():
    """Run a small test for the simplified G-code generator."""
    print("=== Testing Simplified G-code Generator ===\n")

    # Input file to test
    input_file = "geodesic_6_bottom.nc"
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please run this script from the directory containing the test file.")
        sys.exit(1)

    # Read the input file
    with open(input_file, "r") as f:
        original_gcode = f.read()

    # Create a parser and parse the G-code
    parser = GCodeParser()
    shapes = parser.parse_gcode(original_gcode)

    if not shapes:
        print("Error: No shapes found in the input file.")
        sys.exit(1)

    print(f"Found {len(shapes)} shapes in the input file.")

    # Create a simple placement with no rotation
    # This tests basic translation without rotation
    placement1 = Placement(
        shape=shapes[0],
        polygon=shapes[0].to_shapely_polygon(),
        x_offset=100,
        y_offset=100,
        rotation=0,
    )

    # Create a placement with rotation
    # This tests translation with rotation
    placement2 = Placement(
        shape=shapes[0],
        polygon=shapes[0].to_shapely_polygon(),
        x_offset=200,
        y_offset=200,
        rotation=45,
    )

    # Create the G-code generator
    generator = GCodeGenerator()

    # Test both placements
    placements = [placement1, placement2]

    # Generate G-code with the simplified generator
    simplified_gcode = generator.generate_simplified_gcode(
        [original_gcode], placements, parser
    )

    # Save the output
    output_file = "test_small_output.nc"
    with open(output_file, "w") as f:
        f.write(simplified_gcode)

    print(f"Simplified G-code saved to {output_file}")

    # Check for negative coordinates or other issues
    check_gcode_issues(output_file)


def check_gcode_issues(file_path: str):
    """Check the G-code file for common issues."""
    print("\n=== Checking G-code for Issues ===\n")

    issues_found = False
    line_number = 0

    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, 1):
            # Skip comments and empty lines
            if line.strip().startswith("(") or not line.strip():
                continue

            # Check for negative X coordinates
            x_match = re.search(r"X(-\d+\.?\d*)", line)
            if x_match:
                print(f"Line {line_number}: Negative X coordinate: {x_match.group(1)}")
                print(f"  Full line: {line.strip()}")
                issues_found = True

            # Check for negative Y coordinates
            y_match = re.search(r"Y(-\d+\.?\d*)", line)
            if y_match:
                print(f"Line {line_number}: Negative Y coordinate: {y_match.group(1)}")
                print(f"  Full line: {line.strip()}")
                issues_found = True

            # Check for incorrect order (Y before X)
            if re.search(r"G\d+.*Y.*X", line):
                print(f"Line {line_number}: Incorrect coordinate order (Y before X)")
                print(f"  Full line: {line.strip()}")
                issues_found = True

            # Check for missing coordinates in G1/G2/G3 commands
            if (
                line.strip().startswith("G1")
                or line.strip().startswith("G2")
                or line.strip().startswith("G3")
            ):
                if "X" not in line and "Y" not in line:
                    # Only flag if it's not a Z-only move
                    if "Z" not in line and not re.match(
                        r"G[123]\s+[^XYZ]*$", line.strip()
                    ):
                        print(
                            f"Line {line_number}: Missing X/Y coordinates in move command"
                        )
                        print(f"  Full line: {line.strip()}")
                        issues_found = True

            # Skip checking specific G-codes that don't require coordinates
            if line.strip() in ["G17", "G18", "G19", "G20", "G21", "G90", "G91", "G94"]:
                continue

    if not issues_found:
        print("✅ No issues found in the G-code file.")
    else:
        print("\n⚠️ Issues were found in the G-code file. Please check and fix them.")


if __name__ == "__main__":
    main()
