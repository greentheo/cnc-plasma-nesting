#!/usr/bin/env python3
"""
A simple test script to debug coordinate issues in the shappack.py G-code generator.

This script focuses on testing coordinate transformations, especially for single-coordinate
G-code commands like "G1 Y154.500." that might have trailing periods or other issues.
"""

import re
import math
import os


# Mock the basic classes needed for testing
class Shape:
    def __init__(self, center_x, center_y):
        self._center_x = center_x
        self._center_y = center_y

    def center(self):
        return (self._center_x, self._center_y)


class Placement:
    def __init__(self, x_offset=0, y_offset=0, rotation=0):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.rotation = rotation


def test_coordinate_transformation(
    input_line,
    x_offset=0,
    y_offset=0,
    rotation=0,
    center_x=0,
    center_y=0,
    last_x=0,
    last_y=0,
):
    """Test coordinate transformation for a single G-code line."""
    print(f"\n=== Testing line: '{input_line}' ===")
    print(f"  Offsets: x={x_offset}, y={y_offset}, rotation={rotation}°")
    print(f"  Shape center: ({center_x}, {center_y})")
    print(f"  Last position: x={last_x}, y={last_y}")

    # Extract coordinates with flexible regex
    x_match = re.search(r"X([-+]?\d*\.?\d+)\.?", input_line)
    y_match = re.search(r"Y([-+]?\d*\.?\d+)\.?", input_line)

    print(f"  Regex matches: x_match={x_match}, y_match={y_match}")

    # Get coordinate values
    x = float(x_match.group(1)) if x_match else last_x
    y = float(y_match.group(1)) if y_match else last_y

    print(f"  Parsed coordinates: x={x}, y={y}")

    # Apply transformation
    if rotation != 0:
        # Rotate around center
        rad = math.radians(rotation)
        dx = x - center_x
        dy = y - center_y
        new_x = center_x + dx * math.cos(rad) - dy * math.sin(rad) + x_offset
        new_y = center_y + dx * math.sin(rad) + dy * math.cos(rad) + y_offset
    else:
        # Just translate
        new_x = x + x_offset
        new_y = y + y_offset

    print(f"  Transformed coordinates: new_x={new_x:.3f}, new_y={new_y:.3f}")

    # Update the line
    new_line = input_line
    if x_match:
        new_line = re.sub(r"X([-+]?\d*\.?\d+)\.?", f"X{new_x:.3f}", new_line)

    if y_match:
        new_line = re.sub(r"Y([-+]?\d*\.?\d+)\.?", f"Y{new_y:.3f}", new_line)

    print(f"  Result: '{new_line}'")
    return new_line


def main():
    """Run coordinate transformation tests."""
    print("=== Coordinate Transformation Test Suite ===\n")

    # Test cases for problematic coordinates
    test_cases = [
        # Normal coordinates
        "G1 X10 Y20",
        # Single coordinates
        "G1 X10",
        "G1 Y20",
        # Coordinates with trailing periods
        "G1 X10.",
        "G1 Y20.",
        "G1 Y154.500.",
        # Arc commands
        "G2 X30 Y40 I5 J10",
        "G3 X40 Y50 R15",
        # Mixed examples
        "G2 X30 Y40. I5 J10",
        "G1 Y154.500. X30",  # Y with trailing period
    ]

    # Test with different transformations
    transformations = [
        # (x_offset, y_offset, rotation, center_x, center_y)
        (0, 0, 0, 0, 0),  # No transformation
        (100, 200, 0, 0, 0),  # Translation only
        (0, 0, 45, 50, 50),  # Rotation only around (50,50)
        (100, 200, 90, 50, 50),  # Translation and rotation
    ]

    # Run tests
    for i, line in enumerate(test_cases):
        print(f"\n\n=== Test Case {i + 1}: '{line}' ===")

        for j, (x_offset, y_offset, rotation, center_x, center_y) in enumerate(
            transformations
        ):
            print(f"\n-- Transformation {j + 1} --")
            shape = Shape(center_x, center_y)
            placement = Placement(x_offset, y_offset, rotation)

            # Use last known position from previous command if needed
            last_x = 0
            last_y = 0

            # Process the test case
            result = test_coordinate_transformation(
                line, x_offset, y_offset, rotation, center_x, center_y, last_x, last_y
            )

            # Check specific edge cases
            if "Y154.500." in line:
                print(f"\n  ATTENTION: This is similar to the problematic line!")
                print(f"  - Original: '{line}'")
                print(f"  - Transformed: '{result}'")
                print(
                    f"  - Y coordinate should be {last_y + y_offset:.3f} with translation only"
                )


if __name__ == "__main__":
    main()
