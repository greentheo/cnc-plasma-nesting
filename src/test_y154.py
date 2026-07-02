#!/usr/bin/env python3
"""
A very focused test script that examines the specific issue with Y154.500 G-code lines.
"""

import re


def test_y154_regex():
    """Test regex patterns for Y154.500 with and without trailing period."""

    test_lines = [
        "G1 Y154.500",
        "G1 Y154.500.",
        "G1 Y154.500 ",  # trailing space
        "G1 Y154.500. ",  # trailing period and space
    ]

    # Different regex patterns to test
    patterns = [
        r"Y(\d+\.\d+)",  # Basic - will miss trailing period
        r"Y(\d+\.\d+)\.*",  # Handle optional trailing period
        r"Y([-+]?\d*\.?\d+)\.?",  # More robust pattern
    ]

    print("=== Testing Y154.500 Regex Patterns ===\n")

    for i, pattern in enumerate(patterns):
        print(f"Pattern {i + 1}: {pattern}")
        for line in test_lines:
            match = re.search(pattern, line)
            if match:
                y_val = match.group(1)
                try:
                    y_float = float(y_val)
                    print(f"  '{line}' -> match: '{y_val}', float: {y_float}")
                except ValueError:
                    print(f"  '{line}' -> match: '{y_val}', INVALID FLOAT!")
            else:
                print(f"  '{line}' -> NO MATCH!")
        print()


def test_coordinate_transformation():
    """Test transforming G-code lines with Y154.500 in different ways."""

    lines = [
        "G1 Y154.500",
        "G1 Y154.500.",
    ]

    y_offset = 200  # Test with a Y offset of 200

    print("=== Testing Coordinate Transformation ===\n")

    for line in lines:
        print(f"Original line: '{line}'")

        # Extract Y value
        y_match = re.search(r"Y([-+]?\d*\.?\d+)\.?", line)
        if y_match:
            y_val = float(y_match.group(1))
            print(f"  Extracted Y value: {y_val}")

            # Transform Y
            new_y = y_val + y_offset
            print(f"  New Y value (with offset {y_offset}): {new_y}")

            # Create new line
            new_line = re.sub(r"Y([-+]?\d*\.?\d+)\.?", f"Y{new_y:.3f}", line)
            print(f"  New line: '{new_line}'")
        else:
            print("  Could not extract Y value!")

        print()


def simulate_shappack_transformation():
    """Simulate the specific handling in shappack.py for the problem line."""

    # The problematic line
    line = "G1 Y154.500."
    current_y = 0  # Last known position
    y_offset = 200  # Translation offset

    print("=== Simulating ShapPack Transformation Logic ===\n")
    print(f"Line: '{line}'")
    print(f"Last Y position: {current_y}")
    print(f"Y offset: {y_offset}")

    # Extract Y value with our regex
    y_match = re.search(r"Y([-+]?\d*\.?\d+)\.?", line)

    if y_match:
        y_str = y_match.group(1)
        print(f"Y match: '{y_str}'")

        # Parse Y value
        try:
            y = float(y_str)
            print(f"Parsed Y value: {y}")

            # Apply transformation
            new_y = y + y_offset
            print(f"Transformed Y value: {new_y}")

            # Create new line
            new_line = re.sub(r"Y([-+]?\d*\.?\d+)\.?", f"Y{new_y:.3f}", line)
            print(f"Transformed line: '{new_line}'")
        except ValueError:
            print(f"ERROR: Could not convert '{y_str}' to float!")
    else:
        print("ERROR: No Y coordinate found in line!")


if __name__ == "__main__":
    # Run all tests
    test_y154_regex()
    print("\n" + "=" * 50 + "\n")
    test_coordinate_transformation()
    print("\n" + "=" * 50 + "\n")
    simulate_shappack_transformation()
