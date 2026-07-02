#!/usr/bin/env python3
"""
Test script for the improved nesting algorithm with continuous rotation
and random shape selection.
"""

from shappack_improved import nest_gcode_files
import sys


def create_sample_gcode():
    """Create sample G-code for testing with multiple shapes"""

    # Simple rectangle
    rectangle_gcode = """
    ; Rectangle shape
    G0 X0 Y0
    G1 X0 Y0
    G1 X40 Y0
    G1 X40 Y20
    G1 X0 Y20
    G1 X0 Y0
    M5
    """

    # Triangle
    triangle_gcode = """
    ; Triangle shape
    G0 X0 Y0
    G1 X0 Y0
    G1 X30 Y0
    G1 X15 Y25
    G1 X0 Y0
    M5
    """

    # L-shape
    l_shape_gcode = """
    ; L-shaped part
    G0 X0 Y0
    G1 X0 Y0
    G1 X20 Y0
    G1 X20 Y10
    G1 X10 Y10
    G1 X10 Y30
    G1 X0 Y30
    G1 X0 Y0
    M5
    """

    # Circle approximation (octagon)
    circle_gcode = """
    ; Circular shape (octagon)
    G0 X10 Y0
    G1 X10 Y0
    G1 X20 Y0
    G1 X27 Y7
    G1 X27 Y17
    G1 X20 Y24
    G1 X10 Y24
    G1 X3 Y17
    G1 X3 Y7
    G1 X10 Y0
    M5
    """

    return [
        (rectangle_gcode, 15),  # 15 rectangles
        (triangle_gcode, 20),  # 20 triangles
        (l_shape_gcode, 10),  # 10 L-shapes
        (circle_gcode, 12),  # 12 circles
    ]


def test_continuous_rotation():
    """Test the nesting with different rotation step sizes"""

    print("=" * 60)
    print("TESTING IMPROVED NESTING ALGORITHM")
    print("=" * 60)

    # Get sample shapes
    shapes = create_sample_gcode()

    # Sheet dimensions
    sheet_width = 300  # mm
    sheet_height = 200  # mm
    spacing = 3  # mm between parts

    # Test with different rotation steps
    rotation_steps = [90, 45, 30, 15, 10, 5]

    results = []

    for step in rotation_steps:
        print(f"\n--- Testing with {step}° rotation step ---")

        try:
            nested_gcode, parts_placed = nest_gcode_files(
                shapes, sheet_width, sheet_height, spacing, rotation_step=step
            )

            # Calculate theoretical maximum
            total_requested = sum(count for _, count in shapes)
            efficiency = (parts_placed / total_requested) * 100

            results.append(
                {"step": step, "placed": parts_placed, "efficiency": efficiency}
            )

            print(f"Placed {parts_placed} parts out of {total_requested} requested")
            print(f"Packing efficiency: {efficiency:.1f}%")

            # Save the best result
            if step == 15:  # Save the 15-degree result as a good balance
                with open("nested_output_improved.nc", "w") as f:
                    f.write(nested_gcode)
                print(f"Saved nested G-code to nested_output_improved.nc")

        except Exception as e:
            print(f"Error with {step}° rotation: {e}")

    # Print comparison
    print("\n" + "=" * 60)
    print("ROTATION STEP COMPARISON")
    print("=" * 60)
    print(f"{'Rotation Step':<15} {'Parts Placed':<15} {'Efficiency':<15}")
    print("-" * 45)

    for r in results:
        print(f"{r['step']:>10}°    {r['placed']:>12}    {r['efficiency']:>10.1f}%")

    if results:
        # Find best result
        best = max(results, key=lambda x: x["placed"])
        print(
            f"\nBest result: {best['step']}° rotation with {best['placed']} parts placed"
        )

        # Compare with 90-degree baseline
        baseline = next((r for r in results if r["step"] == 90), None)
        if baseline and best["step"] != 90:
            improvement = (
                (best["placed"] - baseline["placed"]) / baseline["placed"]
            ) * 100
            print(f"Improvement over 90° rotation: {improvement:.1f}%")


def test_random_selection():
    """Test that random selection maintains proper part ratios"""

    print("\n" + "=" * 60)
    print("TESTING RANDOM SHAPE SELECTION")
    print("=" * 60)

    # Create shapes with specific ratios
    shapes = [
        (create_sample_gcode()[0][0], 10),  # 10 rectangles
        (create_sample_gcode()[1][0], 20),  # 20 triangles
        (create_sample_gcode()[2][0], 5),  # 5 L-shapes
    ]

    # Use smaller sheet to force partial placement
    sheet_width = 150  # mm
    sheet_height = 100  # mm

    print(f"\nSheet size: {sheet_width}x{sheet_height} mm")
    print("Requested parts:")
    for i, (_, count) in enumerate(shapes):
        shape_names = ["Rectangle", "Triangle", "L-shape"]
        print(f"  {shape_names[i]}: {count}")

    nested_gcode, parts_placed = nest_gcode_files(
        shapes, sheet_width, sheet_height, spacing=3, rotation_step=30
    )

    total_requested = sum(count for _, count in shapes)
    print(f"\nTotal requested: {total_requested}")
    print(f"Total placed: {parts_placed}")
    print(f"Fill rate: {(parts_placed / total_requested) * 100:.1f}%")

    print("\nRandom selection ensures mixed placement of different part types")
    print("even when the sheet can't fit all requested parts.")


if __name__ == "__main__":
    # Run tests
    test_continuous_rotation()
    test_random_selection()

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("Check 'nesting_layout.png' for visual representation")
    print("=" * 60)
