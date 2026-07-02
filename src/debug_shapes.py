#!/usr/bin/env python3
"""
Debug script to check what shapes are being parsed from the geodesic NC files.
This will help identify why the visualization shows hexagons instead of the actual shapes.
"""

import sys
import os
from shappack import GCodeParser, Shape, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon


def debug_parsed_shapes():
    """Debug what shapes are being parsed from the NC files."""

    print("=" * 70)
    print("DEBUG: SHAPE PARSING FROM NC FILES")
    print("=" * 70)

    # Test files to check
    test_files = ["geodesic_6_bottom.nc", "geodesic_6_top.nc", "geodesic_4_top.nc"]

    parser = GCodeParser()

    # Create a figure to visualize the parsed shapes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, file_path in enumerate(test_files):
        if not os.path.exists(file_path):
            print(f"\n✗ File not found: {file_path}")
            continue

        print(f"\n{'=' * 50}")
        print(f"File: {file_path}")
        print(f"{'=' * 50}")

        # Read the raw file content
        with open(file_path, "r") as f:
            gcode_text = f.read()

        print(f"\nRaw G-code preview (first 20 lines):")
        print("-" * 30)
        lines = gcode_text.strip().split("\n")
        for i, line in enumerate(lines[:20]):
            print(f"{i + 1:3}: {line}")

        # Parse the shapes
        shapes = parser.parse_gcode(gcode_text)

        print(f"\nParsing results:")
        print(f"  Number of shapes found: {len(shapes)}")

        if shapes:
            for shape_idx, shape in enumerate(shapes):
                print(f"\n  Shape {shape_idx + 1}:")
                print(f"    Number of points: {len(shape.points)}")
                print(f"    Points:")
                for pt_idx, point in enumerate(
                    shape.points[:10]
                ):  # Show first 10 points
                    print(f"      {pt_idx + 1}: ({point.x:.2f}, {point.y:.2f})")
                if len(shape.points) > 10:
                    print(f"      ... and {len(shape.points) - 10} more points")

                # Calculate bounds
                if shape.points:
                    xs = [p.x for p in shape.points]
                    ys = [p.y for p in shape.points]
                    print(
                        f"    Bounds: X[{min(xs):.2f}, {max(xs):.2f}], Y[{min(ys):.2f}, {max(ys):.2f}]"
                    )
                    print(f"    Width: {max(xs) - min(xs):.2f}mm")
                    print(f"    Height: {max(ys) - min(ys):.2f}mm")

                # Visualize the shape
                if idx < len(axes) and shape.points:
                    ax = axes[idx]
                    coords = [(p.x, p.y) for p in shape.points]

                    # Close the polygon if needed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])

                    # Create and add polygon patch
                    patch = MPLPolygon(
                        coords,
                        alpha=0.6,
                        facecolor="lightblue",
                        edgecolor="darkblue",
                        linewidth=2,
                    )
                    ax.add_patch(patch)

                    # Plot points
                    xs, ys = zip(*coords)
                    ax.plot(xs, ys, "ro", markersize=4)

                    # Add point numbers
                    for i, (x, y) in enumerate(coords[:-1]):  # Skip the closing point
                        ax.text(x, y, str(i + 1), fontsize=8, ha="center", va="bottom")

                    ax.set_xlim(min(xs) - 10, max(xs) + 10)
                    ax.set_ylim(min(ys) - 10, max(ys) + 10)
                    ax.set_aspect("equal")
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f"{file_path}")
                    ax.set_xlabel("X (mm)")
                    ax.set_ylabel("Y (mm)")
        else:
            print("  ⚠ No shapes were parsed!")

            # Debug: Check what the parser is looking for
            print("\n  Debug: Looking for pattern markers in G-code:")
            has_g0 = any("G0" in line for line in lines)
            has_g31 = any("G31" in line for line in lines)
            has_m3 = any("M3" in line for line in lines)
            has_h1 = any("H1" in line for line in lines)
            has_g1 = any("G1" in line for line in lines)
            has_m5 = any("M5" in line for line in lines)

            print(f"    Has G0 (rapid move): {has_g0}")
            print(f"    Has G31 (height sense): {has_g31}")
            print(f"    Has M3 (torch on): {has_m3}")
            print(f"    Has H1 (pierce height): {has_h1}")
            print(f"    Has G1 (cut move): {has_g1}")
            print(f"    Has M5 (torch off): {has_m5}")

    plt.suptitle("Parsed Shapes from NC Files", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("debug_parsed_shapes.png", dpi=150)
    plt.show()

    print("\n" + "=" * 70)
    print("Debug visualization saved to: debug_parsed_shapes.png")
    print("=" * 70)


if __name__ == "__main__":
    debug_parsed_shapes()
