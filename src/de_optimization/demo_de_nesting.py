#!/usr/bin/env python3
"""
Demonstration script for differential evolution shape nesting optimization.
Shows various usage examples and parameter configurations.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from shappack import Shape, Point, GCodeParser, GCodeGenerator
from enhanced_nesting import EnhancedNestingAlgorithm
from de_nest_gcode import nest_gcode_files_de, nest_single_gcode_file_de


def create_sample_shapes() -> List[Shape]:
    """Create some sample shapes for testing."""
    shapes = []

    # Rectangle
    rect_points = [Point(0, 0), Point(30, 0), Point(30, 20), Point(0, 20)]
    shapes.append(Shape(rect_points, [], 0))

    # Triangle
    tri_points = [Point(0, 0), Point(25, 0), Point(12.5, 25)]
    shapes.append(Shape(tri_points, [], 1))

    # L-shape
    l_points = [
        Point(0, 0),
        Point(20, 0),
        Point(20, 10),
        Point(10, 10),
        Point(10, 30),
        Point(0, 30),
    ]
    shapes.append(Shape(l_points, [], 2))

    # Pentagon
    pentagon_points = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        x = 15 + 15 * np.cos(angle)
        y = 15 + 15 * np.sin(angle)
        pentagon_points.append(Point(x, y))
    shapes.append(Shape(pentagon_points, [], 3))

    return shapes


def demo_basic_de_optimization():
    """Demonstrate basic DE optimization with sample shapes."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Differential Evolution Optimization")
    print("=" * 60)

    # Create sample shapes
    shapes = create_sample_shapes()
    print(f"Created {len(shapes)} sample shapes")

    # Setup sheet dimensions
    sheet_width = 300.0
    sheet_height = 200.0
    spacing = 5.0

    # Create nesting algorithm
    nesting = EnhancedNestingAlgorithm(sheet_width, sheet_height, spacing)

    # Run DE optimization with moderate settings
    placements, metrics = nesting.nest_shapes_differential_evolution(
        shapes=shapes,
        max_count=10,  # Try to place up to 10 of each shape
        generations=50,
        population_size=30,
        mutation=(0.5, 1.0),
        recombination=0.7,
        rotation_increment=15.0,  # 15-degree increments
        verbose=True,
    )

    # Show results
    print(f"\nOptimization complete!")
    print(f"Total shapes placed: {len(placements)}")

    if metrics:
        final_metric = metrics[-1]
        print(f"Final utilization: {final_metric.best_utilization:.2%}")
        print(f"Time taken: {final_metric.computation_time:.1f} seconds")

    # Visualize results
    nesting.visualize_nesting_with_stats(
        placements,
        metrics,
        title="Basic DE Optimization Demo",
        output_file="demo1_basic_de.png",
    )

    # Show optimization progress
    if metrics:
        nesting.visualize_optimization_progress(
            metrics, output_file="demo1_progress.png"
        )

    return placements, metrics


def demo_parameter_sensitivity():
    """Demonstrate how different DE parameters affect results."""
    print("\n" + "=" * 60)
    print("DEMO 2: Parameter Sensitivity Analysis")
    print("=" * 60)

    shapes = create_sample_shapes()
    sheet_width = 300.0
    sheet_height = 200.0

    # Test different configurations
    configs = [
        {
            "name": "Conservative",
            "generations": 30,
            "population_size": 20,
            "mutation": (0.3, 0.7),
            "recombination": 0.5,
        },
        {
            "name": "Aggressive",
            "generations": 30,
            "population_size": 50,
            "mutation": (0.8, 1.5),
            "recombination": 0.9,
        },
        {
            "name": "Balanced",
            "generations": 30,
            "population_size": 35,
            "mutation": (0.5, 1.0),
            "recombination": 0.7,
        },
    ]

    results = []

    for config in configs:
        print(f"\nTesting {config['name']} configuration...")
        nesting = EnhancedNestingAlgorithm(sheet_width, sheet_height, 5.0)

        placements, metrics = nesting.nest_shapes_differential_evolution(
            shapes=shapes,
            max_count=10,
            generations=config["generations"],
            population_size=config["population_size"],
            mutation=config["mutation"],
            recombination=config["recombination"],
            rotation_increment=15.0,
            verbose=False,
        )

        utilization = metrics[-1].best_utilization if metrics else 0
        time_taken = metrics[-1].computation_time if metrics else 0

        results.append(
            {
                "name": config["name"],
                "placements": len(placements),
                "utilization": utilization,
                "time": time_taken,
            }
        )

        print(
            f"  Results: {len(placements)} shapes, {utilization:.2%} utilization, {time_taken:.1f}s"
        )

    # Compare results
    print("\n" + "=" * 40)
    print("PARAMETER SENSITIVITY RESULTS")
    print("=" * 40)
    print(f"{'Config':<15} {'Shapes':<10} {'Utilization':<15} {'Time (s)':<10}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['name']:<15} {r['placements']:<10} {r['utilization']:<15.2%} {r['time']:<10.1f}"
        )

    return results


def demo_rotation_increment_comparison():
    """Compare different rotation increment settings."""
    print("\n" + "=" * 60)
    print("DEMO 3: Rotation Increment Comparison")
    print("=" * 60)

    shapes = create_sample_shapes()
    sheet_width = 250.0
    sheet_height = 150.0

    rotation_increments = [90.0, 45.0, 15.0, 5.0]
    results = []

    for rot_inc in rotation_increments:
        print(f"\nTesting {rot_inc}° rotation increment...")
        nesting = EnhancedNestingAlgorithm(sheet_width, sheet_height, 5.0)

        placements, metrics = nesting.nest_shapes_differential_evolution(
            shapes=shapes,
            max_count=8,
            generations=40,
            population_size=30,
            rotation_increment=rot_inc,
            verbose=False,
        )

        utilization = metrics[-1].best_utilization if metrics else 0
        shapes_placed = len(placements)

        results.append(
            {
                "increment": rot_inc,
                "shapes": shapes_placed,
                "utilization": utilization,
                "angles_available": int(360 / rot_inc),
            }
        )

        print(f"  Results: {shapes_placed} shapes, {utilization:.2%} utilization")
        print(f"  Available angles: {list(range(0, 360, int(rot_inc)))}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    increments = [r["increment"] for r in results]
    utilizations = [r["utilization"] * 100 for r in results]
    shapes_counts = [r["shapes"] for r in results]

    ax1.bar(range(len(increments)), utilizations, color="blue", alpha=0.7)
    ax1.set_xticks(range(len(increments)))
    ax1.set_xticklabels([f"{inc}°" for inc in increments])
    ax1.set_xlabel("Rotation Increment")
    ax1.set_ylabel("Utilization (%)")
    ax1.set_title("Sheet Utilization vs Rotation Increment")
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(increments)), shapes_counts, color="green", alpha=0.7)
    ax2.set_xticks(range(len(increments)))
    ax2.set_xticklabels([f"{inc}°" for inc in increments])
    ax2.set_xlabel("Rotation Increment")
    ax2.set_ylabel("Shapes Placed")
    ax2.set_title("Shapes Placed vs Rotation Increment")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Effect of Rotation Increment on Nesting Performance")
    plt.tight_layout()
    plt.savefig("demo3_rotation_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results


def demo_method_comparison():
    """Compare DE optimization with traditional grid-based approach."""
    print("\n" + "=" * 60)
    print("DEMO 4: DE vs Grid-Based Method Comparison")
    print("=" * 60)

    shapes = create_sample_shapes()
    sheet_width = 300.0
    sheet_height = 200.0

    nesting = EnhancedNestingAlgorithm(sheet_width, sheet_height, 5.0)

    # Run comparison
    comparison_results = nesting.compare_methods(
        shapes=shapes,
        max_count=10,
        de_generations=50,
        de_population=30,
        output_file="demo4_comparison.png",
    )

    return comparison_results


def demo_gcode_integration():
    """Demonstrate G-code file processing with DE optimization."""
    print("\n" + "=" * 60)
    print("DEMO 5: G-code Integration (Simulation)")
    print("=" * 60)

    # Create a simple G-code sample with proper formatting
    sample_gcode = """G21 (Units: mm)
G0 X10 Y10
G31 (height sensing)
M3 (torch on)
G1 X40 Y10 F1000
G1 X40 Y30
G1 X10 Y30
G1 X10 Y10
M5 (torch off)

G0 X60 Y10
G31 (height sensing)
M3 (torch on)
G1 X80 Y10 F1000
G1 X70 Y25
G1 X60 Y10
M5 (torch off)
"""

    # Save to temporary file
    temp_file = "temp_demo.nc"
    with open(temp_file, "w") as f:
        f.write(sample_gcode)

    print("Created sample G-code file")

    # Process with DE optimization
    try:
        placements, metrics = nest_single_gcode_file_de(
            input_file=temp_file,
            output_file="demo5_output.nc",
            sheet_width=300,
            sheet_height=200,
            spacing=5.0,
            max_copies=5,
            generations=30,
            population_size=25,
            rotation_increment=15.0,
            verbose=True,
            show_visualization=True,
            show_progress_plot=True,
        )

        print(f"\nG-code processing complete!")
        print(f"Output saved to: demo5_output.nc")

    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return placements, metrics


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" DIFFERENTIAL EVOLUTION NESTING OPTIMIZATION DEMONSTRATIONS")
    print("=" * 70)
    print("\nThis script demonstrates various features of the DE optimization")
    print("system for CNC shape nesting with 15-degree rotation increments.\n")

    # Create output directory for demo files
    output_dir = Path("de_demo_output")
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)

    try:
        # Run demonstrations
        demos = [
            ("Basic DE Optimization", demo_basic_de_optimization),
            ("Parameter Sensitivity", demo_parameter_sensitivity),
            ("Rotation Increment Comparison", demo_rotation_increment_comparison),
            ("Method Comparison", demo_method_comparison),
            ("G-code Integration", demo_gcode_integration),
        ]

        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n{'=' * 70}")
            print(f"Running Demo {i}/{len(demos)}: {name}")
            print("=" * 70)

            try:
                demo_func()
                print(f"\n✓ {name} completed successfully")
            except Exception as e:
                print(f"\n✗ {name} failed: {str(e)}")
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 70)
        print(" ALL DEMONSTRATIONS COMPLETE")
        print("=" * 70)
        print(f"\nOutput files saved to: {output_dir.absolute()}")
        print("\nKey features demonstrated:")
        print("• DE optimization with configurable parameters")
        print("• 15-degree rotation increments for precise placement")
        print("• Real-time progress tracking and metrics")
        print("• Comparison with grid-based methods")
        print("• G-code integration and processing")
        print("• Comprehensive visualization and statistics")

    finally:
        os.chdir("..")


if __name__ == "__main__":
    main()
