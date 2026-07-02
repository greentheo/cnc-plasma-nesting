"""
Main integration for differential evolution optimized G-code nesting.
This module provides the main functions to nest G-code files using DE optimization.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Tuple
import argparse
from pathlib import Path

from shappack import GCodeParser, GCodeGenerator, Shape
from enhanced_nesting import EnhancedNestingAlgorithm
from de_optimizer import OptimizationMetrics


def nest_gcode_files_de(
    gcode_files: List[str],
    output_file: str,
    sheet_width: float,
    sheet_height: float,
    spacing: float = 5.0,
    max_copies: int = 100,
    generations: int = 100,
    population_size: int = 50,
    mutation: Tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    strategy: str = "best1bin",
    rotation_increment: float = 15.0,
    seed: Optional[int] = None,
    workers: int = 1,
    verbose: bool = True,
    show_visualization: bool = True,
    show_progress_plot: bool = True,
    compare_methods: bool = False,
) -> Tuple[List, List[OptimizationMetrics]]:
    """
    Nest multiple G-code files using differential evolution optimization.

    Args:
        gcode_files: List of G-code file paths
        output_file: Output G-code file path
        sheet_width: Sheet width in mm
        sheet_height: Sheet height in mm
        spacing: Minimum spacing between shapes in mm
        max_copies: Maximum copies of each shape
        generations: Number of DE generations
        population_size: DE population size
        mutation: DE mutation factor (F)
        recombination: DE crossover probability (CR)
        strategy: DE strategy
        rotation_increment: Rotation angle increment in degrees
        seed: Random seed for reproducibility
        workers: Number of parallel workers
        verbose: Print progress
        show_visualization: Show nesting visualization
        show_progress_plot: Show optimization progress plot
        compare_methods: Compare with grid-based method

    Returns:
        Tuple of (placements, metrics_history)
    """
    # Parse all G-code files
    parser = GCodeParser()
    all_shapes = []

    print(f"\n=== Parsing {len(gcode_files)} G-code files ===")

    for idx, gcode_file in enumerate(gcode_files):
        print(f"Parsing: {gcode_file}")

        with open(gcode_file, "r") as f:
            gcode_text = f.read()

        shapes = parser.parse_gcode(gcode_text)

        # Set original index for tracking
        for shape in shapes:
            shape.original_idx = idx

        all_shapes.extend(shapes)
        print(f"  Found {len(shapes)} shapes")

    print(f"Total shapes to nest: {len(all_shapes)}")

    # Create enhanced nesting algorithm
    nesting_algo = EnhancedNestingAlgorithm(
        sheet_width=sheet_width, sheet_height=sheet_height, spacing=spacing
    )

    # Compare methods if requested
    if compare_methods:
        comparison = nesting_algo.compare_methods(
            shapes=all_shapes,
            max_count=max_copies,
            de_generations=generations,
            de_population=population_size,
            output_file=output_file.replace(".nc", "_comparison.png"),
        )

        # Use DE results for final output
        placements = comparison["de"]["placements"]
        metrics = comparison["de"]["metrics"]
    else:
        # Run DE optimization
        placements, metrics = nesting_algo.nest_shapes_differential_evolution(
            shapes=all_shapes,
            max_count=max_copies,
            generations=generations,
            population_size=population_size,
            mutation=mutation,
            recombination=recombination,
            strategy=strategy,
            rotation_increment=rotation_increment,
            seed=seed,
            workers=workers,
            verbose=verbose,
        )

    # Generate visualization if requested
    if show_visualization:
        viz_file = output_file.replace(".nc", "_nesting.png")
        nesting_algo.visualize_nesting_with_stats(
            placements=placements,
            metrics=metrics,
            title=f"DE Optimized Nesting - {len(placements)} shapes",
            output_file=viz_file,
        )

    # Show progress plot if requested
    if show_progress_plot and metrics:
        progress_file = output_file.replace(".nc", "_progress.png")
        nesting_algo.visualize_optimization_progress(
            metrics_history=metrics, output_file=progress_file
        )

    # Generate output G-code
    gcode_generator = GCodeGenerator()
    output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
        placements, parser
    )

    # Write output file
    with open(output_file, "w") as f:
        f.write(output_gcode)

    print(f"\n✓ Output G-code written to: {output_file}")

    # Print final statistics
    if metrics:
        final_metric = metrics[-1]
        sheet_area = sheet_width * sheet_height
        total_area = sum(p.polygon.area for p in placements)
        utilization = (total_area / sheet_area) * 100

        print(f"\n=== FINAL RESULTS ===")
        print(f"Shapes placed: {len(placements)}")
        print(f"Sheet utilization: {utilization:.2f}%")
        print(f"Total optimization time: {final_metric.computation_time:.1f} seconds")
        print(f"Generations completed: {final_metric.generation}")

    return placements, metrics


def nest_single_gcode_file_de(
    input_file: str,
    output_file: str,
    sheet_width: float,
    sheet_height: float,
    spacing: float = 5.0,
    max_copies: int = 100,
    **de_params,
) -> Tuple[List, List[OptimizationMetrics]]:
    """
    Convenience function to nest a single G-code file.

    Args:
        input_file: Input G-code file path
        output_file: Output G-code file path
        sheet_width: Sheet width in mm
        sheet_height: Sheet height in mm
        spacing: Minimum spacing between shapes
        max_copies: Maximum copies of each shape
        **de_params: Additional DE optimization parameters

    Returns:
        Tuple of (placements, metrics_history)
    """
    return nest_gcode_files_de(
        gcode_files=[input_file],
        output_file=output_file,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        spacing=spacing,
        max_copies=max_copies,
        **de_params,
    )


def main():
    """Command-line interface for DE-optimized nesting."""
    parser = argparse.ArgumentParser(
        description="Nest G-code shapes using differential evolution optimization"
    )

    # Input/output arguments
    parser.add_argument("input_files", nargs="+", help="Input G-code file(s) to nest")
    parser.add_argument("-o", "--output", required=True, help="Output G-code file")

    # Sheet parameters
    parser.add_argument(
        "-w", "--width", type=float, required=True, help="Sheet width in mm"
    )
    parser.add_argument(
        "-h", "--height", type=float, required=True, help="Sheet height in mm"
    )
    parser.add_argument(
        "-s",
        "--spacing",
        type=float,
        default=5.0,
        help="Minimum spacing between shapes in mm (default: 5.0)",
    )

    # Nesting parameters
    parser.add_argument(
        "-c",
        "--max-copies",
        type=int,
        default=100,
        help="Maximum copies of each shape (default: 100)",
    )

    # DE optimization parameters
    parser.add_argument(
        "-g",
        "--generations",
        type=int,
        default=100,
        help="Number of DE generations (default: 100)",
    )
    parser.add_argument(
        "-p",
        "--population",
        type=int,
        default=50,
        help="DE population size (default: 50)",
    )
    parser.add_argument(
        "-m",
        "--mutation",
        type=float,
        nargs=2,
        default=[0.5, 1.0],
        help="DE mutation factor range (default: 0.5 1.0)",
    )
    parser.add_argument(
        "-r",
        "--recombination",
        type=float,
        default=0.7,
        help="DE crossover probability (default: 0.7)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="best1bin",
        choices=[
            "best1bin",
            "best1exp",
            "rand1bin",
            "rand1exp",
            "best2bin",
            "best2exp",
        ],
        help="DE strategy (default: best1bin)",
    )
    parser.add_argument(
        "--rotation-increment",
        type=float,
        default=15.0,
        help="Rotation angle increment in degrees (default: 15.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )

    # Visualization options
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable visualization output"
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress plot"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with grid-based method"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Run nesting
    nest_gcode_files_de(
        gcode_files=args.input_files,
        output_file=args.output,
        sheet_width=args.width,
        sheet_height=args.height,
        spacing=args.spacing,
        max_copies=args.max_copies,
        generations=args.generations,
        population_size=args.population,
        mutation=tuple(args.mutation),
        recombination=args.recombination,
        strategy=args.strategy,
        rotation_increment=args.rotation_increment,
        seed=args.seed,
        workers=args.workers,
        verbose=not args.quiet,
        show_visualization=not args.no_viz,
        show_progress_plot=not args.no_progress,
        compare_methods=args.compare,
    )


if __name__ == "__main__":
    main()
