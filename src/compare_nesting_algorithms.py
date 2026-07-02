#!/usr/bin/env python3
"""
This script compares the simple greedy nesting algorithm with the hybrid optimization approach.
It runs both algorithms on the same set of shapes and compares their performance metrics.
Output includes visualization and G-code files for both approaches.
"""

import os
import time
import random
import numpy as np
from shapely.affinity import translate, rotate
import argparse

# Import the nesting algorithms
from shappack import GCodeParser, GCodeGenerator, NestingAlgorithm
from shappack_hybrid import HybridNestingAlgorithm
from shappack_multi_objective import MultiObjectiveHybridNestingAlgorithm


def compare_nesting_algorithms(
    # Sheet parameters (in inches, converted to mm internally)
    sheet_width_inches=24,
    sheet_height_inches=48,
    spacing_mm=3,
    # Input files with copy counts
    shape_files=None,
    # Algorithm selection
    run_greedy=True,
    run_hybrid=True,
    # G-code generator selection
    use_simplified_gcode=True,
    # Hybrid algorithm parameters
    hybrid_params=None,
    # Weights for multi-objective optimization
    optimization_weights=None,
    # Use multi-objective optimization
    use_multi_objective=True,
    # Output file names
    output_prefix="",
    # Random seed for reproducibility
    random_seed=42,
    # Multi-panel functionality
    max_panels=10,
    # Randomize shape order for greedy algorithm
    randomize_shape_order=False,
):
    """Run a comparison test between greedy and hybrid nesting algorithms.

    Parameters
    ----------
    sheet_width_inches : float
        Width of the sheet in inches
    sheet_height_inches : float
        Height of the sheet in inches
    spacing_mm : float
        Spacing between parts in mm
    shape_files : list of tuples
        List of (filename, count) pairs specifying the G-code files and how many copies to nest
    hybrid_params : dict
        Dictionary of parameters for the hybrid algorithm
    optimization_weights : dict
        Weights for multi-objective optimization (parts_placed_weight, compactness_weight, utilization_weight)
    use_multi_objective : bool
        Whether to use the multi-objective algorithm or the standard hybrid algorithm
    randomize_shape_order : bool
        When True, randomizes the order of shapes before the greedy algorithm processes them
    output_prefix : str
        Prefix for output filenames (useful for multiple runs)
    random_seed : int
        Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Convert inches to mm for sheet dimensions
    SHEET_WIDTH = sheet_width_inches * 25.4  # mm
    SHEET_HEIGHT = sheet_height_inches * 25.4  # mm
    SPACING = spacing_mm  # mm between parts

    # Output file names will be created for each panel
    # These are just base names that will have panel numbers appended
    greedy_viz_base = f"{output_prefix}greedy_result"
    hybrid_viz_base = f"{output_prefix}hybrid_result"
    progress_file_base = f"{output_prefix}hybrid_progress"
    greedy_output_base = f"{output_prefix}greedy_output"
    hybrid_output_base = f"{output_prefix}hybrid_output"

    print("=" * 70)
    print(f"NESTING ALGORITHM COMPARISON")
    print("=" * 70)
    print(
        f"\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm ({sheet_width_inches} x {sheet_height_inches} inches)"
    )
    print(f"Spacing: {SPACING} mm")

    # Set default optimization weights if none provided
    if optimization_weights is None:
        optimization_weights = {
            "parts_placed_weight": 0.6,  # Weight for parts placed objective
            "compactness_weight": 0.2,  # Weight for compactness objective
            "utilization_weight": 0.2,  # Weight for utilization objective
        }

    # Make sure weights add up to 1.0
    total_weight = (
        optimization_weights["parts_placed_weight"]
        + optimization_weights["compactness_weight"]
        + optimization_weights["utilization_weight"]
    )
    if abs(total_weight - 1.0) > 1e-6:
        print(
            f"Warning: Optimization weights sum to {total_weight:.4f}, normalizing to 1.0"
        )
        for key in optimization_weights:
            optimization_weights[key] /= total_weight

    # Set default shape files if none provided
    if shape_files is None:
        shape_files = [
            ("geodesic_6_top.nc", 5),
            ("geodesic_6_bottom.nc", 5),
            ("geodesic_5_top.nc", 2),
            ("geodesic_5_bottom.nc", 2),
            ("geodesic_4_top.nc", 2),
            ("geodesic_4_bottom.nc", 4),
        ]

    # Parse files
    parser = GCodeParser()
    all_shapes = []
    original_gcodes = []

    print("Parsing G-code files...")
    for idx, (file_path, max_count) in enumerate(shape_files):
        if not os.path.exists(file_path):
            print(f"X Missing: {file_path}")
            continue

        print(f"✓ Found: {file_path}")
        with open(file_path, "r") as f:
            gcode_text = f.read()
            original_gcodes.append(gcode_text)

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

    # Initialize results dictionary
    results = {}

    # Initialize algorithm instances
    greedy_algo = None

    # ======= GREEDY ALGORITHM =======
    if run_greedy:
        print("\n" + "=" * 70)
        print("TESTING GREEDY ALGORITHM")
        print("=" * 70)

        # Initialize variables for multi-panel nesting
        greedy_start_time = time.time()
        remaining_shapes = all_shapes.copy()
        greedy_panels_placements = []
        greedy_total_parts_placed = 0
        greedy_total_area = 0
        panel_count = 0

        while remaining_shapes and panel_count < max_panels:
            panel_count += 1
            print(f"\n--- Panel {panel_count} ---")
            greedy_viz_file = f"{greedy_viz_base}_panel{panel_count}.png"
            greedy_output_file = f"{greedy_output_base}_panel{panel_count}.nc"

            greedy_algo = NestingAlgorithm(
                sheet_width=SHEET_WIDTH,
                sheet_height=SHEET_HEIGHT,
                spacing=SPACING,
            )

            # Randomize shape order if flag is set
            shapes_to_nest = remaining_shapes.copy()
            if randomize_shape_order:
                random.shuffle(shapes_to_nest)
                print(
                    f"Shape placement order randomized for greedy algorithm (panel {panel_count})"
                )

            # Nest shapes for this panel
            panel_placements = greedy_algo.nest_shapes_optimal(
                shapes=shapes_to_nest,
                max_count=len(shapes_to_nest),
            )

            if not panel_placements:
                print(f"No more shapes could be placed on panel {panel_count}")
                break

            greedy_panels_placements.append(
                {
                    "panel": panel_count,
                    "placements": panel_placements,
                    "viz_file": greedy_viz_file,
                    "output_file": greedy_output_file,
                }
            )

            # Remove placed shapes from remaining shapes
            placed_indices = []
            for i, shape in enumerate(remaining_shapes):
                for placement in panel_placements:
                    if shape is placement.shape:
                        placed_indices.append(i)
                        break

            # Remove in reverse order to keep indices valid
            for index in sorted(placed_indices, reverse=True):
                remaining_shapes.pop(index)

            # Update total counts
            greedy_total_parts_placed += len(panel_placements)
            greedy_total_area += sum(p.polygon.area for p in panel_placements)

            # Calculate and show panel metrics
            panel_sheet_area = SHEET_WIDTH * SHEET_HEIGHT
            panel_area = sum(p.polygon.area for p in panel_placements)
            panel_utilization = (panel_area / panel_sheet_area) * 100

            # Calculate panel compactness
            if panel_placements:
                min_x = min(
                    min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                    for p in panel_placements
                )
                min_y = min(
                    min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                    for p in panel_placements
                )
                max_x = max(
                    max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                    for p in panel_placements
                )
                max_y = max(
                    max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                    for p in panel_placements
                )
                bounding_box_area = (max_x - min_x) * (max_y - min_y)
                panel_compactness = (
                    (panel_area / bounding_box_area) * 100
                    if bounding_box_area > 0
                    else 0
                )
            else:
                panel_compactness = 0

            print(f"Panel {panel_count} Results:")
            print(f"Shapes placed: {len(panel_placements)}")
            print(f"Utilization: {panel_utilization:.2f}%")
            print(f"Compactness: {panel_compactness:.2f}%")

            # Save visualization for this panel
            greedy_algo.visualize_nesting(panel_placements, filename=greedy_viz_file)
            print(f"✓ Visualization saved to: {greedy_viz_file}")

        greedy_elapsed = time.time() - greedy_start_time

        # Calculate overall metrics
        greedy_sheet_area = SHEET_WIDTH * SHEET_HEIGHT * panel_count
        greedy_utilization = (
            (greedy_total_area / greedy_sheet_area) * 100 if panel_count > 0 else 0
        )
        greedy_compactness = (
            0  # Overall compactness doesn't make sense across multiple panels
        )

        print("\nGreedy Algorithm Overall Results:")
        print(f"Total panels used: {panel_count}")
        print(f"Total shapes placed: {greedy_total_parts_placed} / {len(all_shapes)}")
        print(f"Average utilization: {greedy_utilization:.2f}%")
        print(f"Total time: {greedy_elapsed:.1f} seconds")

        results["greedy"] = {
            "panels": greedy_panels_placements,
            "panel_count": panel_count,
            "parts_placed": greedy_total_parts_placed,
            "utilization": greedy_utilization,
            "compactness": greedy_compactness,  # Not meaningful for multiple panels
            "time": greedy_elapsed,
        }
    else:
        # Initialize variables with default values
        results["greedy"] = {
            "panels": [],
            "panel_count": 0,
            "parts_placed": 0,
            "utilization": 0.0,
            "compactness": 0.0,
            "time": 0.0,
        }

    # ======= HYBRID ALGORITHM =======
    if run_hybrid:
        print("\n" + "=" * 70)
        print("TESTING HYBRID ALGORITHM")
        print("=" * 70)

        # Set default hybrid parameters if none provided
        if hybrid_params is None:
            hybrid_params = {
                "generations": 10,
                "population_size": 30,
                "rotation_increment": 30.0,  # 30-degree increments
                "max_rotation": 90,  # Maximum rotation in degrees
                "compactness_weight": 0.3,
                "grid_size": 50.0,  # Larger grid for faster search
            }

        print(f"Generations: {hybrid_params['generations']}")
        print(f"Population: {hybrid_params['population_size']}")
        max_rotation = hybrid_params.get("max_rotation", 90)
        print(
            f"Rotation increment: {hybrid_params['rotation_increment']} degrees (0-{max_rotation} degrees only)"
        )
        print(f"Optimization weights (sum={sum(optimization_weights.values()):.2f}):")
        print(f"  - Parts placed: {optimization_weights['parts_placed_weight']:.2f}")
        print(f"  - Compactness: {optimization_weights['compactness_weight']:.2f}")
        print(f"  - Utilization: {optimization_weights['utilization_weight']:.2f}")
        print(f"Grid size: {hybrid_params['grid_size']} mm")
        print(f"Workers: 1 (single worker mode)")

        # Initialize variables for multi-panel nesting
        hybrid_start_time = time.time()
        remaining_shapes = all_shapes.copy()
        hybrid_panels_placements = []
        hybrid_total_parts_placed = 0
        hybrid_total_area = 0
        panel_count = 0
        all_hybrid_metrics = []

        while remaining_shapes and panel_count < max_panels:
            panel_count += 1
            print(f"\n--- Panel {panel_count} ---")
            hybrid_viz_file = f"{hybrid_viz_base}_panel{panel_count}.png"
            hybrid_output_file = f"{hybrid_output_base}_panel{panel_count}.nc"
            progress_file = f"{progress_file_base}_panel{panel_count}.png"

            # Choose between standard hybrid and multi-objective algorithms
            if use_multi_objective:
                print("Using multi-objective optimization algorithm")
                hybrid_algo = MultiObjectiveHybridNestingAlgorithm(
                    sheet_width=SHEET_WIDTH,
                    sheet_height=SHEET_HEIGHT,
                    spacing=SPACING,
                    grid_size=float(hybrid_params["grid_size"]),
                    parts_placed_weight=float(
                        optimization_weights["parts_placed_weight"]
                    ),
                    compactness_weight=float(
                        optimization_weights["compactness_weight"]
                    ),
                    utilization_weight=float(
                        optimization_weights["utilization_weight"]
                    ),
                    verbose=True,
                )
            else:
                print("Using standard hybrid optimization algorithm")
                hybrid_algo = HybridNestingAlgorithm(
                    sheet_width=SHEET_WIDTH,
                    sheet_height=SHEET_HEIGHT,
                    spacing=SPACING,
                    grid_size=float(hybrid_params["grid_size"]),
                    compactness_weight=float(
                        optimization_weights["compactness_weight"]
                    ),
                )

            # Nest shapes for this panel
            panel_placements, panel_metrics = hybrid_algo.nest_shapes_hybrid(
                shapes=remaining_shapes,
                generations=int(hybrid_params["generations"]),
                population_size=int(hybrid_params["population_size"]),
                rotation_increment=float(hybrid_params["rotation_increment"]),
                verbose=True,
                max_rotation=float(hybrid_params.get("max_rotation", 90)),
                workers=1,  # Single worker mode
                seed=int(random_seed) + panel_count,  # Different seed for each panel
            )

            all_hybrid_metrics.append(panel_metrics)

            if not panel_placements:
                print(f"No more shapes could be placed on panel {panel_count}")
                break

            hybrid_panels_placements.append(
                {
                    "panel": panel_count,
                    "placements": panel_placements,
                    "metrics": panel_metrics,
                    "viz_file": hybrid_viz_file,
                    "output_file": hybrid_output_file,
                    "progress_file": progress_file,
                }
            )

            # Remove placed shapes from remaining shapes
            placed_indices = []
            for i, shape in enumerate(remaining_shapes):
                for placement in panel_placements:
                    if shape is placement.shape:
                        placed_indices.append(i)
                        break

            # Remove in reverse order to keep indices valid
            for index in sorted(placed_indices, reverse=True):
                remaining_shapes.pop(index)

            # Update total counts
            hybrid_total_parts_placed += len(panel_placements)
            hybrid_total_area += sum(p.polygon.area for p in panel_placements)

            # Calculate and show panel metrics
            panel_sheet_area = SHEET_WIDTH * SHEET_HEIGHT
            panel_area = sum(p.polygon.area for p in panel_placements)
            panel_utilization = (panel_area / panel_sheet_area) * 100

            # Calculate panel compactness
            if panel_placements:
                min_x = min(
                    min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                    for p in panel_placements
                )
                min_y = min(
                    min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                    for p in panel_placements
                )
                max_x = max(
                    max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                    for p in panel_placements
                )
                max_y = max(
                    max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                    for p in panel_placements
                )
                bounding_box_area = (max_x - min_x) * (max_y - min_y)
                panel_compactness = (
                    (panel_area / bounding_box_area) * 100
                    if bounding_box_area > 0
                    else 0
                )
            else:
                panel_compactness = 0

            print(f"Panel {panel_count} Results:")
            print(f"Shapes placed: {len(panel_placements)}")
            print(f"Utilization: {panel_utilization:.2f}%")
            print(f"Compactness: {panel_compactness:.2f}%")

            # Save visualization for this panel
            hybrid_algo.visualize_nesting(panel_placements, filename=hybrid_viz_file)
            print(f"✓ Visualization saved to: {hybrid_viz_file}")

            # Save progress visualization for this panel
            if panel_metrics:  # Only visualize if we have metrics
                hybrid_algo.visualize_progress(panel_metrics, filename=progress_file)
                print(f"✓ Progress visualization saved to: {progress_file}")

        hybrid_elapsed = time.time() - hybrid_start_time

        # Calculate overall metrics
        hybrid_sheet_area = SHEET_WIDTH * SHEET_HEIGHT * panel_count
        hybrid_utilization = (
            (hybrid_total_area / hybrid_sheet_area) * 100 if panel_count > 0 else 0
        )
        hybrid_compactness = (
            0  # Overall compactness doesn't make sense across multiple panels
        )

        print("\nHybrid Algorithm Overall Results:")
        print(f"Total panels used: {panel_count}")
        print(f"Total shapes placed: {hybrid_total_parts_placed} / {len(all_shapes)}")
        print(f"Average utilization: {hybrid_utilization:.2f}%")
        print(f"Total time: {hybrid_elapsed:.1f} seconds")

        results["hybrid"] = {
            "panels": hybrid_panels_placements,
            "panel_count": panel_count,
            "parts_placed": hybrid_total_parts_placed,
            "utilization": hybrid_utilization,
            "compactness": hybrid_compactness,  # Not meaningful for multiple panels
            "time": hybrid_elapsed,
        }
    else:
        # Set empty results if hybrid algorithm is not run
        results["hybrid"] = {
            "panels": [],
            "panel_count": 0,
            "parts_placed": 0,
            "utilization": 0.0,
            "compactness": 0.0,
            "time": 0.0,
        }

    # ======= PERFORMANCE COMPARISON =======
    if run_greedy or run_hybrid:
        print("\n" + "=" * 70)
        print("ALGORITHM PERFORMANCE COMPARISON")
        print("=" * 70)

        print(
            "\nAlgorithm | Panels | Parts Placed | Utilization | Compactness | Time (s)"
        )
        print("-" * 80)

        # Print results for greedy algorithm if it was run
        if run_greedy:
            r = results["greedy"]
            panel_count = r.get("panel_count", 0)
            parts_placed = r.get("parts_placed", 0)
            utilization = r.get("utilization", 0.0)
            compactness = r.get("compactness", 0.0)
            time_taken = r.get("time", 0.0)

            print(
                f"{'greedy':8} | {panel_count:6} | {parts_placed:12} | "
                f"{utilization:10.2f}% | {compactness:10.2f}% | {time_taken:7.1f}"
            )

        # Print results for hybrid algorithm if it was run
        if run_hybrid:
            r = results["hybrid"]
            panel_count = r.get("panel_count", 0)
            parts_placed = r.get("parts_placed", 0)
            utilization = r.get("utilization", 0.0)
            compactness = r.get("compactness", 0.0)
            time_taken = r.get("time", 0.0)

            print(
                f"{'hybrid':8} | {panel_count:6} | {parts_placed:12} | "
                f"{utilization:10.2f}% | {compactness:10.2f}% | {time_taken:7.1f}"
            )

        # Generate G-code for both algorithms
        gcode_generator = GCodeGenerator()

        # Generate G-code for the greedy algorithm panels
        if run_greedy and results["greedy"].get("panels"):
            panels = results["greedy"]["panels"]
            if panels and isinstance(panels, list) and len(panels) > 0:
                print("\nGenerating G-code for greedy algorithm panels...")
                for panel_data in panels:
                    if not isinstance(panel_data, dict):
                        continue

                    panel_num = panel_data.get("panel", 0)
                    panel_placements = panel_data.get("placements", [])
                    output_file = panel_data.get("output_file", "")

                    try:
                        print(f"Generating G-code for panel {panel_num}...")
                        if use_simplified_gcode:
                            panel_gcode = gcode_generator.generate_simplified_gcode(
                                original_gcodes, panel_placements, parser
                            )
                        else:
                            panel_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                                original_gcodes, panel_placements, parser
                            )

                        with open(output_file, "w") as f:
                            f.write(panel_gcode)

                        print(f"✓ G-code saved to: {output_file}")
                    except Exception as e:
                        print(
                            f"! G-code generation for panel {panel_num} encountered an issue: {str(e)}"
                        )

        # Generate G-code for the hybrid algorithm panels
        if run_hybrid and results["hybrid"].get("panels"):
            panels = results["hybrid"]["panels"]
            if panels and isinstance(panels, list) and len(panels) > 0:
                print("\nGenerating G-code for hybrid algorithm panels...")
                for panel_data in panels:
                    if not isinstance(panel_data, dict):
                        continue

                    panel_num = panel_data.get("panel", 0)
                    panel_placements = panel_data.get("placements", [])
                    output_file = panel_data.get("output_file", "")

                    try:
                        print(f"Generating G-code for panel {panel_num}...")
                        if use_simplified_gcode:
                            panel_gcode = gcode_generator.generate_simplified_gcode(
                                original_gcodes, panel_placements, parser
                            )
                        else:
                            panel_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                                original_gcodes, panel_placements, parser
                            )

                        with open(output_file, "w") as f:
                            f.write(panel_gcode)

                        print(f"✓ G-code saved to: {output_file}")
                    except Exception as e:
                        print(
                            f"! G-code generation for panel {panel_num} encountered an issue: {str(e)}"
                        )

    # Determine which algorithm performed better if both were run
    if run_greedy and run_hybrid:
        # Get the algorithm results
        hybrid_parts = int(results["hybrid"].get("parts_placed", 0))
        greedy_parts = int(results["greedy"].get("parts_placed", 0))

        # Get panel counts (fewer is better)
        hybrid_panels = int(results["hybrid"].get("panel_count", 0))
        greedy_panels = int(results["greedy"].get("panel_count", 0))

        if hybrid_parts > greedy_parts:
            print("\n✓ Hybrid algorithm placed more parts!")
        elif hybrid_parts < greedy_parts:
            print("\n✓ Greedy algorithm placed more parts!")
        else:
            # Same number of parts, compare by number of panels (fewer is better)
            if hybrid_panels < greedy_panels:
                print("\n✓ Hybrid algorithm used fewer panels!")
            elif hybrid_panels > greedy_panels:
                print("\n✓ Greedy algorithm used fewer panels!")
            else:
                # Same number of panels, compare utilization
                hybrid_util = float(results["hybrid"].get("utilization", 0.0))
                greedy_util = float(results["greedy"].get("utilization", 0.0))

                if hybrid_util > greedy_util:
                    print("\n✓ Overall: Hybrid algorithm achieved better utilization!")
                elif hybrid_util < greedy_util:
                    print("\n✓ Overall: Greedy algorithm achieved better utilization!")
                else:
                    print("\n✓ Both algorithms performed equally!")

    # Show appropriate completion message and summary about remaining shapes
    if run_greedy and run_hybrid:
        print("\nComparison complete!")
    elif run_greedy:
        print("\nGreedy algorithm run complete!")
    elif run_hybrid:
        print("\nHybrid algorithm run complete!")
    else:
        print("\nNo algorithms were run.")

    # Check if all shapes were placed by looking at parts placed count
    total_placed = 0
    if run_greedy and run_hybrid:
        # Use the better algorithm's result
        total_placed = max(
            int(results["greedy"].get("parts_placed", 0)),
            int(results["hybrid"].get("parts_placed", 0)),
        )
    elif run_greedy:
        total_placed = int(results["greedy"].get("parts_placed", 0))
    elif run_hybrid:
        total_placed = int(results["hybrid"].get("parts_placed", 0))

    shapes_count = len(all_shapes)
    if total_placed < shapes_count:
        print(
            f"\nWarning: {shapes_count - total_placed} shapes could not be placed within {max_panels} panels!"
        )
    else:
        print(f"\nSuccess: All {shapes_count} shapes were successfully placed!")

    # Return results for potential further analysis
    return results


if __name__ == "__main__":
    # Example of how to call the function with custom parameters
    compare_nesting_algorithms(
        # Sheet dimensions (in inches)
        sheet_width_inches=24,
        sheet_height_inches=30,
        spacing_mm=3,
        # Input files with copy counts
        shape_files=[
            # Main parts (more copies)
            ("geodesic_6_top.nc", 0),  # 15
            ("geodesic_6_bottom.nc", 7),  # 5
            # Medium parts
            ("geodesic_5_top.nc", 0),
            ("geodesic_5_bottom.nc", 5),  # 5
            # Small parts
            ("geodesic_4_top.nc", 0),
            ("geodesic_4_bottom.nc", 2),  # 5
        ],
        # Algorithm selection
        run_greedy=True,  # Set to False to skip the greedy algorithm
        run_hybrid=False,  # Set to False to skip the hybrid algorithm
        # Randomize shape placement order for greedy algorithm
        randomize_shape_order=False,  # Set to True to randomize shape order before placement
        # Use multi-objective algorithm (vs standard hybrid algorithm)
        use_multi_objective=True,
        # Objective function weights
        optimization_weights={
            "parts_placed_weight": 0.5,  # Higher priority on placing more parts
            "compactness_weight": 0.25,  # Medium priority on compactness
            "utilization_weight": 0.25,  # Lower priority on utilization
        },
        # Hybrid algorithm parameters
        hybrid_params={
            "generations": 5,  # Number of generations for DE algorithm
            "population_size": 1,  # Population size for DE algorithm
            "rotation_increment": 30,  # Rotation increment in degrees
            "max_rotation": 60,  # Maximum rotation to try
            "grid_size": 50.0,  # Grid size for placement (larger = faster)
        },
        # Output file prefix (useful for comparing different settings)
        output_prefix="30inch_",
        # Random seed for reproducibility
        random_seed=42,
    )
