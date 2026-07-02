#!/usr/bin/env python3
"""
Optimized test script for the hybrid nesting algorithm.
Compares performance against both the original greedy algorithm and the DE algorithm.
Includes optimizations for improved performance and results.
"""

import sys
import os
import time
import random
from pathlib import Path

# Import the different nesting algorithms
from shappack import GCodeParser, GCodeGenerator, NestingAlgorithm
from shappack_de import DENetworkingAlgorithm
from shappack_hybrid import HybridNestingAlgorithm


def test_hybrid_nesting():
    """Run a comparison test of the hybrid nesting algorithm with optimized parameters."""
    # Set random seed for reproducibility
    random.seed(42)

    # Configure nesting parameters
    SHEET_WIDTH = 24 * 25.4  # mm (24 inches)
    SHEET_HEIGHT = 48 * 25.4  # mm (48 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("NESTING ALGORITHM COMPARISON TEST (OPTIMIZED)")
    print("=" * 70)
    print(f"\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm")
    print(f"Spacing: {SPACING} mm\n")

    # Test with several files and manageable number of copies
    test_files = [
        ("geodesic_6_top.nc", 5),  # Reduced from 15
        ("geodesic_6_bottom.nc", 5),  # Reduced from 20
        ("geodesic_5_top.nc", 2),  # Was 0
        ("geodesic_5_bottom.nc", 2),  # Reduced from 4
        ("geodesic_4_top.nc", 2),  # Was 0
        ("geodesic_4_bottom.nc", 4),  # Reduced from 10
    ]

    # Parse files
    parser = GCodeParser()
    all_shapes = []
    original_gcodes = []

    print("Parsing G-code files...")
    for idx, (file_path, max_count) in enumerate(test_files):
        if not os.path.exists(file_path):
            print(f"✗ Missing: {file_path}")
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

    # ======= Test 1: Original Greedy Algorithm =======
    print("\n" + "=" * 70)
    print("TESTING ORIGINAL GREEDY ALGORITHM")
    print("=" * 70)

    greedy_algo = NestingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    greedy_start_time = time.time()
    greedy_placements = greedy_algo.nest_shapes_optimal(
        shapes=all_shapes,
        max_count=1,  # Already duplicated
    )
    greedy_elapsed = time.time() - greedy_start_time

    greedy_sheet_area = SHEET_WIDTH * SHEET_HEIGHT
    greedy_total_area = sum(p.polygon.area for p in greedy_placements)
    greedy_utilization = (greedy_total_area / greedy_sheet_area) * 100

    print("\nGreedy Algorithm Results:")
    print(f"Shapes placed: {len(greedy_placements)} / {len(all_shapes)}")
    print(f"Utilization: {greedy_utilization:.2f}%")
    print(f"Time: {greedy_elapsed:.1f} seconds")

    greedy_viz_file = "test_greedy_result.png"
    greedy_algo.visualize_nesting(greedy_placements, filename=greedy_viz_file)
    print(f"✓ Visualization saved to: {greedy_viz_file}")

    # ======= Test 2: DE Algorithm =======
    print("\n" + "=" * 70)
    print("TESTING DE ALGORITHM")
    print("=" * 70)

    de_params = {
        "generations": 3,  # Small number of generations
        "population_size": 10,  # Modest population size
        "rotation_increment": 30.0,  # 30-degree increments (fewer rotation options)
        "workers": 1,  # Single worker to avoid pickling issues
    }

    print(f"Generations: {de_params['generations']}")
    print(f"Population: {de_params['population_size']}")
    print(f"Rotation increment: {de_params['rotation_increment']}°")

    de_algo = DENetworkingAlgorithm(
        sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
    )

    de_start_time = time.time()
    de_placements, de_metrics = de_algo.nest_shapes_de(
        shapes=all_shapes,
        max_count=1,  # Already duplicated
        generations=de_params["generations"],
        population_size=de_params["population_size"],
        rotation_increment=de_params["rotation_increment"],
        workers=de_params["workers"],
        verbose=True,
    )
    de_elapsed = time.time() - de_start_time

    de_sheet_area = SHEET_WIDTH * SHEET_HEIGHT
    de_total_area = sum(p.polygon.area for p in de_placements)
    de_utilization = (de_total_area / de_sheet_area) * 100

    print("\nDE Algorithm Results:")
    print(f"Shapes placed: {len(de_placements)} / {len(all_shapes)}")
    print(f"Utilization: {de_utilization:.2f}%")
    print(f"Time: {de_elapsed:.1f} seconds")

    de_viz_file = "test_de_result.png"
    de_algo.visualize_nesting(de_placements, filename=de_viz_file)
    print(f"✓ Visualization saved to: {de_viz_file}")

    # ======= Test 3: Hybrid Algorithm (Optimized) =======
    print("\n" + "=" * 70)
    print("TESTING HYBRID ALGORITHM (OPTIMIZED)")
    print("=" * 70)

    hybrid_params = {
        "generations": 5,  # Few generations for testing
        "population_size": 20,  # Larger population to explore more options
        "rotation_increment": 30.0,  # 30-degree increments (fewer rotation options)
        "compactness_weight": 0.3,  # Balance between utilization and compactness
        "grid_size": 25.0,  # Moderate grid size for faster search but still precise
    }

    print(f"Generations: {hybrid_params['generations']}")
    print(f"Population: {hybrid_params['population_size']}")
    print(f"Rotation increment: {hybrid_params['rotation_increment']}° (0-90° only)")
    print(f"Compactness weight: {hybrid_params['compactness_weight']}")
    print(f"Grid size: {hybrid_params['grid_size']} mm")
    print(f"Workers: 1 (single process)")

    hybrid_algo = HybridNestingAlgorithm(
        sheet_width=SHEET_WIDTH,
        sheet_height=SHEET_HEIGHT,
        spacing=SPACING,
        grid_size=hybrid_params["grid_size"],
        compactness_weight=hybrid_params["compactness_weight"],
    )

    hybrid_start_time = time.time()
    hybrid_placements, hybrid_metrics = hybrid_algo.nest_shapes_hybrid(
        shapes=all_shapes,
        generations=hybrid_params["generations"],
        population_size=hybrid_params["population_size"],
        rotation_increment=hybrid_params["rotation_increment"],
        verbose=True,
        max_rotation=90,  # Set maximum rotation to 90 degrees (symmetric shapes)
        workers=1,  # Single worker to avoid pickling issues
        seed=42,  # Set seed for reproducibility
    )
    hybrid_elapsed = time.time() - hybrid_start_time

    hybrid_sheet_area = SHEET_WIDTH * SHEET_HEIGHT
    hybrid_total_area = sum(p.polygon.area for p in hybrid_placements)
    hybrid_utilization = (hybrid_total_area / hybrid_sheet_area) * 100

    print("\nHybrid Algorithm Results:")
    print(f"Shapes placed: {len(hybrid_placements)} / {len(all_shapes)}")
    print(f"Utilization: {hybrid_utilization:.2f}%")
    print(f"Time: {hybrid_elapsed:.1f} seconds")

    hybrid_viz_file = "test_hybrid_result.png"
    hybrid_algo.visualize_nesting(hybrid_placements, filename=hybrid_viz_file)
    print(f"✓ Visualization saved to: {hybrid_viz_file}")

    # Show progress visualization
    progress_file = "hybrid_progress.png"
    hybrid_algo.visualize_progress(hybrid_metrics, filename=progress_file)
    print(f"Progress visualization saved to {progress_file}")
    print(f"✓ Progress visualization saved to: {progress_file}")

    # ======= Comparison Summary =======
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\nAlgorithm | Parts Placed | Utilization | Compactness | Time (s)")
    print("-" * 75)

    # Calculate compactness for greedy (if available)
    greedy_compactness = 0
    if greedy_placements:
        try:
            min_x = min(
                min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in greedy_placements
            )
            min_y = min(
                min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in greedy_placements
            )
            max_x = max(
                max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in greedy_placements
            )
            max_y = max(
                max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in greedy_placements
            )
            bounding_box_area = (max_x - min_x) * (max_y - min_y)
            greedy_compactness = (
                (greedy_total_area / bounding_box_area) * 100
                if bounding_box_area > 0
                else 0
            )
        except Exception:
            greedy_compactness = 0

    # Calculate compactness for DE (if available)
    de_compactness = 0
    if de_placements:
        try:
            min_x = min(
                min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in de_placements
            )
            min_y = min(
                min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in de_placements
            )
            max_x = max(
                max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in de_placements
            )
            max_y = max(
                max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in de_placements
            )
            bounding_box_area = (max_x - min_x) * (max_y - min_y)
            de_compactness = (
                (de_total_area / bounding_box_area) * 100
                if bounding_box_area > 0
                else 0
            )
        except Exception:
            de_compactness = 0

    # Calculate compactness for hybrid (if available)
    hybrid_compactness = 0
    if hybrid_placements:
        try:
            min_x = min(
                min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in hybrid_placements
            )
            min_y = min(
                min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in hybrid_placements
            )
            max_x = max(
                max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                for p in hybrid_placements
            )
            max_y = max(
                max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                for p in hybrid_placements
            )
            bounding_box_area = (max_x - min_x) * (max_y - min_y)
            hybrid_compactness = (
                (hybrid_total_area / bounding_box_area) * 100
                if bounding_box_area > 0
                else 0
            )
        except Exception:
            hybrid_compactness = 0

    # Print comparison results
    print(
        f"Greedy    | {len(greedy_placements):12} | {greedy_utilization:10.2f}% | {greedy_compactness:10.2f}% | {greedy_elapsed:7.1f}"
    )
    print(
        f"DE        | {len(de_placements):12} | {de_utilization:10.2f}% | {de_compactness:10.2f}% | {de_elapsed:7.1f}"
    )
    print(
        f"Hybrid    | {len(hybrid_placements):12} | {hybrid_utilization:10.2f}% | {hybrid_compactness:10.2f}% | {hybrid_elapsed:7.1f}"
    )

    # Calculate scores and find best algorithm
    def calculate_score(placements, utilization, compactness):
        if not placements:
            return 0
        # Weight parts most heavily, then utilization, then compactness
        parts_weight = 0.6
        util_weight = 0.3
        compact_weight = 0.1
        normalized_parts = len(placements) / len(all_shapes)
        normalized_util = utilization / 100.0
        normalized_compact = compactness / 100.0
        return (
            parts_weight * normalized_parts
            + util_weight * normalized_util
            + compact_weight * normalized_compact
        )

    # Calculate scores
    greedy_score = calculate_score(
        greedy_placements, greedy_utilization, greedy_compactness
    )
    de_score = calculate_score(de_placements, de_utilization, de_compactness)
    hybrid_score = calculate_score(
        hybrid_placements, hybrid_utilization, hybrid_compactness
    )

    scores = {"Greedy": greedy_score, "DE": de_score, "Hybrid": hybrid_score}
    best_algo_name = max(scores, key=scores.get)

    # Get best placements
    if best_algo_name == "Greedy":
        best_placements = greedy_placements
    elif best_algo_name == "DE":
        best_placements = de_placements
    else:
        best_placements = hybrid_placements

    # Display best algorithm
    print(
        f"\nBest algorithm by combined score: {best_algo_name} (Score: {scores[best_algo_name]:.4f})"
    )
    print(
        f"Scores - Greedy: {greedy_score:.4f}, DE: {de_score:.4f}, Hybrid: {hybrid_score:.4f}"
    )

    # Generate G-code for best result
    if best_placements:
        print("\nGenerating G-code for best result...")
        gcode_generator = GCodeGenerator()

        try:
            output_gcode = gcode_generator.generate_nested_gcode_with_optimal_travel(
                original_gcodes, best_placements, parser
            )

            output_file = f"test_{best_algo_name.lower()}_output.nc"
            with open(output_file, "w") as f:
                f.write(output_gcode)

            print(f"✓ G-code saved to: {output_file}")
        except Exception as e:
            print(f"⚠ G-code generation encountered an issue: {str(e)}")

    print("\nTest complete!")


# Also create a faster test function with even fewer parts for quick evaluations
def test_hybrid_nesting_quick():
    """Run a quicker comparison test with fewer parts for rapid testing."""
    # Set random seed for reproducibility
    random.seed(42)

    # Configure nesting parameters (smaller sheet and fewer parts)
    SHEET_WIDTH = 18 * 25.4  # mm (18 inches)
    SHEET_HEIGHT = 24 * 25.4  # mm (24 inches)
    SPACING = 3  # mm between parts

    print("=" * 70)
    print("NESTING ALGORITHM QUICK TEST")
    print("=" * 70)
    print(f"\nSheet: {SHEET_WIDTH:.0f} x {SHEET_HEIGHT:.0f} mm")
    print(f"Spacing: {SPACING} mm\n")

    # Test with minimal number of parts
    test_files = [
        ("geodesic_6_top.nc", 2),
        ("geodesic_6_bottom.nc", 2),
        ("geodesic_5_top.nc", 1),
        ("geodesic_5_bottom.nc", 1),
        ("geodesic_4_top.nc", 1),
        ("geodesic_4_bottom.nc", 1),
    ]

    # Parse files
    parser = GCodeParser()
    all_shapes = []
    original_gcodes = []

    print("Parsing G-code files...")
    for idx, (file_path, max_count) in enumerate(test_files):
        if not os.path.exists(file_path):
            print(f"✗ Missing: {file_path}")
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

    print(f"\nTotal shape instances: {len(all_shapes)}")

    # Test all three algorithms with optimized parameters
    algorithms = [
        {
            "name": "Greedy",
            "algo": NestingAlgorithm(
                sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
            ),
            "func": lambda algo, shapes: algo.nest_shapes_optimal(shapes, max_count=1),
        },
        {
            "name": "DE",
            "algo": DENetworkingAlgorithm(
                sheet_width=SHEET_WIDTH, sheet_height=SHEET_HEIGHT, spacing=SPACING
            ),
            "func": lambda algo, shapes: algo.nest_shapes_de(
                shapes,
                max_count=1,
                generations=3,
                population_size=5,
                rotation_increment=30.0,
                verbose=True,
                workers=1,
            )[0],  # Return just the placements
        },
        {
            "name": "Hybrid",
            "algo": HybridNestingAlgorithm(
                sheet_width=SHEET_WIDTH,
                sheet_height=SHEET_HEIGHT,
                spacing=SPACING,
                grid_size=20.0,
                compactness_weight=0.3,
            ),
            "func": lambda algo, shapes: algo.nest_shapes_hybrid(
                shapes,
                generations=3,
                population_size=10,
                rotation_increment=30.0,
                verbose=True,
                max_rotation=90,
                workers=1,
                seed=42,
            )[0],  # Return just the placements
        },
    ]

    results = []
    for algo_info in algorithms:
        print(f"\n{'-' * 40}")
        print(f"Testing {algo_info['name']} algorithm...")

        start_time = time.time()
        placements = algo_info["func"](algo_info["algo"], all_shapes)
        elapsed = time.time() - start_time

        # Calculate metrics
        sheet_area = SHEET_WIDTH * SHEET_HEIGHT
        if placements:
            total_area = sum(p.polygon.area for p in placements)
            utilization = (total_area / sheet_area) * 100

            # Calculate compactness
            try:
                min_x = min(
                    min(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                    for p in placements
                )
                min_y = min(
                    min(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                    for p in placements
                )
                max_x = max(
                    max(p.polygon.exterior.coords, key=lambda c: c[0])[0]
                    for p in placements
                )
                max_y = max(
                    max(p.polygon.exterior.coords, key=lambda c: c[1])[1]
                    for p in placements
                )
                bounding_box_area = (max_x - min_x) * (max_y - min_y)
                compactness = (
                    (total_area / bounding_box_area) * 100
                    if bounding_box_area > 0
                    else 0
                )
            except Exception:
                compactness = 0
        else:
            utilization = 0
            compactness = 0

        # Save results
        results.append(
            {
                "name": algo_info["name"],
                "placements": placements,
                "parts_placed": len(placements),
                "utilization": utilization,
                "compactness": compactness,
                "time": elapsed,
            }
        )

        # Print results
        print(f"Parts placed: {len(placements)} / {len(all_shapes)}")
        print(f"Utilization: {utilization:.2f}%")
        print(f"Compactness: {compactness:.2f}%")
        print(f"Time: {elapsed:.2f} seconds")

        # Save visualization
        viz_file = f"test_quick_{algo_info['name'].lower()}_result.png"
        algo_info["algo"].visualize_nesting(placements, filename=viz_file)
        print(f"✓ Visualization saved to: {viz_file}")

    # Print comparison
    print("\n" + "=" * 70)
    print("QUICK TEST SUMMARY")
    print("=" * 70)

    print("\nAlgorithm | Parts Placed | Utilization | Compactness | Time (s)")
    print("-" * 75)

    for r in results:
        print(
            f"{r['name']:<9} | {r['parts_placed']:12} | {r['utilization']:10.2f}% | "
            f"{r['compactness']:10.2f}% | {r['time']:7.1f}"
        )

    print("\nTest complete!")


if __name__ == "__main__":
    # Choose which test to run
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        test_hybrid_nesting_quick()
    else:
        test_hybrid_nesting()
