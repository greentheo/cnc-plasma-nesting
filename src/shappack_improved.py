import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union
import random


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Shape:
    points: List[Point]
    gcode_lines: List[str]
    original_idx: int = -1  # Stores the original index in the source file

    def to_shapely_polygon(self) -> Polygon:
        """Convert to Shapely polygon for collision detection"""
        coords = [(p.x, p.y) for p in self.points]
        if len(coords) < 3:
            # Create a small circle for very small shapes
            return ShapelyPoint(coords[0] if coords else (0, 0)).buffer(0.1)
        return Polygon(coords)

    def bounds(self):
        """Get bounding box of shape"""
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def width(self):
        minx, _, maxx, _ = self.bounds()
        return maxx - minx

    def height(self):
        _, miny, _, maxy = self.bounds()
        return maxy - miny

    def center(self):
        minx, miny, maxx, maxy = self.bounds()
        return (minx + maxx) / 2, (miny + maxy) / 2


@dataclass
class Placement:
    shape: Shape
    x_offset: float
    y_offset: float
    rotation: float
    polygon: Polygon  # Transformed polygon for collision checking


class GCodeParser:
    def __init__(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.pierce_sequence_map = {}  # Maps shape index to pierce sequence
        self.original_pierce_sequences = []  # Stores original pierce sequences in order
        self.travel_height = 25.4  # Default travel height (mm)
        self.travel_feedrate = 4000  # Default travel feedrate (mm/min)
        self.end_cut_sequence = []  # Commands to end a cut

    def parse_gcode(self, gcode_text: str) -> List[Shape]:
        """Parse G-code and extract all shapes with their pierce sequences"""
        shapes = []
        current_shape_points = []
        current_shape_gcode = []
        current_pierce_sequence = []
        in_cut = False
        in_pierce = False
        shape_idx = 0

        # Try to infer torch travel parameters from the file
        self._infer_torch_travel_params(gcode_text)

        lines = gcode_text.strip().split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Skip comments and empty lines for processing (but keep for pierce sequences)
            if not line or line.startswith("("):
                i += 1
                continue

            # Detect start of pierce sequence (G0 move followed by height sensing)
            if "G0 X" in line and "Y" in line and not in_cut and not in_pierce:
                # Look ahead to see if this is a pierce sequence
                j = i
                pierce_candidate = [lines[i].strip()]
                j += 1

                while j < len(lines) and "M3" not in lines[j]:
                    pierce_candidate.append(lines[j].strip())
                    j += 1

                # If we found M3, this is a complete pierce sequence
                if j < len(lines) and "M3" in lines[j]:
                    in_pierce = True
                    pierce_candidate.append(lines[j].strip())  # Add the M3 line
                    current_pierce_sequence = pierce_candidate.copy()

                    # Move to the line after M3
                    i = j + 1
                    continue

            # Start of actual cutting after pierce
            if "H1" in line and in_pierce:
                in_pierce = False
                in_cut = True
                current_shape_gcode = []
                current_shape_points = []
                current_shape_gcode.append(line)
                i += 1
                continue

            # End of shape
            elif "M5" in line and in_cut:
                in_cut = False

                # Capture the end cut sequence if not already captured
                if not self.end_cut_sequence and i < len(lines) - 1:
                    self.end_cut_sequence = [line]  # M5
                    # Get the next lines until we have a complete end sequence
                    j = i + 1
                    while j < len(lines) and (
                        not lines[j].strip().startswith("G0 X") or "G0 Z" in lines[j]
                    ):
                        self.end_cut_sequence.append(lines[j].strip())
                        j += 1

                if current_shape_points:
                    # Save the shape with its gcode
                    shape = Shape(
                        points=current_shape_points.copy(),
                        gcode_lines=current_shape_gcode.copy(),
                    )
                    shapes.append(shape)

                    # Store pierce sequence for this shape
                    self.pierce_sequence_map[shape_idx] = current_pierce_sequence
                    # Set original index in the shape
                    shape.original_idx = shape_idx
                    shape_idx += 1

                current_shape_points = []
                current_shape_gcode = []
                # Before moving to next sequence, check for travel height
                if line.startswith("G0 Z"):
                    try:
                        # Extract the travel height (e.g. G0 Z25.4)
                        height_match = re.search(r"Z([-+]?\d*\.?\d+)", line)
                        if height_match:
                            self.travel_height = float(height_match.group(1))
                    except:
                        pass

                current_pierce_sequence = []
                i += 1
                continue

            # Parse movement commands during cutting
            if in_cut:
                if "G1" in line or "G2" in line or "G3" in line or "G0" in line:
                    x, y = self._extract_xy(line)
                    if x is not None:
                        self.current_x = x
                    if y is not None:
                        self.current_y = y

                    if "G1" in line or "G2" in line or "G3" in line:
                        current_shape_points.append(
                            Point(self.current_x, self.current_y)
                        )
                        current_shape_gcode.append(line)
                    elif "G0" in line:
                        # Include G0 moves that are part of the cut path
                        current_shape_gcode.append(line)
                else:
                    current_shape_gcode.append(line)

            i += 1

        return shapes

    def _infer_torch_travel_params(self, gcode_text):
        """Infer torch travel parameters from the G-code"""
        lines = gcode_text.strip().split("\n")

        # Look for travel height after cut
        for i, line in enumerate(lines):
            if "M5" in line and i < len(lines) - 3:  # End of cut
                # Look for travel height in next few lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "G0 Z" in lines[j]:
                        # Extract Z value
                        z_match = re.search(r"Z([-+]?\d*\.?\d+)", lines[j])
                        if z_match:
                            self.travel_height = float(z_match.group(1))
                            break
                break

        # Look for travel speed (F value in G0 command)
        for line in lines:
            if "G0 " in line and " F" in line:
                f_match = re.search(r"F(\d+\.?\d*)", line)
                if f_match:
                    self.travel_feedrate = float(f_match.group(1))
                    break

        print(f"Inferred torch travel height: {self.travel_height} mm")
        print(f"Using travel feedrate: {self.travel_feedrate} mm/min")

    def _extract_xy(self, line) -> Tuple[float, float]:
        """Extract X and Y coordinates from G-code line"""
        x = None
        y = None

        x_match = re.search(r"X([-+]?\d*\.?\d+)", line)
        y_match = re.search(r"Y([-+]?\d*\.?\d+)", line)

        if x_match:
            x = float(x_match.group(1))
        if y_match:
            y = float(y_match.group(1))

        return x, y


class NestingAlgorithm:
    def __init__(
        self,
        sheet_width: float,
        sheet_height: float,
        spacing: float = 5.0,
        rotation_step: float = 15.0,
    ):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.spacing = spacing
        self.rotation_step = rotation_step  # Rotation increment in degrees
        self.sheet_polygon = Polygon(
            [(0, 0), (sheet_width, 0), (sheet_width, sheet_height), (0, sheet_height)]
        )

    def nest_shapes_optimal(
        self, shapes: List[Shape], max_count: int = 100
    ) -> List[Placement]:
        """
        Nest shapes with collision detection and continuous rotation testing.
        Uses random selection from shape pool to maintain proper ratios.
        """
        placements = []
        placed_polygons = []

        # Generate rotation angles (0 to 360 in steps)
        rotations = list(range(0, 360, int(self.rotation_step)))

        # Calculate grid search parameters
        step_size = 5  # mm - search grid resolution

        # Create a pool of shapes with their indices
        shape_pool = []
        shape_counts = {}  # Track how many of each original shape

        # Build the pool with all requested copies
        for shape in shapes:
            if shape.original_idx not in shape_counts:
                shape_counts[shape.original_idx] = 0
            shape_pool.append(shape)

        # Track shapes placed per type
        shapes_placed_by_type = {idx: 0 for idx in shape_counts.keys()}
        total_requested = len(shape_pool)

        print(f"Attempting to place {total_requested} shapes total...")
        print(
            f"Using rotation step of {self.rotation_step} degrees ({len(rotations)} angles tested)"
        )

        # Randomly shuffle the pool to ensure good mixing
        random.shuffle(shape_pool)

        shapes_not_placed = []

        for shape in shape_pool:
            placed = False
            best_placement = None
            best_area_used = float("inf")

            # Try each rotation
            for rotation in rotations:
                # Create rotated polygon
                poly = shape.to_shapely_polygon()
                cx, cy = shape.center()
                rotated_poly = rotate(poly, rotation, origin=(cx, cy))

                # Get bounds of rotated shape
                minx, miny, maxx, maxy = rotated_poly.bounds
                shape_width = maxx - minx
                shape_height = maxy - miny

                # Try positions on a grid
                for y in np.arange(
                    self.spacing,
                    self.sheet_height - shape_height - self.spacing,
                    step_size,
                ):
                    for x in np.arange(
                        self.spacing,
                        self.sheet_width - shape_width - self.spacing,
                        step_size,
                    ):
                        # Calculate offset to move shape to this position
                        x_offset = x - minx
                        y_offset = y - miny

                        # Transform polygon to test position
                        test_poly = translate(
                            rotated_poly, xoff=x_offset, yoff=y_offset
                        )

                        # Add spacing buffer
                        buffered_poly = test_poly.buffer(self.spacing / 2)

                        # Check if it fits in sheet
                        if not self.sheet_polygon.contains(buffered_poly):
                            continue

                        # Check collision with existing placements
                        collision = False
                        for placed_poly in placed_polygons:
                            if buffered_poly.intersects(placed_poly):
                                collision = True
                                break

                        if not collision:
                            # Calculate how far from origin (prefer bottom-left)
                            area_metric = x + y

                            if area_metric < best_area_used:
                                best_area_used = area_metric
                                best_placement = Placement(
                                    shape=shape,
                                    x_offset=x_offset,
                                    y_offset=y_offset,
                                    rotation=rotation,
                                    polygon=test_poly,
                                )
                                placed = True

            # Add best placement if found
            if best_placement:
                placements.append(best_placement)
                placed_polygons.append(best_placement.polygon.buffer(self.spacing / 2))
                shapes_placed_by_type[shape.original_idx] = (
                    shapes_placed_by_type.get(shape.original_idx, 0) + 1
                )
            else:
                shapes_not_placed.append(shape)

        # Calculate utilization
        total_placed_area = sum([p.polygon.area for p in placements])
        sheet_area = self.sheet_width * self.sheet_height
        utilization = (total_placed_area / sheet_area) * 100

        # Print placement summary
        print(f"\n=== Placement Summary ===")
        print(f"Total shapes placed: {len(placements)} out of {total_requested}")
        print(f"Could not place: {len(shapes_not_placed)} shapes")
        print(f"Sheet utilization: {utilization:.1f}%")

        if shapes_placed_by_type:
            print("\nShapes placed by type:")
            for idx, count in shapes_placed_by_type.items():
                print(f"  Shape type {idx}: {count} placed")

        if len(shapes_not_placed) > 0:
            print(
                f"\nNote: Sheet is full. Could not place {len(shapes_not_placed)} remaining shapes."
            )
            print("Consider using a larger sheet or reducing the number of parts.")

        print(
            f"\nOptimization used {len(rotations)} rotation angles (0-360° in {self.rotation_step}° steps)"
        )

        return placements

    def visualize_nesting(
        self, placements: List[Placement], filename: str = "nesting_layout.png"
    ):
        """Create a visualization of the nested layout with rotation angles"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Draw sheet boundary
        sheet_rect = plt.Rectangle(
            (0, 0),
            self.sheet_width,
            self.sheet_height,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(sheet_rect)

        # Color map for different parts (by original shape type)
        unique_shapes = len(set(p.shape.original_idx for p in placements))
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, unique_shapes)))

        # Track rotation angles used
        rotation_angles = {}

        # Draw each placed shape
        for idx, placement in enumerate(placements):
            poly = placement.polygon
            shape_type = placement.shape.original_idx

            # Track rotation angles
            if placement.rotation not in rotation_angles:
                rotation_angles[placement.rotation] = 0
            rotation_angles[placement.rotation] += 1

            if poly.geom_type == "Polygon":
                coords = list(poly.exterior.coords)
                patch = MPLPolygon(
                    coords,
                    alpha=0.6,
                    facecolor=colors[shape_type % len(colors)],
                    edgecolor="darkblue",
                    linewidth=1,
                )
                ax.add_patch(patch)

                # Add label with rotation angle
                centroid = poly.centroid
                ax.text(
                    centroid.x,
                    centroid.y,
                    f"{idx + 1}\n{placement.rotation}°",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )

        # Add statistics text
        total_area = sum(p.polygon.area for p in placements)
        sheet_area = self.sheet_width * self.sheet_height
        utilization = (total_area / sheet_area) * 100

        stats_text = f"Parts: {len(placements)}\nUtilization: {utilization:.1f}%\nRotation step: {self.rotation_step}°"
        ax.text(
            self.sheet_width + 5,
            self.sheet_height * 0.9,
            stats_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5),
        )

        # Show rotation distribution
        if rotation_angles:
            angles_text = "Rotation angles used:\n"
            for angle, count in sorted(rotation_angles.items()):
                angles_text += f"{angle}°: {count} parts\n"
            ax.text(
                self.sheet_width + 5,
                self.sheet_height * 0.5,
                angles_text,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5),
            )

        ax.set_xlim(-10, self.sheet_width + 100)
        ax.set_ylim(-10, self.sheet_height + 10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(
            f"Nesting Layout: {len(placements)} parts on {self.sheet_width}x{self.sheet_height}mm sheet"
        )

        # Calculate efficiency
        total_part_area = sum(p.polygon.area for p in placements)
        sheet_area = self.sheet_width * self.sheet_height
        efficiency = (total_part_area / sheet_area) * 100

        ax.text(
            self.sheet_width / 2,
            -5,
            f"Sheet Utilization: {efficiency:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"\nLayout visualization saved to {filename}")
        print(f"Sheet utilization: {efficiency:.1f}%")
        plt.close()


class GCodeGenerator:
    @staticmethod
    def transform_gcode(
        shape: Shape, x_offset: float, y_offset: float, rotation: float = 0.0
    ) -> List[str]:
        """Transform G-code by translation and rotation"""
        transformed_lines = []

        # Get shape center for rotation
        cx, cy = shape.center()

        for line in shape.gcode_lines:
            new_line = line

            # Extract and transform coordinates
            x_match = re.search(r"X([-+]?\d*\.?\d+)", line)
            y_match = re.search(r"Y([-+]?\d*\.?\d+)", line)

            if x_match or y_match:
                x = float(x_match.group(1)) if x_match else None
                y = float(y_match.group(1)) if y_match else None

                if x is not None and y is not None:
                    # Apply rotation if needed
                    if rotation != 0:
                        rad = math.radians(rotation)
                        dx = x - cx
                        dy = y - cy
                        x = cx + dx * math.cos(rad) - dy * math.sin(rad)
                        y = cy + dx * math.sin(rad) + dy * math.cos(rad)

                    # Apply translation
                    x += x_offset
                    y += y_offset

                    # Replace coordinates in line
                    if x_match:
                        new_line = re.sub(r"X[-+]?\d*\.?\d+", f"X{x:.3f}", new_line)
                    if y_match:
                        new_line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{y:.3f}", new_line)

            transformed_lines.append(new_line)

        return transformed_lines

    @staticmethod
    def generate_nested_gcode_with_optimal_travel(
        original_gcodes: List[str],
        placements: List[Placement],
        parser=None,  # type: Optional[GCodeParser]
    ) -> str:
        """Generate complete nested G-code file with optimal travel path"""
        # We'll use the first file to extract header and footer
        lines = original_gcodes[0].strip().split("\n")

        # Find header and footer
        header = []
        footer = []
        in_header = True

        for i, line in enumerate(lines):
            if "M3" in line and i > 0 and "Pierce" in lines[i - 1]:
                in_header = False
            if in_header:
                header.append(line)
            if "M30" in line:
                footer.append(line)

        # Build new G-code
        output = header + ["", "(Nested parts with optimal travel path)", ""]

        # Add initial position at origin with inferred travel height
        travel_height = 25.4  # Default
        travel_feedrate = 4000  # Default
        end_cut_sequence = ["H0", "M5", f"G0 Z{travel_height:.3f}"]

        if parser:
            travel_height = parser.travel_height
            travel_feedrate = parser.travel_feedrate
            if parser.end_cut_sequence:
                end_cut_sequence = parser.end_cut_sequence.copy()
                # Ensure the last command has the correct Z height
                for i in range(len(end_cut_sequence)):
                    if "G0 Z" in end_cut_sequence[i]:
                        end_cut_sequence[i] = f"G0 Z{travel_height:.3f}"

        output.append(f"G0 Z{travel_height:.3f}")
        output.append(f"G0 X0 Y0 F{travel_feedrate:.1f}")
        output.append("(Starting from origin)")

        # Calculate travel points (torch on/off positions) for each placement
        torch_points = []
        for idx, placement in enumerate(placements):
            # Get torch on point (entry point)
            poly = placement.polygon
            entry_point = (
                poly.bounds[0],
                poly.bounds[1],
            )  # Use bottom-left corner as entry

            # Store shape info with its entry point
            torch_points.append(
                {
                    "idx": idx,
                    "placement": placement,
                    "point": entry_point,
                    "processed": False,
                }
            )

        # Current torch position starts at origin (0,0)
        current_pos = (0, 0)

        # Process all placements in optimal travel order
        while any(not p["processed"] for p in torch_points):
            # Find closest unprocessed point to current position
            min_dist = float("inf")
            next_point_idx = -1

            for i, point_info in enumerate(torch_points):
                if not point_info["processed"]:
                    dist = (
                        (current_pos[0] - point_info["point"][0]) ** 2
                        + (current_pos[1] - point_info["point"][1]) ** 2
                    ) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        next_point_idx = i

            # Mark as processed and move to this point
            point_info = torch_points[next_point_idx]
            point_info["processed"] = True

            # Add this part to the output
            placement = point_info["placement"]
            shape = placement.shape
            pierce_x, pierce_y = point_info["point"]
            current_pos = (pierce_x, pierce_y)

            # Add part info
            output.append(
                f"(Part {point_info['idx'] + 1} - Original index: {shape.original_idx} - Rotation: {placement.rotation}°)"
            )

            # If we need to travel to this point, ensure torch is at travel height
            if next_point_idx > 0:  # Not the first point
                output.append(f"G0 Z{travel_height:.3f}")
                output.append(
                    f"G0 X{pierce_x:.3f} Y{pierce_y:.3f} F{travel_feedrate:.1f}"
                )

            # Use original pierce sequence if available, otherwise use default
            original_pierce_idx = shape.original_idx
            if (
                parser
                and original_pierce_idx >= 0
                and original_pierce_idx in parser.pierce_sequence_map
            ):
                # Get the original pierce sequence
                pierce_seq = parser.pierce_sequence_map[original_pierce_idx]

                # Transform pierce position to match new placement
                first_line = pierce_seq[0]  # This should be the G0 X... Y... line
                x_match = re.search(r"X([-+]?\d*\.?\d+)", first_line)
                y_match = re.search(r"Y([-+]?\d*\.?\d+)", first_line)

                if x_match and y_match:
                    orig_x = float(x_match.group(1))
                    orig_y = float(y_match.group(1))

                    # Apply the same transformation as the shape
                    # First rotate around original center (if needed)
                    cx, cy = shape.center()
                    if placement.rotation != 0:
                        rad = math.radians(placement.rotation)
                        dx = orig_x - cx
                        dy = orig_y - cy
                        new_x = cx + dx * math.cos(rad) - dy * math.sin(rad)
                        new_y = cy + dx * math.sin(rad) + dy * math.cos(rad)
                    else:
                        new_x = orig_x
                        new_y = orig_y

                    # Then apply offset
                    new_x += placement.x_offset
                    new_y += placement.y_offset

                    # Update the G0 line with new position
                    modified_g0 = re.sub(
                        r"X[-+]?\d*\.?\d+", f"X{new_x:.3f}", first_line
                    )
                    modified_g0 = re.sub(
                        r"Y[-+]?\d*\.?\d+", f"Y{new_y:.3f}", modified_g0
                    )

                    # Add the transformed pierce sequence (skip the G0 X Y since we already did it)
                    # Only add the G0 line if we're on the first shape
                    if next_point_idx == 0:  # First shape only
                        output.append(modified_g0)
                    for line in pierce_seq[1:]:
                        output.append(line)
                else:
                    # Fallback to default if something goes wrong with position extraction
                    # Skip the G0 positioning if we've already done it
                    if next_point_idx == 0:  # First shape only
                        output.extend(
                            [
                                f"G0 X{pierce_x:.3f} Y{pierce_y:.3f}",
                                "G92 Z0.",
                                "G38.2 Z-127. F2540.",
                                "G38.4 Z12.7 F508.",
                                "G92 Z0.",
                                "G0 Z1.016",
                                "G92 Z0.",
                                "G0 Z3.81",
                                "M3",
                                "G4 P0.5",
                                "G0 Z1.6",
                                "H1",
                            ]
                        )
                    else:
                        output.extend(
                            [
                                "G92 Z0.",
                                "G38.2 Z-127. F2540.",
                                "G38.4 Z12.7 F508.",
                                "G92 Z0.",
                                "G0 Z1.016",
                                "G92 Z0.",
                                "G0 Z3.81",
                                "M3",
                                "G4 P0.5",
                                "G0 Z1.6",
                                "H1",
                            ]
                        )
            else:
                # Use default pierce sequence
                # Skip the G0 positioning if we've already done it
                if next_point_idx == 0:  # First shape only
                    output.extend(
                        [
                            f"G0 X{pierce_x:.3f} Y{pierce_y:.3f}",
                            "G92 Z0.",
                            "G38.2 Z-127. F2540.",
                            "G38.4 Z12.7 F508.",
                            "G92 Z0.",
                            "G0 Z1.016",
                            "G92 Z0.",
                            "G0 Z3.81",
                            "M3",
                            "G4 P0.5",
                            "G0 Z1.6",
                            "H1",
                        ]
                    )
                else:
                    output.extend(
                        [
                            "G92 Z0.",
                            "G38.2 Z-127. F2540.",
                            "G38.4 Z12.7 F508.",
                            "G92 Z0.",
                            "G0 Z1.016",
                            "G92 Z0.",
                            "G0 Z3.81",
                            "M3",
                            "G4 P0.5",
                            "G0 Z1.6",
                            "H1",
                        ]
                    )

            # Add transformed cut path
            transformed = GCodeGenerator.transform_gcode(
                shape, placement.x_offset, placement.y_offset, placement.rotation
            )
            output.extend(transformed)

            # End cut with inferred travel height and sequence
            if "G0 Z" in end_cut_sequence[-1]:
                output.extend(end_cut_sequence + [""])
            else:
                output.extend(end_cut_sequence + [f"G0 Z{travel_height:.3f}", ""])

        # Return to origin after all cuts (at travel height)
        output.append("(Returning to origin)")
        output.append(f"G0 Z{travel_height:.3f}")
        output.append(f"G0 X0 Y0 F{travel_feedrate:.1f}")
        output.append("")

        # Add footer
        output.extend(footer)
        return "\n".join(output)

    @staticmethod
    def generate_nested_gcode(
        original_gcode: str,
        placements: List[Placement],
        parser=None,  # type: Optional[GCodeParser]
    ) -> str:
        """
        Legacy method for backward compatibility.
        Generate complete nested G-code file
        """
        return GCodeGenerator.generate_nested_gcode_with_optimal_travel(
            [original_gcode], placements, parser
        )


def nest_gcode_files(
    input_files_with_counts: List[Tuple[str, int]],
    sheet_width: float,
    sheet_height: float,
    spacing: float = 5.0,
    rotation_step: float = 15.0,
) -> Tuple[str, int]:
    """
    Main function to nest G-code shapes with maximum packing from multiple files.

    Args:
        input_files_with_counts: List of tuples containing (file_content, count_to_place)
        sheet_width: Width of sheet in mm or inches (match G-code units)
        sheet_height: Height of sheet in mm or inches
        spacing: Minimum spacing between parts
        rotation_step: Rotation increment in degrees (smaller = better packing but slower)

    Returns:
        Tuple of (nested G-code string, number of parts placed)
    """
    # Parse all G-code files and prepare shapes with quantities
    all_shapes = []
    parser = GCodeParser()

    for input_gcode, count in input_files_with_counts:
        file_shapes = parser.parse_gcode(input_gcode)
        print(f"Found {len(file_shapes)} shapes in G-code file")

        # Add shapes with requested count (original_idx is already set in parse_gcode)

        for _ in range(count):
            all_shapes.extend(file_shapes)

    print(f"Total shapes to place: {len(all_shapes)}")

    # Nest shapes with optimal packing
    nester = NestingAlgorithm(sheet_width, sheet_height, spacing, rotation_step)
    placements = nester.nest_shapes_optimal(
        all_shapes, 1
    )  # We're not using "sets" anymore

    # Visualize the result
    nester.visualize_nesting(placements)

    # Generate new G-code with optimized travel paths
    nested_gcode = GCodeGenerator.generate_nested_gcode_with_optimal_travel(
        [gcode for gcode, _ in input_files_with_counts], placements, parser
    )

    print(f"Using inferred travel height: {parser.travel_height} mm")
    print(f"Using travel feedrate: {parser.travel_feedrate} mm/min")

    return nested_gcode, len(placements)


def nest_gcode_file(
    input_gcode: str,
    sheet_width: float,
    sheet_height: float,
    spacing: float = 5.0,
    max_count: int = 100,
    rotation_step: float = 15.0,
) -> Tuple[str, int]:
    """
    Main function to nest G-code shapes with maximum packing.

    Legacy version that accepts a single file.

    Args:
        input_gcode: Original G-code as string
        sheet_width: Width of sheet in mm or inches (match G-code units)
        sheet_height: Height of sheet in mm or inches
        spacing: Minimum spacing between parts
        max_count: Maximum number of complete sets to attempt
        rotation_step: Rotation increment in degrees (smaller = better packing but slower)

    Returns:
        Tuple of (nested G-code string, number of parts placed)
    """
    # Forward to new implementation
    return nest_gcode_files(
        [(input_gcode, max_count)], sheet_width, sheet_height, spacing, rotation_step
    )


# Example usage
if __name__ == "__main__":
    # Configure nesting parameters
    SHEET_WIDTH = 48 * 25.4  # mm (adjust to your sheet size)
    SHEET_HEIGHT = 96 * 25.4  # mm
    SPACING = 3  # mm between parts

    # Define up to 6 NC files and the quantities to place
    input_files_with_counts = []

    # File 1 - Required
    file_path_1 = "geodesic_6_bottom.nc"
    count_1 = 10
    with open(file_path_1, "r") as f:
        gcode_1 = f.read()
        input_files_with_counts.append((gcode_1, count_1))

    # File 2 - Optional
    file_path_2 = "geodesic_6_top.nc"
    count_2 = 25
    try:
        with open(file_path_2, "r") as f:
            gcode_2 = f.read()
            input_files_with_counts.append((gcode_2, count_2))
    except FileNotFoundError:
        pass

    # File 3 - Optional
    file_path_3 = "geodesic_5_top.nc"
    count_3 = 5
    try:
        if count_3 > 0:
            with open(file_path_3, "r") as f:
                gcode_3 = f.read()
                input_files_with_counts.append((gcode_3, count_3))
    except FileNotFoundError:
        pass

    # File 4 - Optional
    file_path_4 = "geodesic_5_bottom.nc"
    count_4 = 5
    try:
        if count_4 > 0:
            with open(file_path_4, "r") as f:
                gcode_4 = f.read()
                input_files_with_counts.append((gcode_4, count_4))
    except FileNotFoundError:
        pass

    # File 5 - Optional
    file_path_5 = "geodesic_4_top.nc"
    count_5 = 15
    try:
        if count_5 > 0:
            with open(file_path_5, "r") as f:
                gcode_5 = f.read()
                input_files_with_counts.append((gcode_5, count_5))
    except FileNotFoundError:
        pass

    # File 6 - Optional
    file_path_6 = "geodesic_4_bottom.nc"
    count_6 = 5
    try:
        if count_6 > 0:
            with open(file_path_6, "r") as f:
                gcode_6 = f.read()
                input_files_with_counts.append((gcode_6, count_6))
    except FileNotFoundError:
        pass

    # Generate nested G-code
    nested, parts_count = nest_gcode_files(
        input_files_with_counts, SHEET_WIDTH, SHEET_HEIGHT, SPACING
    )

    # Save result
    with open("nested_output.nc", "w") as f:
        f.write(nested)

    print(f"\nNested G-code saved to nested_output.nc")
    print(f"Total parts placed: {parts_count}")
