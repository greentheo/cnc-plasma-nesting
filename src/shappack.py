import re
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.ops import unary_union


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

        # Look for travel height after cut - more robust detection
        # First try to find explicit travel height after M5 (torch off)
        travel_height_found = False
        for i, line in enumerate(lines):
            if "M5" in line and i < len(lines) - 5:  # End of cut
                # Look for travel height in next few lines
                for j in range(i + 1, min(i + 7, len(lines))):
                    if "G0 Z" in lines[j]:
                        # Extract Z value
                        z_match = re.search(r"Z([-+]?\d*\.?\d+)", lines[j])
                        if z_match:
                            self.travel_height = float(z_match.group(1))
                            travel_height_found = True
                            break
                if travel_height_found:
                    break

        # If not found, look for any G0 Z moves with large values (likely travel moves)
        if not travel_height_found:
            for line in lines:
                if "G0 Z" in line:
                    z_match = re.search(r"Z([-+]?\d*\.?\d+)", line)
                    if z_match:
                        z_val = float(z_match.group(1))
                        # If Z value is significant (e.g., > 10mm), it's likely a travel height
                        if z_val > 10.0:
                            self.travel_height = z_val
                            travel_height_found = True
                            break

        # Default if not found
        if not travel_height_found:
            self.travel_height = 25.4  # 1 inch default travel height

        # Identify torch on/off sequences
        self.pierce_sequence = []
        self.end_cut_sequence = []

        # Find pierce sequence (from IHS to torch on)
        in_pierce_sequence = False
        for i, line in enumerate(lines):
            # Look for IHS probe sequence start
            if "G38.2" in line and not in_pierce_sequence:
                in_pierce_sequence = True
                self.pierce_sequence = []

                # Backtrack one line to get the G0 Z0 setup
                if i > 0 and "G92 Z" in lines[i - 1]:
                    self.pierce_sequence.append(lines[i - 1])

                # Add the probe sequence
                self.pierce_sequence.append(line)
                continue

            # If in sequence, add lines until torch on
            if in_pierce_sequence:
                self.pierce_sequence.append(line)

                # End sequence at torch on and height control on
                if "M3" in line:
                    # Get a few more lines after M3 (typically includes dwell and height control)
                    for j in range(i + 1, min(i + 4, len(lines))):
                        self.pierce_sequence.append(lines[j])
                        if "H1" in lines[j]:
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
    def __init__(self, sheet_width: float, sheet_height: float, spacing: float = 5.0):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.spacing = spacing
        self.sheet_polygon = Polygon(
            [(0, 0), (sheet_width, 0), (sheet_width, sheet_height), (0, sheet_height)]
        )

    def nest_shapes_optimal(
        self, shapes: List[Shape], max_count: int = 100
    ) -> List[Placement]:
        """
        Nest shapes with collision detection and rotation testing.
        Tries to fit as many copies as possible up to max_count.
        """
        placements = []
        placed_polygons = []

        # Disable rotation - only use 0 degrees
        rotations = [0]

        # Calculate grid search parameters
        step_size = 5  # mm - search grid resolution

        parts_placed = 0
        target_sets = max_count

        print(
            f"Attempting to place up to {max_count} complete sets ({len(shapes)} shapes each)..."
        )

        for set_num in range(target_sets):
            set_placed_count = 0

            for shape_idx, shape in enumerate(shapes):
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
                                    break

                        if placed:
                            break

                    if placed:
                        break

                # Add best placement if found
                if best_placement:
                    placements.append(best_placement)
                    placed_polygons.append(
                        best_placement.polygon.buffer(self.spacing / 2)
                    )
                    set_placed_count += 1
                else:
                    # Can't place this shape, stop trying
                    print(f"Could not place shape {shape_idx + 1} of set {set_num + 1}")
                    print(
                        f"Total parts placed: {len(placements)} ({len(placements) // len(shapes)} complete sets)"
                    )
                    return placements

            if set_placed_count == len(shapes):
                parts_placed += 1
                print(
                    f"Set {set_num + 1} complete ({parts_placed} total sets, {len(placements)} total parts)"
                )
            else:
                print(f"Set {set_num + 1} incomplete - stopping")
                break

        print(
            f"\nFinal: {len(placements)} parts placed ({len(placements) // len(shapes)} complete sets)"
        )
        return placements

    def visualize_nesting(
        self, placements: List[Placement], filename: str = "nesting_layout.png"
    ):
        """Create a visualization of the nested layout"""
        fig, ax = plt.subplots(figsize=(12, 8))

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

        # Color map for different parts
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        # Draw each placed shape
        for idx, placement in enumerate(placements):
            poly = placement.polygon
            if poly.geom_type == "Polygon":
                coords = list(poly.exterior.coords)
                patch = MPLPolygon(
                    coords,
                    alpha=0.6,
                    facecolor=colors[idx % len(colors)],
                    edgecolor="darkblue",
                    linewidth=1,
                )
                ax.add_patch(patch)

                # Add label
                centroid = poly.centroid
                ax.text(
                    centroid.x,
                    centroid.y,
                    str(idx + 1),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        # Visualize the torch travel paths
        # Calculate entry points (starting point) for each placement
        torch_points = []
        for idx, placement in enumerate(placements):
            # Get entry point - use the leftmost point of the polygon (same as in gcode generation)
            poly = placement.polygon
            coords = list(poly.exterior.coords)
            leftmost_x = float("inf")
            leftmost_y = 0
            for x, y in coords:
                if x < leftmost_x:
                    leftmost_x = x
                    leftmost_y = y

            entry_point = (leftmost_x, leftmost_y)
            torch_points.append(
                {
                    "idx": idx,
                    "point": entry_point,
                    "processed": False,
                    "placement": placement,
                }
            )

        # Start at origin and draw torch travel paths
        current_pos = (0, 0)

        # Start at origin and visualize travel paths
        current_pos = (0, 0)
        unprocessed = torch_points.copy()

        # Process all placements in optimal travel order (nearest neighbor approach)
        while unprocessed:
            # Find closest unprocessed point to current position
            min_dist = float("inf")
            next_point_idx = -1

            for i, point_info in enumerate(unprocessed):
                dist = (
                    (current_pos[0] - point_info["point"][0]) ** 2
                    + (current_pos[1] - point_info["point"][1]) ** 2
                ) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    next_point_idx = i

            # Get the next point and remove it from unprocessed
            point_info = unprocessed.pop(next_point_idx)
            next_point = point_info["point"]

            # Draw torch off travel path (blue dashed line)
            next_point = point_info["point"]
            ax.plot(
                [current_pos[0], next_point[0]],
                [current_pos[1], next_point[1]],
                color="blue",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

            # Draw pierce point as an orange dot
            ax.scatter(next_point[0], next_point[1], color="orange", s=40, zorder=10)

            # Draw torch on cutting path (yellow dashed line)
            poly = point_info["placement"].polygon
            if poly.geom_type == "Polygon":
                coords = list(poly.exterior.coords)
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                # Close the path by adding the first point again
                if x_coords[0] != x_coords[-1] or y_coords[0] != y_coords[-1]:
                    x_coords.append(x_coords[0])
                    y_coords.append(y_coords[0])
                ax.plot(
                    x_coords,
                    y_coords,
                    color="yellow",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

            # Update current position
            current_pos = next_point

        # Add legend for travel paths
        ax.plot([], [], color="blue", linestyle="--", label="Torch off travel")
        ax.plot([], [], color="yellow", linestyle="--", label="Torch on cutting")
        scatter_handle = ax.scatter([], [], color="orange", s=40, label="Pierce points")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")

        ax.set_xlim(-10, self.sheet_width + 10)
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
        current_feed_rate = 2400  # Default feed rate if none is specified

        # Get shape center for rotation
        cx, cy = shape.center()

        for line in shape.gcode_lines:
            new_line = line

            # Check for G-code commands that might need feed rates
            is_g1 = line.strip().startswith("G1 ")
            is_g2 = line.strip().startswith("G2 ")
            is_g3 = line.strip().startswith("G3 ")

            # Extract and transform coordinates
            x_match = re.search(r"X([-+]?\d*\.?\d+)", line)
            y_match = re.search(r"Y([-+]?\d*\.?\d+)", line)

            # Extract feed rate from the line
            feed_match = re.search(r"F(\d*\.?\d*)", line)
            if feed_match and feed_match.group(1).strip():
                current_feed_rate = float(feed_match.group(1))

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

                    # Ensure coordinates are positive (add minimum margin of 1mm)
                    if x < 1.0:
                        x = 1.0
                    if y < 1.0:
                        y = 1.0

                    # Replace coordinates in line
                    if x_match:
                        new_line = re.sub(r"X[-+]?\d*\.?\d+", f"X{x:.3f}", new_line)
                    if y_match:
                        new_line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{y:.3f}", new_line)

                    # Ensure feed rates use Langmuir format - add to G1/G2/G3 commands if missing
                    feed_match = re.search(r"F(\d*\.?\d*)", new_line)

                    # Format feed rate for Langmuir post-processor
                    if feed_match and feed_match.group(1).strip():
                        feed_rate = float(feed_match.group(1))
                        new_line = re.sub(
                            r"F\d*\.?\d*", f"F{int(feed_rate)}.", new_line
                        )
                    # Add feed rate to G2/G3 commands if it exists on G1 commands
                    elif (is_g2 or is_g3) and current_feed_rate is not None:
                        new_line += f" F{int(current_feed_rate)}."

            transformed_lines.append(new_line)

        return transformed_lines

    @staticmethod
    def generate_nested_gcode_with_optimal_travel(
        original_gcodes: List[str],
        placements: List[Placement],
        parser=None,  # type: Optional[GCodeParser]
    ) -> str:
        """Generate complete nested G-code file with optimal travel path

        The algorithm uses a nearest-neighbor approach to determine the order of part
        cutting. For each part, it finds the closest unprocessed part to the current
        position and moves there. While not globally optimal (which would require solving
        the Traveling Salesman Problem), this approach provides a good balance between
        path efficiency and computational simplicity.
        """
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
            # Find a better entry point - use the leftmost point of the polygon
            # This is more natural for plasma cutting than using the bottom-left bounding box corner
            coords = list(poly.exterior.coords)
            leftmost_x = float("inf")
            leftmost_y = 0
            for x, y in coords:
                if x < leftmost_x:
                    leftmost_x = x
                    leftmost_y = y

            entry_point = (leftmost_x, leftmost_y)

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

        # Process all placements in optimal travel order (nearest neighbor approach)
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
                f"(Part {idx + 1} - Original index: {original_idx} - Rotation: {placement.rotation}° - Offset: X{placement.x_offset:.1f} Y{placement.y_offset:.1f})"
            )

            # Pre-process: Calculate bounds and adjust placement offset if needed to avoid negative coordinates
            min_x, min_y = float("inf"), float("inf")

            # Extract all X,Y coordinates from the part's G-code
            for line in part_gcode:
                if line.strip().startswith("(") or line.strip() == "":
                    continue

                x_match = re.search(r"X([-+]?\d*\.?\d*)", line)
                y_match = re.search(r"Y([-+]?\d*\.?\d*)", line)

                if x_match and x_match.group(1).strip():
                    x = float(x_match.group(1).strip())
                    if placement.rotation != 0:
                        # Apply rotation to calculate transformed position
                        cx, cy = shape.center()
                        rad = math.radians(placement.rotation)
                        dx, dy = x - cx, 0  # Only considering X for now
                        tx = cx + dx * math.cos(rad) + placement.x_offset
                        min_x = min(min_x, tx)

                if y_match and y_match.group(1).strip():
                    y = float(y_match.group(1).strip())
                    if placement.rotation != 0:
                        # Apply rotation to calculate transformed position
                        cx, cy = shape.center()
                        rad = math.radians(placement.rotation)
                        dx, dy = 0, y - cy  # Only considering Y for now
                        ty = cy + dy * math.cos(rad) + placement.y_offset
                        min_y = min(min_y, ty)

            # If any coordinates would become negative after transformation, adjust offset
            if min_x != float("inf") and min_x < 5:  # 5mm safety margin
                x_adjust = 5 - min_x
                placement.x_offset += x_adjust
                output.append(
                    f"(Adjusting X offset by {x_adjust:.3f} to prevent negative coordinates)"
                )

            if min_y != float("inf") and min_y < 5:  # 5mm safety margin
                y_adjust = 5 - min_y
                placement.y_offset += y_adjust
                output.append(
                    f"(Adjusting Y offset by {y_adjust:.3f} to prevent negative coordinates)"
                )

            # Calculate the bounds for this shape to ensure all points are positive
            # We'll do this by simulating all transformations on the shape's bounds
            shape_min_x, shape_min_y = float("inf"), float("inf")

            # Get the bounding box of the original shape
            bounds = shape.bounds()
            x_vals = [bounds[0], bounds[2]]  # min_x, max_x
            y_vals = [bounds[1], bounds[3]]  # min_y, max_y

            # Check all corners of the bounding box after transformation
            for corner_x in x_vals:
                for corner_y in y_vals:
                    # Apply the same transformation we'll apply to all points
                    if placement.rotation != 0:
                        cx, cy = shape.center()
                        rad = math.radians(placement.rotation)
                        dx, dy = corner_x - cx, corner_y - cy

                        # Rotate
                        tx = cx + dx * math.cos(rad) - dy * math.sin(rad)
                        ty = cy + dx * math.sin(rad) + dy * math.cos(rad)

                        # Translate
                        tx += placement.x_offset
                        ty += placement.y_offset
                    else:
                        # Just translate
                        tx = corner_x + placement.x_offset
                        ty = corner_y + placement.y_offset

                    # Track minimum values
                    shape_min_x = min(shape_min_x, tx)
                    shape_min_y = min(shape_min_y, ty)

            # If any coordinates would be negative, adjust the placement offset
            if shape_min_x < 0 or shape_min_y < 0:
                offset_x = max(0, -shape_min_x) + 5  # Add 5mm safety margin
                offset_y = max(0, -shape_min_y) + 5  # Add 5mm safety margin

                # Update the placement offset for all points in this shape
                placement.x_offset += offset_x
                placement.y_offset += offset_y

                # Log the adjustment
                output.append(
                    f"(Adjusted placement offset by X{offset_x:.3f} Y{offset_y:.3f} to avoid negative coordinates)"
                )

            # If we need to travel to this point, ensure torch is at travel height and torch is off
            if next_point_idx > 0:  # Not the first point
                # Make sure torch is off before travel
                output.append("H0")  # Turn off torch height control
                output.append("M5")  # Turn off torch
                output.append(f"G0 Z{travel_height:.3f}")  # Raise to travel height
                # Now travel to next pierce point
                output.append(
                    f"G0 X{pierce_x:.3f} Y{pierce_y:.3f} F{travel_feedrate:.1f}"
                )
            else:
                # For first shape, just position at travel height
                output.append(f"G0 Z{travel_height:.3f}")
                output.append(
                    f"G0 X{pierce_x:.3f} Y{pierce_y:.3f} F{travel_feedrate:.1f}"
                )

            # Use parser's pierce sequence if available
            original_pierce_idx = shape.original_idx
            if parser and hasattr(parser, "pierce_sequence") and parser.pierce_sequence:
                # Use the inferred pierce sequence from the parser
                for line in parser.pierce_sequence:
                    # Skip position lines as we've already positioned the torch
                    if "G0 X" in line and "Y" in line:
                        continue
                    output.append(line)
            # If no global pierce sequence, try the part-specific one
            elif (
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
                    # We've already positioned the torch at the right location using G0
                    # Now just continue with the pierce sequence (skip the position command)
                    # Find where the M3 command is in the sequence
                    pierce_start_idx = 1  # Default to start at second line
                    for i, line in enumerate(pierce_seq):
                        if "M3" in line:
                            pierce_start_idx = i
                            break

                    # Transform all the commands in the pierce sequence
                    for line in pierce_seq[pierce_start_idx:]:
                        # For G-code moves, we need to update the coordinates
                        if ("G0" in line or "G1" in line) and (
                            "X" in line or "Y" in line
                        ):
                            x_match = re.search(r"X([-+]?\d*\.?\d+)", line)
                            y_match = re.search(r"Y([-+]?\d*\.?\d+)", line)

                            if x_match and y_match:
                                orig_x = float(x_match.group(1))
                                orig_y = float(y_match.group(1))

                                # Apply the same transformation as we did for the pierce point
                                if placement.rotation != 0:
                                    rad = math.radians(placement.rotation)
                                    # Rotate around center of shape
                                    cx, cy = shape.center()
                                    rad = math.radians(placement.rotation)
                                    dx, dy = x - cx, y - cy

                                    # Calculate rotated coordinates (separate steps for clarity)
                                    rx = cx + dx * math.cos(rad) - dy * math.sin(rad)
                                    ry = cy + dx * math.sin(rad) + dy * math.cos(rad)

                                    # Apply offsets
                                    new_x = rx + placement.x_offset
                                    new_y = ry + placement.y_offset
                                else:
                                    new_x = orig_x + placement.x_offset
                                    new_y = orig_y + placement.y_offset

                                # Update the line with new coordinates
                                line = re.sub(r"X[-+]?\d*\.?\d+", f"X{new_x:.3f}", line)
                                line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{new_y:.3f}", line)

                        output.append(line)

                    # Add H1 if it's not already in the sequence
                    if not any("H1" in line for line in pierce_seq):
                        output.append("H1")
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
                                "M3",  # Turn torch on
                                "G4 P0.5",  # Dwell for 0.5 seconds
                                "G0 Z1.6",  # Move to cutting height
                                "H1",  # Enable height control
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
                            "G92 Z0.",  # Set Z zero at current position
                            "G38.2 Z-127. F2540.",  # Probe toward workpiece
                            "G38.4 Z12.7 F508.",  # Back off from workpiece
                            "G92 Z0.",  # Set Z zero at current position
                            "G0 Z1.016",  # Move to initial height
                            "G92 Z0.",  # Set Z zero again
                            "G0 Z3.81",  # Move to pierce height
                            "M3",  # Turn torch on
                            "G4 P0.5",  # Dwell for 0.5 seconds
                            "G0 Z1.6",  # Move to cutting height
                            "H1",  # Enable height control
                        ]
                    )

            # Add transformed cut path
            transformed = GCodeGenerator.transform_gcode(
                shape, placement.x_offset, placement.y_offset, placement.rotation
            )
            output.extend(transformed)

            # End cut with proper torch off sequence and travel height
            # Make sure we have a complete end sequence
            complete_end_sequence = []

            # Use parser's end cut sequence if available, otherwise use the default
            if (
                parser
                and hasattr(parser, "end_cut_sequence")
                and parser.end_cut_sequence
                and len(parser.end_cut_sequence) > 0
            ):
                end_cut_commands = parser.end_cut_sequence
            else:
                end_cut_commands = end_cut_sequence

            # First disable height control if not already in sequence
            if not any("H0" in cmd for cmd in end_cut_commands):
                complete_end_sequence.append("H0")

            # Then turn off torch if not already in sequence
            if not any("M5" in cmd for cmd in end_cut_commands):
                complete_end_sequence.append("M5")

            # Finally raise to travel height if not already in sequence
            if not any("G0 Z" in cmd for cmd in end_cut_commands):
                complete_end_sequence.append(f"G0 Z{travel_height:.3f}")
            else:
                # Update the Z height in existing command
                for cmd in end_cut_commands:
                    if "G0 Z" in cmd:
                        complete_end_sequence.append(f"G0 Z{travel_height:.3f}")
                    else:
                        complete_end_sequence.append(cmd)

            # Add the complete sequence plus a blank line
            output.extend(complete_end_sequence + [""])

        # Make sure torch is off and return to origin at travel height
        output.append("(Returning to origin)")
        output.append("H0")  # Disable height control
        output.append("M5")  # Turn off torch
        output.append(f"G0 Z{travel_height:.3f}")  # Set travel height
        output.append(f"G0 X0 Y0 F{int(travel_feedrate)}.")  # Move to origin
        output.append("")

        # Add footer
        output.extend(footer)
        # Add Langmuir post-processor tag
        output.append("(PS2400)")
        return "\n".join(output)

    @staticmethod
    def generate_simplified_gcode(
        original_gcodes: List[str],
        placements: List[Placement],
        parser=None,  # type: Optional[GCodeParser]
    ) -> str:
        """
        Generate G-code by directly transforming and concatenating original G-code files.

        This simplified approach:
        1) Rotates and moves all X,Y coordinates for each placed part
        2) Preserves all Z coordinates, M codes, and H codes
        3) Concatenates transformed G-code in placement order

        No path optimization is performed - parts are cut in the order they were placed.
        """
        # We'll use the first file to extract header and footer
        lines = original_gcodes[0].strip().split("\n")

        # Find header (everything up to the first M3 command)
        header = []
        footer = []
        in_header = True

        for i, line in enumerate(lines):
            if "M3" in line and i > 0 and in_header:
                in_header = False
                continue
            if in_header:
                header.append(line)
            if "M30" in line:
                footer.append(line)

        # Build new G-code
        output = header + ["", "(Nested parts with simplified G-code)", ""]

        # Process each placement
        for idx, placement in enumerate(placements):
            # Get the original shape's G-code
            shape = placement.shape
            original_idx = shape.original_idx

            if original_idx < 0 or original_idx >= len(original_gcodes):
                print(f"Warning: Invalid original index {original_idx} for shape {idx}")
                continue

            # Get original G-code for this shape
            original_gcode = original_gcodes[original_idx]
            gcode_lines = original_gcode.strip().split("\n")

            # Add part info
            output.append(
                f"(Part {idx + 1} - Original index: {original_idx} - Rotation: {placement.rotation}° - Offset: X{placement.x_offset:.1f} Y{placement.y_offset:.1f})"
            )

            # Find start and end of the actual cutting code
            start_line = 0
            end_line = len(gcode_lines)

            for i, line in enumerate(gcode_lines):
                if "M3" in line and start_line == 0:
                    # Found start of cutting
                    # Back up to include the positioning and IHS sequence
                    for j in range(i - 1, 0, -1):
                        if "G0 X" in gcode_lines[j] and "Y" in gcode_lines[j]:
                            start_line = j
                            break

                if "M5" in line and "H0" in gcode_lines[i - 1] and i > start_line:
                    # Found end of cutting (exclude travel height at end)
                    for j in range(i + 1, len(gcode_lines)):
                        if "G0 Z" in gcode_lines[j]:
                            end_line = j + 1
                            break
                    else:
                        end_line = i + 1
                    break

                if "M30" in line:
                    # End of program
                    end_line = i
                    break

            # Extract the part G-code and transform it
            part_gcode = gcode_lines[start_line:end_line]

            # Track current position for transforming single-coordinate commands
            current_x = 0.0
            current_y = 0.0
            transformed_gcode = []

            for line in part_gcode:
                # Skip comments
                if line.strip().startswith("(") or line.strip() == "":
                    transformed_gcode.append(line)
                    continue

                # Check if this is a Z-only operation command that should not get X/Y coordinates
                # This includes probe commands, Z position sets, dwells, and Z-only moves
                is_probe_command = "G38" in line
                is_z_set_command = "G92 Z" in line
                is_z_only_move = ("G0 Z" in line or "G1 Z" in line) and not any(
                    c in line for c in ["X", "Y"]
                )
                is_dwell_command = "G4" in line
                is_spindle_command = line.strip().startswith(
                    "M3"
                ) or line.strip().startswith("M5")
                is_height_control = line.strip().startswith(
                    "H0"
                ) or line.strip().startswith("H1")
                is_non_motion = line.strip() in [
                    "G17",
                    "G18",
                    "G19",
                    "G20",
                    "G21",
                    "G90",
                    "G91",
                    "G94",
                ]

                # Skip coordinate transformation for any special commands
                # Skip coordinate transformation for any Z-only operations
                # These commands should never have X/Y coordinates added to them
                if (
                    is_probe_command
                    or is_z_set_command
                    or is_z_only_move
                    or is_dwell_command
                    or is_spindle_command
                    or is_height_control
                    or is_non_motion
                ):
                    transformed_gcode.append(line)
                    continue

                # Identify motion commands
                is_g0 = line.strip().startswith("G0 ")
                is_g1 = line.strip().startswith("G1 ")
                is_g2 = line.strip().startswith("G2 ")
                is_g3 = line.strip().startswith("G3 ")
                is_motion = is_g0 or is_g1 or is_g2 or is_g3

                if not is_motion:
                    transformed_gcode.append(line)
                    continue

                # Extract X, Y coordinates and other parameters
                # Use more specific regex to properly capture all valid G-code float formats
                x_match = re.search(r"X([-+]?\d*\.?\d*)", line)
                y_match = re.search(r"Y([-+]?\d*\.?\d*)", line)
                i_match = re.search(r"I([-+]?\d*\.?\d*)", line)
                j_match = re.search(r"J([-+]?\d*\.?\d*)", line)
                r_match = re.search(r"R([-+]?\d*\.?\d*)", line)
                f_match = re.search(r"F(\d*\.?\d*)", line)

                # Get the G-code command (G0, G1, G2, G3)
                command = line.strip().split(" ")[0]

                # Get coordinate values, using current position for missing coordinates
                try:
                    if x_match:
                        x_str = x_match.group(1).strip()
                        if x_str == "" or x_str == "-0" or x_str == "+0":
                            x = 0.0
                        else:
                            x = float(x_str)
                        current_x = x  # Update current position
                    else:
                        x = current_x

                    if y_match:
                        y_str = y_match.group(1).strip()
                        if y_str == "" or y_str == "-0" or y_str == "+0":
                            y = 0.0
                        else:
                            y = float(y_str)
                        current_y = y  # Update current position
                    else:
                        y = current_y

                    # Extract other parameters with careful handling of empty values
                    if i_match:
                        i_str = i_match.group(1).strip()
                        i_val = (
                            0.0
                            if i_str == "" or i_str == "-0" or i_str == "+0"
                            else float(i_str)
                        )
                    else:
                        i_val = 0.0

                    if j_match:
                        j_str = j_match.group(1).strip()
                        j_val = (
                            0.0
                            if j_str == "" or j_str == "-0" or j_str == "+0"
                            else float(j_str)
                        )
                    else:
                        j_val = 0.0

                    if r_match:
                        r_str = r_match.group(1).strip()
                        r_val = (
                            0.0
                            if r_str == "" or r_str == "-0" or r_str == "+0"
                            else float(r_str)
                        )
                    else:
                        r_val = 0.0

                    feed_rate = float(f_match.group(1)) if f_match else 0.0

                    # Track feed rate
                    if f_match:
                        current_feed_rate = feed_rate

                    # Apply the transformation
                    if placement.rotation != 0:
                        # Rotate around center of shape
                        cx, cy = shape.center()
                        rad = math.radians(placement.rotation)
                        dx, dy = x - cx, y - cy

                        # Calculate the rotated coordinates
                        rx = cx + dx * math.cos(rad) - dy * math.sin(rad)
                        ry = cy + dx * math.sin(rad) + dy * math.cos(rad)

                        # Apply offset
                        new_x = rx + placement.x_offset
                        new_y = ry + placement.y_offset

                        # Debug if we're getting weird values
                        if new_x < 0 or new_y < 0:
                            # Increase offset to avoid negative coordinates
                            offset_x = max(0, -new_x + 1)  # +1 mm margin
                            offset_y = max(0, -new_y + 1)  # +1 mm margin

                            # Apply additional offset to all coordinates for this shape
                            new_x += offset_x
                            new_y += offset_y

                            # Update placement offsets for subsequent lines in this part
                            placement.x_offset += offset_x
                            placement.y_offset += offset_y

                        # Check if any point in this path would become negative and adjust
                        if new_x < 0:
                            # For negative X, adjust offset of entire shape
                            x_adjust = -new_x + 5.0  # Add 5mm safety margin
                            placement.x_offset += x_adjust
                            new_x += x_adjust
                            transformed_gcode.append(
                                f"(Adjusted X offset by {x_adjust:.3f}mm to avoid negative values)"
                            )

                        if new_y < 0:
                            # For negative Y, adjust offset of entire shape
                            y_adjust = -new_y + 5.0  # Add 5mm safety margin
                            placement.y_offset += y_adjust
                            new_y += y_adjust
                            transformed_gcode.append(
                                f"(Adjusted Y offset by {y_adjust:.3f}mm to avoid negative values)"
                            )

                        # Rotate I and J for arc commands (G2/G3)
                        if (is_g2 or is_g3) and (i_match or j_match):
                            new_i = i_val * math.cos(rad) - j_val * math.sin(rad)
                            new_j = i_val * math.sin(rad) + j_val * math.cos(rad)
                        else:
                            new_i = i_val
                            new_j = j_val
                    else:
                        # Just translate
                        new_x = x + placement.x_offset
                        new_y = y + placement.y_offset

                        # Handle any negative coordinates
                        if new_x < 0 or new_y < 0:
                            offset_x = max(0, -new_x + 1)  # +1 mm margin
                            offset_y = max(0, -new_y + 1)  # +1 mm margin

                            new_x += offset_x
                            new_y += offset_y

                            # Update placement offsets for subsequent lines in this part
                            placement.x_offset += offset_x
                            placement.y_offset += offset_y

                        new_i = i_val
                        new_j = j_val

                    # Build a new line with consistent coordinate order
                    new_line = command

                    # Add X and Y coordinates - properly handle the coordinates
                    # Instead of forcing negative values to zero, check if both X and Y are negative
                    # If both X and Y are negative, preserve their values but offset the entire path

                    # Detect negative coordinates
                    has_negative = new_x < 0 or new_y < 0

                    # Apply X coordinate
                    if x_match:
                        new_line += f" X{new_x:.3f}"

                    # Apply Y coordinate
                    if y_match:
                        new_line += f" Y{new_y:.3f}"

                    # Add arc parameters if present
                    if is_g2 or is_g3:
                        if i_match or j_match:
                            # Add I/J separately to maintain proper order as in original
                            if i_match:
                                new_line += f" I{new_i:.3f}"
                            if j_match:
                                new_line += f" J{new_j:.3f}"
                        elif r_match:
                            new_line += f" R{abs(r_val):.3f}"

                    # Add feed rate to every G1, G2, and G3 command
                    if is_g1 or is_g2 or is_g3:
                        # Use the latest feed rate we've seen
                        if f_match and feed_rate > 0:
                            new_line += f" F{int(feed_rate)}."
                            current_feed_rate = feed_rate  # Update current feed rate
                        elif current_feed_rate > 0:
                            # If no feed rate in this command but we have a previous one, add it
                            new_line += f" F{int(current_feed_rate)}."

                    # Update the current position
                    current_x = new_x
                    current_y = new_y

                    transformed_gcode.append(new_line)

                except ValueError as e:
                    # If there's a parsing error, keep the original line
                    print(f"Warning: Error transforming line '{line}': {e}")
                    transformed_gcode.append(line)
            # Add the transformed G-code to the output
            output.extend(transformed_gcode)
            output.append("")

        # Return to origin
        output.append("(Returning to origin)")
        output.append("H0")
        output.append("M5")
        output.append(f"G0 Z25.4")
        output.append(f"G0 X0 Y0 F4000.")
        output.append("")

        # Add footer
        output.extend(footer)
        # Add Langmuir post-processor tag
        output.append("(PS2400)")
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
) -> Tuple[str, int]:
    """
    Main function to nest G-code shapes with maximum packing from multiple files.

    Args:
        input_files_with_counts: List of tuples containing (file_content, count_to_place)
        sheet_width: Width of sheet in mm or inches (match G-code units)
        sheet_height: Height of sheet in mm or inches
        spacing: Minimum spacing between parts

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
    nester = NestingAlgorithm(sheet_width, sheet_height, spacing)
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

    Returns:
        Tuple of (nested G-code string, number of parts placed)
    """
    # Forward to new implementation
    return nest_gcode_files(
        [(input_gcode, max_count)], sheet_width, sheet_height, spacing
    )


# Example usage
if __name__ == "__main__":
    # Configure nesting parameters
    SHEET_WIDTH = 24 * 25.4  # mm (adjust to your sheet size)
    SHEET_HEIGHT = 24 * 25.4  # mm
    SPACING = 3  # mm between parts

    # Define up to 6 NC files and the quantities to place
    input_files_with_counts = []

    # File 1 - Required
    file_path_1 = "geodesic_6_bottom.nc"
    count_1 = 5
    with open(file_path_1, "r") as f:
        gcode_1 = f.read()
        input_files_with_counts.append((gcode_1, count_1))

    # File 2 - Optional
    file_path_2 = "geodesic_6_top.nc"
    count_2 = 5
    try:
        with open(file_path_2, "r") as f:
            gcode_2 = f.read()
            input_files_with_counts.append((gcode_2, count_2))
    except FileNotFoundError:
        pass

    # File 3 - Optional
    file_path_3 = "geodesic_5_top.nc"
    count_3 = 0
    try:
        if count_3 > 0:
            with open(file_path_3, "r") as f:
                gcode_3 = f.read()
                input_files_with_counts.append((gcode_3, count_3))
    except FileNotFoundError:
        pass

    # File 4 - Optional
    file_path_4 = "geodesic_5_bottom.nc"
    count_4 = 0
    try:
        if count_4 > 0:
            with open(file_path_4, "r") as f:
                gcode_4 = f.read()
                input_files_with_counts.append((gcode_4, count_4))
    except FileNotFoundError:
        pass

    # File 5 - Optional
    file_path_5 = "geodesic_4_top.nc"
    count_5 = 4
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
