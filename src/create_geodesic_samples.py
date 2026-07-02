#!/usr/bin/env python3
"""
Create sample geodesic NC files for testing the DE optimization.
These are simplified representations of geodesic dome parts.
"""


def create_geodesic_6_bottom():
    """Create a hexagonal bottom piece."""
    return """G21
G90
F1000

G0 X50 Y50
G31
M3
H1
G1 X100 Y50 F1000
G1 X125 Y93.3
G1 X100 Y136.6
G1 X50 Y136.6
G1 X25 Y93.3
G1 X50 Y50
M5

M30
"""


def create_geodesic_6_top():
    """Create a smaller hexagonal top piece with different orientation."""
    return """G21
G90
F1000

G0 X70 Y45
G31
M3
H1
G1 X95 Y45 F1000
G1 X107.5 Y66.65
G1 X95 Y88.3
G1 X70 Y88.3
G1 X57.5 Y66.65
G1 X70 Y45
M5

M30
"""


def create_geodesic_5_top():
    """Create a pentagonal top piece."""
    return """G21
G90
F1000

G0 X75 Y50
G31
M3
H1
G1 X115 Y68 F1000
G1 X100 Y108
G1 X50 Y108
G1 X35 Y68
G1 X75 Y50
M5

M30
"""


def create_geodesic_5_bottom():
    """Create a larger pentagonal bottom piece."""
    return """G21
G90
F1000

G0 X80 Y35
G31
M3
H1
G1 X135 Y60 F1000
G1 X120 Y120
G1 X40 Y120
G1 X25 Y60
G1 X80 Y35
M5

M30
"""


def create_geodesic_4_top():
    """Create a triangular top piece (despite the name)."""
    return """G21
G90
F1000

G0 X75 Y40
G31
M3
H1
G1 X120 Y110 F1000
G1 X30 Y110
G1 X75 Y40
M5

M30
"""


def create_geodesic_4_bottom():
    """Create a rectangular bottom piece."""
    return """G21
G90
F1000

G0 X35 Y25
G31
M3
H1
G1 X115 Y25 F1000
G1 X115 Y85
G1 X35 Y85
G1 X35 Y25
M5

M30
"""


def main():
    """Create all sample geodesic NC files."""

    files = [
        ("geodesic_6_bottom.nc", create_geodesic_6_bottom()),
        ("geodesic_6_top.nc", create_geodesic_6_top()),
        ("geodesic_5_top.nc", create_geodesic_5_top()),
        ("geodesic_5_bottom.nc", create_geodesic_5_bottom()),
        ("geodesic_4_top.nc", create_geodesic_4_top()),
        ("geodesic_4_bottom.nc", create_geodesic_4_bottom()),
    ]

    print("Creating sample geodesic NC files...")
    print("-" * 40)

    for filename, content in files:
        with open(filename, "w") as f:
            f.write(content)
        print(f"✓ Created: {filename}")

    print("\nAll sample files created successfully!")
    print("\nYou can now run:")
    print("  python run_geodesic_de.py")
    print("\nTo optimize the nesting of these shapes.")


if __name__ == "__main__":
    main()
