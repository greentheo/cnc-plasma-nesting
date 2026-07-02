(Sample G-code file for testing the visualizer)
(Simple rectangular pattern with pierce points)

G21 (Use mm units)
G90 (Use absolute coordinates)

(Part 1 - Square)
G0 X10 Y10 (Rapid to starting position)
G92 Z0 (Set Z zero)
G38.2 Z-127 F2540 (Pierce sequence start)
G38.4 Z12.7 F508
G92 Z0
G0 Z1.016
G92 Z0
G0 Z3.81
M3 (Torch on)
G4 P0.5
G0 Z1.6
H1 (Torch on)
G1 X10 Y60 F1000 (Cut up)
G1 X60 Y60 (Cut right)
G1 X60 Y10 (Cut down)
G1 X10 Y10 (Cut left to close shape)
H0 (Torch off)
M5 (Torch off)
G0 Z25.4 (Raise to travel height)

(Part 2 - Triangle)
G0 X100 Y10 (Rapid to next starting position)
G92 Z0
G38.2 Z-127 F2540
G38.4 Z12.7 F508
G92 Z0
G0 Z1.016
G92 Z0
G0 Z3.81
M3 (Torch on)
G4 P0.5
G0 Z1.6
H1 (Torch on)
G1 X150 Y10 F1000 (Cut right)
G1 X125 Y60 (Cut up and left)
G1 X100 Y10 (Cut down and left to close shape)
H0 (Torch off)
M5 (Torch off)
G0 Z25.4 (Raise to travel height)

(Part 3 - Circle using G2/G3)
G0 X200 Y35 (Rapid to next starting position)
G92 Z0
G38.2 Z-127 F2540
G38.4 Z12.7 F508
G92 Z0
G0 Z1.016
G92 Z0
G0 Z3.81
M3 (Torch on)
G4 P0.5
G0 Z1.6
H1 (Torch on)
G3 X200 Y35 I0 J25 F800 (Full counterclockwise circle with 25mm radius)
H0 (Torch off)
M5 (Torch off)
G0 Z25.4 (Raise to travel height)

(Return to origin)
G0 X0 Y0

M30 (End program)
