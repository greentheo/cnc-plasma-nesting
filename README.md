# cnc-plasma-nesting

2D nesting / bin-packing for CNC plasma cutting. The tools parse cut shapes out of
G-code (`.nc`) files, pack them onto a sheet to minimize wasted material, and emit
new G-code panels for the cutter — along with matplotlib visualizations of each layout.

Shapes are represented as polygons and collision-checked with [Shapely](https://shapely.readthedocs.io/),
so parts can be packed tightly and rotated without overlapping.

## Layout

```
src/        all source: core library, algorithm variants, runners, and tests
samples/    example input G-code (geodesic dome panels) and DXF sketches
```

The scripts import each other with flat imports (`from shappack import ...`) and read
input files relative to the current directory, so run them from inside `src/` with the
sample inputs available (see Quick start).

## Algorithms

The core lives in `shappack.py` (G-code parsing, sheet layout, greedy bottom-left
placement, and G-code generation). Several optimization strategies build on it:

- `shappack.py` — core parser/generator plus the baseline greedy nesting algorithm.
- `shappack_improved.py` — greedy variant with better rotation handling.
- `shappack_de.py` — differential-evolution placement optimizer (SciPy).
- `shappack_hybrid.py` — hybrid: differential evolution seeded with greedy bottom-left placement.
- `shappack_hybrid_fast.py` / `shappack_hybrid_parallel.py` — speed-tuned and multi-core versions of the hybrid approach.
- `shappack_multi_objective.py` — multi-objective hybrid (e.g. material use vs. cut path).
- `improved_nesting/shappack_continuous.py` — continuous-placement experiment.
- `de_optimization/` — standalone differential-evolution nesting package (see its own README).
- `gcode_visualizer/` — browser-based G-code viewer (`index.html`).

`compare_nesting_algorithms.py` benchmarks the greedy, hybrid, and multi-objective
approaches against each other. `PARALLEL_README.md` documents the multi-core options.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
# make the sample inputs available to the scripts
cp ../samples/geodesic_*.nc .

# run the greedy baseline
python shappack.py

# compare algorithms
python compare_nesting_algorithms.py
```

## Requirements

Python 3 with numpy, scipy, matplotlib, and shapely (see `requirements.txt`).

## Notes

Generated outputs (`*_output*.nc`, result/progress `*.png`) are git-ignored since the
scripts regenerate them. This project was extracted from a larger CNC scratch folder;
the sample geodesic `.nc` files are the inputs the scripts reference by default.
