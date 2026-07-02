# Differential Evolution Optimization for CNC Shape Nesting

This module implements a differential evolution (DE) optimization algorithm for efficiently packing CNC cut shapes onto sheet material. It supports continuous rotation angles (in configurable increments, default 15°) and provides comprehensive tracking of optimization metrics.

## Features

- **Differential Evolution Optimization**: Uses scipy's DE algorithm to find optimal shape placements
- **Continuous Rotation**: Supports rotation in configurable increments (default 15°)
- **Real-time Progress Tracking**: Monitor utilization and shape placement during optimization
- **Metrics History**: Track performance across generations
- **Visualization Tools**: Generate plots of nesting results and optimization progress
- **G-code Integration**: Direct processing of G-code files with optimized output
- **Method Comparison**: Compare DE optimization with traditional grid-based approaches

## Installation

Required dependencies:
```bash
pip install numpy scipy matplotlib shapely
```

## File Structure

```
de_optimization/
├── de_optimizer.py           # Core DE optimization engine
├── enhanced_nesting.py       # Extended nesting algorithm with DE support
├── de_nest_gcode.py         # Main integration for G-code processing
├── demo_de_nesting.py       # Demonstration and examples
└── README.md                # This file
```

## Quick Start

### Basic Usage

```python
from enhanced_nesting import EnhancedNestingAlgorithm
from shappack import Shape, Point

# Create shapes
shapes = [...]  # Your Shape objects

# Initialize nesting algorithm
nesting = EnhancedNestingAlgorithm(
    sheet_width=300.0,
    sheet_height=200.0,
    spacing=5.0
)

# Run DE optimization
placements, metrics = nesting.nest_shapes_differential_evolution(
    shapes=shapes,
    max_count=10,           # Max copies per shape
    generations=100,        # Number of DE generations
    population_size=50,     # DE population size
    rotation_increment=15.0 # Rotation angle increment
)

# Visualize results
nesting.visualize_nesting_with_stats(placements, metrics)
nesting.visualize_optimization_progress(metrics)
```

### G-code Processing

```python
from de_nest_gcode import nest_gcode_files_de

# Process G-code files
placements, metrics = nest_gcode_files_de(
    gcode_files=["part1.nc", "part2.nc"],
    output_file="nested_output.nc",
    sheet_width=600.0,
    sheet_height=400.0,
    generations=100,
    rotation_increment=15.0
)
```

### Command Line Interface

```bash
# Basic nesting
python de_nest_gcode.py input1.nc input2.nc -o output.nc -w 600 -h 400

# With custom parameters
python de_nest_gcode.py input.nc -o output.nc -w 600 -h 400 \
    --generations 150 \
    --population 75 \
    --rotation-increment 15 \
    --mutation 0.5 1.2 \
    --recombination 0.8

# Compare with grid-based method
python de_nest_gcode.py input.nc -o output.nc -w 600 -h 400 --compare
```

## Key Parameters

### DE Optimization Parameters

- **generations**: Number of evolution generations (default: 100)
- **population_size**: Size of the population per generation (default: 50)
- **mutation**: Mutation factor (F), can be tuple (min, max) (default: (0.5, 1.0))
- **recombination**: Crossover probability (CR) (default: 0.7)
- **strategy**: DE strategy (default: "best1bin")
- **rotation_increment**: Angle increment in degrees (default: 15.0)

### Sheet Parameters

- **sheet_width**: Width of the sheet material in mm
- **sheet_height**: Height of the sheet material in mm
- **spacing**: Minimum spacing between shapes in mm (default: 5.0)
- **max_count**: Maximum copies of each shape to place (default: 100)

## How It Works

### 1. Vector Encoding

Each potential shape placement is encoded as 3 values:
- X position (0 to sheet_width)
- Y position (0 to sheet_height)  
- Rotation index (index into valid rotation angles)

For N shapes with M max copies each, the solution vector has length: N × M × 3

### 2. Objective Function

The algorithm minimizes negative sheet utilization (to maximize utilization):
```
utilization = (total_area_of_placed_shapes) / (sheet_area)
objective = -utilization
```

### 3. Constraints

- Shapes must fit entirely within sheet boundaries
- No overlapping between shapes (with spacing buffer)
- Valid rotation angles based on increment (e.g., 0°, 15°, 30°, ...)

### 4. Evolution Process

1. Initialize random population of solution vectors
2. For each generation:
   - Evaluate fitness (utilization) of each solution
   - Create new solutions via mutation and crossover
   - Select best solutions for next generation
3. Track best solution and metrics across generations

## Metrics Tracking

The system tracks the following metrics per generation:
- **best_utilization**: Best sheet utilization achieved
- **shapes_placed**: Number of shapes successfully placed
- **total_shapes_attempted**: Maximum possible shapes
- **computation_time**: Cumulative time elapsed
- **best_vector**: Current best solution vector

## Visualization Options

### 1. Nesting Visualization
Shows the final placement of shapes with:
- Color-coding by shape type
- Rotation angle indicators
- Utilization statistics
- Shape count summary

### 2. Progress Plot
Displays optimization progress:
- Utilization over generations
- Shapes placed over time
- Computation time tracking

### 3. Method Comparison
Side-by-side comparison of:
- Grid-based nesting
- DE-optimized nesting
- Performance metrics

## Examples

Run the demonstration script to see various features:

```bash
python demo_de_nesting.py
```

This will run 5 demonstrations:
1. Basic DE optimization
2. Parameter sensitivity analysis
3. Rotation increment comparison
4. DE vs Grid method comparison
5. G-code integration example

Output files will be saved in `de_demo_output/` directory.

## Advanced Usage

### Custom Objective Function

```python
def custom_objective(vector):
    # Decode vector into placements
    placements = decode_vector(vector)
    
    # Calculate custom metric
    score = calculate_custom_score(placements)
    
    # Return negative for minimization
    return -score

# Use with optimizer
optimizer = DEOptimizer(sheet_width, sheet_height)
optimizer.create_objective_function = lambda shapes, max_copies: custom_objective
```

### Parallel Processing

Enable parallel evaluation for faster optimization:

```python
placements, metrics = nesting.nest_shapes_differential_evolution(
    shapes=shapes,
    workers=-1  # Use all available CPU cores
)
```

### Reproducible Results

Set a random seed for consistent results:

```python
placements, metrics = nesting.nest_shapes_differential_evolution(
    shapes=shapes,
    seed=42  # Fixed seed for reproducibility
)
```

## Performance Tips

1. **Start Small**: Begin with fewer generations and smaller populations to test
2. **Rotation Increment**: Balance between precision (small increment) and speed (large increment)
3. **Population Size**: Larger populations explore more solutions but take longer
4. **Mutation Factor**: Higher values increase exploration, lower values refine solutions
5. **Early Stopping**: Monitor metrics and stop if utilization plateaus

## Troubleshooting

### Slow Optimization
- Reduce population size or generations
- Increase rotation increment (e.g., 30° instead of 15°)
- Use parallel workers (`workers=-1`)

### Poor Utilization
- Increase generations for more exploration
- Adjust mutation factor range
- Try different DE strategies
- Decrease rotation increment for finer control

### Memory Issues
- Reduce max_count parameter
- Process shapes in batches
- Lower population size

## License

This module extends the original shappack.py functionality. Refer to the main project license for terms of use.

## Contributing

To contribute improvements:
1. Test changes with demo_de_nesting.py
2. Ensure backward compatibility with shappack.py
3. Document new parameters and features
4. Include visualization examples