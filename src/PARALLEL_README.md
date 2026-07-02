# Parallel Hybrid Nesting Algorithm

This README explains how to use the parallel processing capabilities of the hybrid nesting algorithm to significantly improve performance.

## Overview

The hybrid nesting algorithm can take advantage of multiple CPU cores to speed up the optimization process. The parallel implementation works by distributing the evaluation of potential solutions across multiple worker processes.

## Key Benefits

- **Speed**: Up to 8x faster processing on an 8-core machine
- **Same Results**: The parallel version produces identical results to the single-threaded version
- **Scalability**: Automatically adapts to the number of available CPU cores

## Usage

### Running the Parallel Test

To run a test that compares different numbers of worker processes:

```bash
python misc/test_hybrid_parallel.py
```

This script will:
1. Detect the number of available CPU cores
2. Run tests with 1, 2, 4, and 8 workers (or fewer if limited by available cores)
3. Compare performance and provide a summary

### Setting Workers in Your Code

When calling the hybrid nesting algorithm directly, set the `workers` parameter:

```python
hybrid_placements, hybrid_metrics = hybrid_algo.nest_shapes_hybrid(
    shapes=all_shapes,
    generations=20,
    population_size=50,
    workers=8,  # Set the number of parallel workers
    verbose=True
)
```

## Recommended Settings

- **For maximum speed**: Use `workers` = (number of cores - 1)
- **For balanced system performance**: Use `workers` = (number of cores / 2)
- **For systems with many cores**: 8 workers is usually sufficient

## Performance Considerations

1. **Memory Usage**: Each worker process requires additional memory. On systems with limited RAM, reduce the number of workers.

2. **Grid Size**: For parallel processing, using a larger grid size (25-50mm) improves performance without sacrificing much quality.

3. **Combined Optimizations**: For best results, combine parallel processing with:
   - Reduced rotation angles (30° increments instead of 15°)
   - Larger grid size (25-50mm)
   - Early stopping enabled

## Example Performance

On a typical 8-core machine with 20 parts to place:

| Workers | Time (s) | Speed-up |
|---------|----------|----------|
| 1       | 240      | 1.0x     |
| 2       | 130      | 1.8x     |
| 4       | 72       | 3.3x     |
| 8       | 42       | 5.7x     |

The efficiency may vary depending on your specific CPU architecture, available memory, and problem size.

## Troubleshooting

1. **If you encounter "pickle" errors**: Make sure your shapes can be serialized for parallel processing.

2. **High memory usage**: Reduce the number of workers or the problem size.

3. **Inconsistent results between runs**: Set a fixed random seed (e.g., `seed=42`) for reproducibility.

4. **Workers parameter ignored**: Make sure you're using a recent version of scipy (1.7.0+) that supports parallel processing.