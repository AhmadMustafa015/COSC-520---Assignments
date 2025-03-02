# Tree Data Structures Benchmark and Analysis

This project benchmarks and compares different tree data structures including Binary Search Trees (BST), AVL Trees, and B-Trees. The code analyzes performance characteristics for insertion, search, and deletion operations across various dataset sizes.

## Prerequisites

- Python 3.6 or higher
- Required packages:
  - matplotlib
  - numpy
  - bintrees

## Installation

Install the required dependencies:

```bash
pip install matplotlib numpy bintrees
```

## Running the Code

### Full Benchmark

To run the complete benchmark that tests all tree implementations with different dataset sizes:

```bash
python assignment_2_main.py
```

This will:
- Generate random datasets of different sizes
- Run benchmarks for insert, search, and delete operations
- Generate performance comparison plots
- Save the datasets used in a CSV file

### Running Unit Tests

To verify the correct functionality of the tree implementations:

```bash
python assignment_2_main.py -t
```

## Output Files

The script generates several output files:

- `insert_benchmark.png`: Graph showing performance comparison for insertion operations
- `search_benchmark.png`: Graph showing performance comparison for search operations
- `delete_benchmark.png`: Graph showing performance comparison for deletion operations
- `benchmark_datasets.csv`: CSV file containing the datasets used in the benchmarks

## Customizing the Experiment

You can modify the following parameters in the `run_experiment()` function:

- Dataset sizes: Change the `dataset_sizes` list to test with different data volumes
- B-Tree order: Modify the `data_structures` dictionary to change the B-Tree order parameter

## Notes

- The full benchmark may take several minutes to complete, especially with larger dataset sizes
- The plots use logarithmic scaling for better visualization of growth patterns
- All tree implementations have been tested for correctness using unit tests
