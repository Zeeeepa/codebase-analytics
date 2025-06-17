# Codebase Analytics Backend

This directory contains the backend code for the Codebase Analytics tool, which provides comprehensive analysis of codebases.

## Files

- `analysis.py`: Core analysis functionality
- `visualize.py`: Visualization methods
- `run_analysis.py`: Script to run analysis on a GitHub repository
- `simple_analysis.py`: Simplified analysis script
- `run_full_analysis.py`: Comprehensive analysis script

## Usage

### Basic Analysis

```bash
python run_analysis.py https://github.com/username/repo --output-dir ./output
```

### Simple Analysis

```bash
python simple_analysis.py https://github.com/username/repo --output-dir ./output
```

### Full Comprehensive Analysis

```bash
python run_full_analysis.py https://github.com/username/repo --output-dir ./output
```

## Output

The analysis scripts generate the following output:

- `analysis.json`: JSON file containing the analysis results
- `report.html`: HTML report with visualizations and analysis results
- `visualizations/`: Directory containing visualizations
  - `dependency_graph.png`: Visualization of the dependency graph
  - `complexity.png`: Visualization of code complexity
  - `issues.png`: Visualization of issues by severity

## Features

- **Code Complexity Analysis**: Analyzes the cyclomatic complexity of code
- **Dependency Analysis**: Identifies dependencies between files
- **Circular Dependency Detection**: Detects circular dependencies
- **Issue Detection**: Identifies potential issues in the code
- **Visualization**: Generates visualizations of the analysis results
- **HTML Report**: Generates a comprehensive HTML report

## Requirements

- Python 3.6+
- NetworkX
- Matplotlib
- NumPy

