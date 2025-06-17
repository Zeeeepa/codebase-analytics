# Codebase Analytics Backend

A comprehensive codebase analysis tool that provides deep insights into code quality, dependencies, and potential issues.

## Overview

This backend provides a powerful set of tools for analyzing codebases, detecting issues, and generating visualizations. It combines functionality from multiple analysis modules into a single, cohesive system with an API for easy integration.

## Features

### Analysis Capabilities

- **Dependency Analysis**: Identify dependencies between files, detect circular dependencies, and find critical dependencies.
- **Call Graph Analysis**: Analyze function calls, identify entry points, leaf functions, and calculate call depth.
- **Code Quality Metrics**: Calculate maintainability index, cyclomatic complexity, Halstead volume, and more.
- **Issue Detection**: Detect various issues such as:
  - Implementation errors
  - Misspelled function names
  - Null references
  - Unsafe assertions
  - Improper exception handling
  - Incomplete implementations
  - Inefficient patterns
  - Code duplication
  - Unused parameters
  - Redundant code
  - Formatting issues
  - Suboptimal defaults
  - Wrong parameters
  - Runtime errors
  - Dead code
  - Security vulnerabilities
  - Performance issues

### Visualization Capabilities

- **Dependency Graph**: Visualize dependencies between files.
- **Call Graph**: Visualize function calls.
- **Issue Visualization**: Visualize issues by severity and category.
- **Code Quality Visualization**: Visualize code quality metrics.
- **Repository Structure Visualization**: Visualize the repository structure.
- **HTML Reports**: Generate comprehensive HTML reports with all visualizations.

## Architecture

The backend consists of three main modules:

1. **analysis.py**: Core analysis functionality for analyzing codebases, detecting issues, and calculating metrics.
2. **visualize.py**: Visualization functionality for generating visualizations of analysis results.
3. **api.py**: FastAPI server for exposing analysis and visualization functionality via a REST API.

## API Endpoints

- `GET /`: Root endpoint
- `POST /api/analyze`: Analyze a codebase from a Git repository
- `GET /api/analysis/{analysis_id}/status`: Get the status of an analysis task
- `GET /api/analysis/{analysis_id}/result`: Get the result of an analysis task
- `GET /api/analysis/{analysis_id}/report`: Get the HTML report for an analysis task
- `GET /api/visualizations/{analysis_id}/{visualization_file}`: Get a visualization file for an analysis task
- `POST /api/upload`: Upload and analyze a codebase (zip or tar.gz)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API Server

```bash
cd backend
python api.py
```

The server will start on http://localhost:8000.

### Using the API

#### Analyzing a Git Repository

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/username/repo", "branch": "main", "include_visualizations": true}'
```

#### Checking Analysis Status

```bash
curl -X GET "http://localhost:8000/api/analysis/{analysis_id}/status"
```

#### Getting Analysis Results

```bash
curl -X GET "http://localhost:8000/api/analysis/{analysis_id}/result"
```

#### Viewing the HTML Report

Open the following URL in a browser:
```
http://localhost:8000/api/analysis/{analysis_id}/report
```

## Dependencies

- FastAPI: Web framework for building APIs
- Uvicorn: ASGI server for running FastAPI
- NetworkX: Graph library for dependency and call graph analysis
- Matplotlib: Visualization library
- Codegen SDK: SDK for code analysis

## License

MIT

