# Comprehensive Codebase Analytics API Documentation

## Overview

This API provides comprehensive codebase analysis using graph-sitter for deep code understanding. It offers advanced features for analyzing code quality, detecting entry points, identifying critical files, and finding code issues with detailed context and fix suggestions.

## New Comprehensive Features

### 🎯 Entry Point Detection
- **Main functions**: `main()`, `__main__`, `run()`, `start()`, `execute()`
- **CLI entry points**: Functions with CLI-related keywords
- **Web endpoints**: FastAPI, Flask, Express routes with decorators
- **Test functions**: Functions starting with `test_` or containing test keywords
- **Script entry points**: Files with `if __name__ == "__main__"`

### 🔍 Critical File Identification
- **Importance scoring**: 0-100 based on multiple factors
- **Dependency analysis**: Files with high incoming/outgoing dependencies
- **Symbol density**: Files defining many functions, classes, variables
- **Complexity analysis**: Files with high cyclomatic complexity
- **Size analysis**: Large files that may need attention

### 🚨 Comprehensive Issue Detection
- **Complexity Issues**: High cyclomatic complexity (>15), long functions (>50 lines)
- **Maintainability Issues**: God classes (>20 methods), deep inheritance (>5 levels)
- **Code Smells**: Too many parameters (>7), empty files
- **Security Vulnerabilities**: `eval()`, `exec()`, shell injection, hardcoded secrets
- **Performance Issues**: Inefficient patterns and anti-patterns

### 🕸️ Dependency Graph Analysis
- **Node types**: Files, functions, classes, modules
- **Centrality scoring**: Identifies most important nodes in the dependency graph
- **Connection analysis**: Most connected and influential components
- **Impact analysis**: Understanding how changes propagate through the codebase

## API Endpoints

### 1. Comprehensive Analysis
**POST** `/comprehensive_analysis`

Performs full codebase analysis including all features.

```json
{
  "repo_url": "owner/repository",
  "include_issues": true,
  "include_entrypoints": true,
  "include_critical_files": true,
  "include_dependency_graph": true,
  "max_issues": 100
}
```

**Response includes:**
- Entry points by type with confidence scores
- Top 10 critical files with importance scores and reasons
- Issues categorized by severity and type with fix suggestions
- Dependency graph with centrality analysis

### 2. Entry Point Detection
**POST** `/detect_entrypoints`

Detects all entry points in the codebase.

```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "total_entrypoints": 15,
  "entrypoints_by_type": {
    "main": [
      {
        "type": "main",
        "file_path": "src/main.py",
        "function_name": "main",
        "line_number": 45,
        "description": "Main function 'main' in main.py",
        "confidence": 0.9,
        "dependencies": ["argparse", "logging", "config"]
      }
    ],
    "web_endpoint": [...],
    "cli": [...],
    "test": [...],
    "script": [...]
  }
}
```

### 3. Critical File Identification
**POST** `/identify_critical_files`

Identifies the most important files in the codebase.

```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "critical_files": [
    {
      "file_path": "src/core/engine.py",
      "importance_score": 87.5,
      "reasons": [
        "High dependency usage (12 files depend on it)",
        "Defines many symbols (25)",
        "High total complexity (156)"
      ],
      "metrics": {
        "functions_count": 15,
        "classes_count": 3,
        "total_complexity": 156,
        "average_complexity": 10.4
      },
      "dependencies_count": 8,
      "dependents_count": 12,
      "lines_of_code": 450
    }
  ]
}
```

### 4. Comprehensive Issue Analysis
**POST** `/analyze_issues`

Performs detailed code issue analysis with context and fix suggestions.

```json
{
  "repo_url": "owner/repository",
  "max_issues": 100
}
```

**Response:**
```json
{
  "analysis_summary": {
    "total_issues": 47,
    "files_with_issues": 12,
    "critical_issues": 3,
    "high_priority_issues": 8
  },
  "issues_by_file": {
    "src/complex_module.py": {
      "issue_count": 5,
      "critical_count": 1,
      "issues": [
        {
          "id": "complexity_a1b2c3d4",
          "type": "complexity_issue",
          "severity": "critical",
          "file_path": "src/complex_module.py",
          "function_name": "process_data",
          "line_number": 125,
          "message": "High cyclomatic complexity: 32",
          "description": "Function 'process_data' has cyclomatic complexity of 32, which exceeds recommended threshold of 15",
          "context": {
            "complexity": 32,
            "complexity_rank": "F",
            "parameters_count": 8,
            "return_statements": 12
          },
          "related_symbols": ["process_data"],
          "affected_functions": [],
          "affected_classes": [],
          "fix_suggestions": [
            "Break down the function into smaller, more focused functions",
            "Extract complex conditional logic into separate methods",
            "Consider using strategy pattern for complex branching logic"
          ]
        }
      ]
    }
  }
}
```

### 5. File-Specific Analysis
**POST** `/analyze_file`

Get detailed analysis for a specific file.

```json
{
  "repo_url": "owner/repository",
  "file_path": "src/main.py"
}
```

**Response includes:**
- Line metrics (LOC, LLOC, SLOC, comments)
- Function analysis with complexity and maintainability scores
- Class analysis with inheritance depth
- Import analysis
- File-specific issues

### 6. Dependency Graph Analysis
**POST** `/dependency_graph`

Build and analyze the complete dependency graph.

```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "graph_statistics": {
    "total_nodes": 245,
    "total_edges": 387,
    "average_connections": 1.58,
    "node_types": {
      "files": 45,
      "functions": 156,
      "classes": 44
    }
  },
  "most_connected_nodes": [
    {
      "name": "utils.py",
      "type": "file",
      "file_path": "src/utils.py",
      "total_connections": 23,
      "dependencies_count": 5,
      "dependents_count": 18
    }
  ],
  "most_central_nodes": [...]
}
```

## Issue Types and Severities

### Issue Types
- **`syntax_error`**: Parsing and syntax issues
- **`type_error`**: Type-related problems
- **`security_vulnerability`**: Security risks and vulnerabilities
- **`code_smell`**: Design and maintainability issues
- **`performance_issue`**: Performance bottlenecks
- **`maintainability_issue`**: Code that's hard to maintain
- **`complexity_issue`**: High complexity functions/classes
- **`dependency_issue`**: Dependency-related problems

### Severity Levels
- **`critical`**: Must fix immediately (security, major bugs)
- **`high`**: Should fix soon (performance, maintainability)
- **`medium`**: Should address (code smells, minor issues)
- **`low`**: Nice to fix (style, minor improvements)
- **`info`**: Informational (suggestions, best practices)

## Advanced Features

### 🔍 Context-Aware Analysis
Each issue includes:
- **Related symbols**: Functions, classes, variables affected
- **Interconnected components**: Dependencies and dependents
- **Impact analysis**: What else might be affected by changes
- **Fix suggestions**: Specific, actionable recommendations

### 📊 Comprehensive Metrics
- **Cyclomatic Complexity**: With letter grades (A-F)
- **Halstead Metrics**: Volume, difficulty, effort
- **Maintainability Index**: 0-100 score with rankings
- **Depth of Inheritance**: Class hierarchy analysis
- **Lines of Code**: LOC, LLOC, SLOC, comment density

### 🎯 Smart Detection Patterns
- **Entry Points**: Pattern matching for common entry point signatures
- **Security Issues**: Regex patterns for common vulnerabilities
- **Code Smells**: Heuristic-based detection of anti-patterns
- **Critical Files**: Multi-factor scoring algorithm

## Usage Examples

### Example 1: Full Repository Analysis
```bash
curl -X POST "https://your-api-url/comprehensive_analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "facebook/react",
    "include_issues": true,
    "include_entrypoints": true,
    "include_critical_files": true,
    "include_dependency_graph": true,
    "max_issues": 50
  }'
```

### Example 2: Security-Focused Analysis
```bash
curl -X POST "https://your-api-url/analyze_issues" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "your-org/your-repo",
    "max_issues": 200
  }'
```

### Example 3: Architecture Analysis
```bash
curl -X POST "https://your-api-url/dependency_graph" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "your-org/your-repo"
  }'
```

## Integration Tips

### 1. CI/CD Integration
- Use `/analyze_issues` in pull request checks
- Set thresholds for critical and high-severity issues
- Block deployments with security vulnerabilities

### 2. Code Review Enhancement
- Use `/analyze_file` for detailed file reviews
- Check `/identify_critical_files` before major refactoring
- Use dependency graph for impact analysis

### 3. Technical Debt Management
- Regular `/comprehensive_analysis` for health monitoring
- Track maintainability index trends over time
- Prioritize fixes based on file importance scores

## Performance Considerations

- **Large repositories**: Use `max_issues` parameter to limit analysis scope
- **Incremental analysis**: Analyze specific files with `/analyze_file`
- **Caching**: Results are suitable for caching based on commit hash
- **Parallel processing**: Multiple endpoints can be called concurrently

## Error Handling

All endpoints return structured error responses:

```json
{
  "detail": "Analysis failed: Repository not found or access denied"
}
```

Common error scenarios:
- Repository access issues (404, 403)
- Analysis timeouts for very large repositories
- Graph-sitter parsing errors for unsupported languages
- Memory limits for extremely complex codebases

## Supported Languages

Currently optimized for:
- **Python** (full support)
- **JavaScript/TypeScript** (partial support)
- **Java** (basic support)
- **C/C++** (basic support)

Additional languages can be added by extending the graph-sitter language parsers.
