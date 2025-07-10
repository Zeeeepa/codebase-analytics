# Comprehensive Codebase Analytics API Documentation

This API provides comprehensive codebase analysis using the graph-sitter/codegen SDK, offering detailed insights into code quality, complexity, architecture, and maintainability.

## Overview

The API has been significantly enhanced with a new `Analysis` class that wraps the graph-sitter Codebase functionality to provide comprehensive analysis capabilities across multiple dimensions:

- **Code Quality Analysis**: Cyclomatic complexity, maintainability index, Halstead metrics
- **Architecture Analysis**: Design patterns, coupling/cohesion, dependency analysis
- **Security Analysis**: Vulnerability detection, security pattern analysis
- **Performance Analysis**: Performance bottleneck identification, optimization suggestions
- **Technical Debt Analysis**: Code smells, refactoring suggestions
- **Test Coverage Analysis**: Test-to-code ratios, coverage estimation

## API Endpoints

### 1. Comprehensive Analysis
**POST** `/comprehensive_analysis`

Get comprehensive codebase analysis with configurable analysis types.

**Request Body:**
```json
{
  "repo_url": "owner/repository",
  "analysis_type": "comprehensive"  // Options: comprehensive, quality, complexity, dependencies, architecture, patterns, debt, security, performance, testing
}
```

**Response:**
```json
{
  "overview": {
    "total_files": 150,
    "total_functions": 500,
    "total_classes": 75,
    "total_imports": 200,
    "total_lines_of_code": 15000,
    "language_distribution": {"python": 120, "javascript": 30},
    "average_file_size": 100,
    "largest_file_size": 500,
    "smallest_file_size": 10
  },
  "quality_metrics": {
    "function_metrics": [...],
    "class_metrics": [...],
    "averages": {
      "cyclomatic_complexity": 3.2,
      "maintainability_index": 75.5,
      "depth_of_inheritance": 1.8
    },
    "quality_distribution": {"A": 45, "B": 30, "C": 20, "D": 5}
  },
  "complexity_analysis": {...},
  "dependency_analysis": {...},
  "architecture_analysis": {...},
  "code_patterns": {...},
  "technical_debt": {...},
  "test_coverage_analysis": {...},
  "security_analysis": {...},
  "performance_insights": {...},
  "repository": {
    "url": "owner/repository",
    "description": "Repository description",
    "analysis_timestamp": "2024-01-01T12:00:00",
    "analysis_type": "comprehensive"
  }
}
```

### 2. Function Analysis
**POST** `/function_analysis`

Get detailed function-level analysis including complexity metrics and properties.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repository": "owner/repository",
  "total_functions": 500,
  "functions": [
    {
      "name": "process_data",
      "file": "src/processor.py",
      "line_number": 45,
      "metrics": {
        "cyclomatic_complexity": 8,
        "complexity_rank": "B",
        "maintainability_index": 65,
        "maintainability_rank": "B",
        "lines_of_code": 25,
        "halstead_volume": 120.5,
        "halstead_metrics": {
          "operators_count": 15,
          "operands_count": 20,
          "unique_operators": 8,
          "unique_operands": 12
        }
      },
      "properties": {
        "is_async": false,
        "is_generator": false,
        "parameter_count": 3,
        "has_docstring": true,
        "decorators": ["@staticmethod"]
      }
    }
  ],
  "summary": {
    "average_complexity": 4.2,
    "average_maintainability": 72.3,
    "high_complexity_count": 15,
    "low_maintainability_count": 8
  }
}
```

### 3. Class Analysis
**POST** `/class_analysis`

Get detailed class-level analysis including inheritance metrics and method analysis.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repository": "owner/repository",
  "total_classes": 75,
  "classes": [
    {
      "name": "DataProcessor",
      "file": "src/processor.py",
      "line_number": 10,
      "metrics": {
        "depth_of_inheritance": 2,
        "method_count": 8,
        "attribute_count": 5,
        "total_complexity": 45,
        "average_method_complexity": 5.6
      },
      "properties": {
        "is_abstract": false,
        "superclasses": ["BaseProcessor"],
        "has_docstring": true,
        "decorators": ["@dataclass"]
      },
      "methods": [
        {
          "name": "process",
          "is_async": false,
          "parameter_count": 2,
          "complexity": 6
        }
      ]
    }
  ],
  "summary": {
    "average_doi": 1.8,
    "average_methods_per_class": 6.7,
    "complex_classes_count": 5,
    "abstract_classes_count": 3
  }
}
```

### 4. Dependency Graph Analysis
**POST** `/dependency_graph`

Get dependency graph analysis including import patterns and external dependencies.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repository": "owner/repository",
  "dependency_graph": {
    "src/main.py": {
      "imports": [
        {
          "module": "requests",
          "is_external": true,
          "import_type": "module"
        }
      ],
      "import_count": 5,
      "external_import_count": 3
    }
  },
  "external_dependencies": ["requests", "numpy", "pandas"],
  "metrics": {
    "total_files": 150,
    "total_imports": 200,
    "total_external_dependencies": 15,
    "average_imports_per_file": 1.33,
    "external_dependency_ratio": 0.6
  }
}
```

### 5. Code Quality Report
**POST** `/code_quality_report`

Generate comprehensive code quality report with overall scoring and recommendations.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repository": "owner/repository",
  "overall_quality_score": 78.5,
  "quality_grade": "B",
  "quality_factors": {
    "complexity": 85.0,
    "maintainability": 75.0,
    "technical_debt": 70.0,
    "security": 84.0
  },
  "overview": {...},
  "detailed_metrics": {
    "quality": {...},
    "complexity": {...},
    "technical_debt": {...},
    "security": {...}
  },
  "recommendations": {
    "high_priority": [],
    "medium_priority": ["Consider breaking down long functions"],
    "low_priority": ["Use parameterized queries to prevent SQL injection"]
  },
  "report_timestamp": "2024-01-01T12:00:00"
}
```

### 6. Symbol Analysis
**POST** `/symbol_analysis`

Get detailed symbol analysis including usage patterns and symbol distribution.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repository": "owner/repository",
  "total_symbols": 1000,
  "symbols": [
    {
      "name": "process_data",
      "type": "function",
      "file": "src/processor.py",
      "line_number": 45,
      "usage_count": 15,
      "is_exported": true,
      "is_imported": false,
      "parameter_count": 3,
      "is_async": false
    }
  ],
  "usage_analysis": {
    "unused_symbols": [...],
    "heavily_used_symbols": [...],
    "average_usage": 5.2
  },
  "symbol_distribution": {
    "function": 500,
    "class": 75,
    "variable": 425
  }
}
```

### 7. Architecture Insights
**POST** `/architecture_insights`

Get architectural insights including design patterns and architectural smells.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repository": "owner/repository",
  "architecture_analysis": {
    "package_structure": {...},
    "design_patterns": [...],
    "coupling_metrics": {...},
    "architectural_violations": [...]
  },
  "code_patterns": {
    "detected_patterns": {
      "singleton_classes": [],
      "factory_methods": ["create_processor"],
      "async_functions": ["async_process"],
      "generator_functions": ["data_generator"]
    },
    "pattern_statistics": {...}
  },
  "dependency_metrics": {...},
  "module_metrics": {...},
  "architectural_smells": [
    {
      "type": "Large Modules",
      "description": "Modules with too many symbols",
      "affected_files": ["src/large_module.py"]
    }
  ],
  "recommendations": {
    "modularity": "Consider breaking down large modules",
    "coupling": "Reduce dependencies between modules",
    "cohesion": "Group related functionality together"
  }
}
```

### 8. Legacy Repository Analysis (Backward Compatible)
**POST** `/analyze_repo`

Enhanced version of the original endpoint with backward compatibility.

**Request Body:**
```json
{
  "repo_url": "owner/repository"
}
```

**Response:**
```json
{
  "repo_url": "owner/repository",
  "description": "Repository description",
  "monthly_commits": {...},
  "line_metrics": {
    "total": {
      "loc": 15000,
      "lloc": 12000,
      "sloc": 13000,
      "comments": 2000,
      "comment_density": 13.33
    }
  },
  "cyclomatic_complexity": {
    "average": 4.2
  },
  "depth_of_inheritance": {
    "average": 1.8
  },
  "halstead_metrics": {
    "total_volume": 50000,
    "average_volume": 100
  },
  "maintainability_index": {
    "average": 75
  },
  "num_files": 150,
  "num_functions": 500,
  "num_classes": 75,
  "enhanced_analysis": {
    "overview": {...},
    "quality_distribution": {...},
    "language_distribution": {...},
    "file_size_stats": {...}
  }
}
```

### 9. Health Check
**GET** `/health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00"
}
```

## Analysis Types

The `/comprehensive_analysis` endpoint supports different analysis types:

- **comprehensive**: Full analysis across all dimensions
- **quality**: Code quality metrics and maintainability
- **complexity**: Cyclomatic complexity and complexity distribution
- **dependencies**: Import analysis and dependency graphs
- **architecture**: Design patterns and architectural analysis
- **patterns**: Code pattern detection and statistics
- **debt**: Technical debt analysis and refactoring suggestions
- **security**: Security vulnerability detection and recommendations
- **performance**: Performance bottleneck identification
- **testing**: Test coverage analysis and test metrics

## Quality Metrics

### Cyclomatic Complexity Ranks
- **A**: 1-5 (Low complexity, easy to maintain)
- **B**: 6-10 (Moderate complexity)
- **C**: 11-20 (High complexity, consider refactoring)
- **D**: 21-30 (Very high complexity)
- **E**: 31-40 (Extremely high complexity)
- **F**: 41+ (Unmaintainable complexity)

### Maintainability Index Ranks
- **A**: 85-100 (Highly maintainable)
- **B**: 65-84 (Moderately maintainable)
- **C**: 45-64 (Somewhat maintainable)
- **D**: 25-44 (Difficult to maintain)
- **F**: 0-24 (Very difficult to maintain)

## Error Handling

All endpoints return appropriate HTTP status codes:
- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error (analysis failed)

Error responses include detailed error messages:
```json
{
  "detail": "Analysis failed: Repository not found"
}
```

## Usage Examples

### Python
```python
import requests

# Comprehensive analysis
response = requests.post("https://your-api-url/comprehensive_analysis", 
                        json={"repo_url": "owner/repo", "analysis_type": "comprehensive"})
data = response.json()

# Function analysis
response = requests.post("https://your-api-url/function_analysis", 
                        json={"repo_url": "owner/repo"})
functions = response.json()["functions"]
```

### JavaScript
```javascript
// Quality report
const response = await fetch('https://your-api-url/code_quality_report', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({repo_url: 'owner/repo'})
});
const report = await response.json();
```

### cURL
```bash
# Architecture insights
curl -X POST "https://your-api-url/architecture_insights" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "owner/repo"}'
```

## Performance Considerations

- Analysis time scales with repository size
- Large repositories (>10k files) may take several minutes
- Consider using specific analysis types for faster results
- Results are not cached; each request performs fresh analysis

## Limitations

- Supports repositories accessible via GitHub
- Analysis depth depends on code language support in graph-sitter
- Some advanced metrics require complete code parsing
- External dependency analysis limited to import statements

## Future Enhancements

- Caching for improved performance
- Incremental analysis for large repositories
- Additional language support
- Real-time analysis via webhooks
- Historical trend analysis
- Custom metric definitions
