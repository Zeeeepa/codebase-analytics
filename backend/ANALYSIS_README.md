# Comprehensive Codebase Analysis Engine

This document describes the consolidated `analysis.py` module that provides comprehensive codebase analysis using the Codegen SDK (graph-sitter).

## Overview

The consolidated `analysis.py` module combines functionality from:
- `comprehensive_analysis.py` - Full-featured analysis implementation (merged)
- Codegen SDK `codebase_analysis.py` - Summary and analysis functions (imported)
- Codegen SDK `codebase_context.py` - Graph management and context (imported)
- Codegen SDK `codebase_ai.py` - AI prompt generation functions (imported)

## Key Features

### 🔍 **Comprehensive Analysis**
- **Dead Code Detection**: Unused functions, classes, imports, parameters
- **Parameter Issues**: Unused parameters, parameter mismatches, type issues
- **Type Annotation Analysis**: Missing type annotations for functions and parameters
- **Circular Dependency Detection**: Identifies circular dependencies in call graphs
- **Implementation Issues**: Empty functions, missing return statements, etc.

### ���� **Advanced Metrics**
- **Function Context Analysis**: Dependencies, usages, call chains
- **Call Graph Metrics**: Graph density, strongly connected components
- **Halstead Metrics**: Vocabulary, length, volume calculations
- **Complexity Analysis**: Cyclomatic complexity scoring

### 🤖 **AI Integration**
- **System Prompt Generation**: AI-powered code analysis prompts
- **Context Generation**: Rich context for AI interactions
- **Flagging Tools**: AI-powered code flagging decisions

### 🏗️ **Graph-sitter Integration**
- **Pre-computed Relationships**: Leverages Codegen SDK's graph structure
- **Efficient Analysis**: Uses pre-computed symbol dependencies and usages
- **Rich Summaries**: Detailed codebase, file, class, function, and symbol summaries
- **Direct Imports**: Functions imported from SDK modules rather than copied

## Architecture

### Core Classes

#### `ComprehensiveCodebaseAnalyzer`
Main analysis engine that orchestrates all analysis steps:

```python
analyzer = ComprehensiveCodebaseAnalyzer("./path/to/repo")
result = analyzer.analyze()
```

#### `Issue`
Represents analysis issues with severity levels:

```python
@dataclass
class Issue:
    item: Any
    type: str
    message: str
    severity: str
    location: Optional[str]
    suggestion: Optional[str]
```

#### `FunctionContext`
Comprehensive function analysis context:

```python
@dataclass
class FunctionContext:
    name: str
    filepath: str
    source: str
    parameters: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    usages: List[Dict[str, Any]]
    function_calls: List[str]
    called_by: List[str]
    max_call_chain: List[str]
    issues: List[Issue]
    is_entry_point: bool
    is_dead_code: bool
    complexity_score: int
    halstead_metrics: Dict[str, Any]
```

### Analysis Types

#### Issue Types
- `UNUSED_FUNCTION`, `UNUSED_CLASS`, `UNUSED_IMPORT`, `UNUSED_PARAMETER`
- `PARAMETER_MISMATCH`, `MISSING_TYPE_ANNOTATION`
- `CIRCULAR_DEPENDENCY`, `EMPTY_FUNCTION`, `MISSING_RETURN`
- `IMPLEMENTATION_ERROR` and many more...

#### Severity Levels
- `CRITICAL`: Critical issues requiring immediate attention
- `ERROR`: Errors that may cause runtime failures
- `WARNING`: Issues that should be addressed
- `INFO`: Informational suggestions

## Usage Examples

### Basic Analysis

```python
from analysis import ComprehensiveCodebaseAnalyzer

# Analyze local repository
analyzer = ComprehensiveCodebaseAnalyzer("./my-project")
result = analyzer.analyze()

print(f"Found {result['summary']['total_issues']} issues")
print(f"Analysis took {result['duration']:.2f} seconds")
```

### GitHub Repository Analysis

```python
# Analyze GitHub repository
analyzer = ComprehensiveCodebaseAnalyzer("https://github.com/user/repo")
result = analyzer.analyze()

# Access different analysis results
issues_by_severity = result['issues']['by_severity']
function_contexts = result['function_contexts']
dead_code = result['dead_code_analysis']
```

### Using SDK Summary Functions

```python
from analysis import get_codebase_summary, get_function_summary

# Get codebase summary
summary = get_codebase_summary(codebase)
print(summary)

# Get function summary
for function in codebase.functions:
    func_summary = get_function_summary(function)
    print(func_summary)
```

### AI Prompt Generation

```python
from analysis import generate_system_prompt, generate_tools

# Generate AI system prompt
prompt = generate_system_prompt(target=function, context=context)

# Generate tool definitions
tools = generate_tools()
```

## Analysis Output Structure

The `analyze()` method returns a comprehensive dictionary:

```python
{
    "success": True,
    "timestamp": "2024-01-01T12:00:00",
    "duration": 5.23,
    "repository": "./my-project",
    "summary": {
        "total_issues": 42,
        "critical_issues": 2,
        "error_issues": 5,
        "warning_issues": 20,
        "info_issues": 15,
        "dead_code_items": 8
    },
    "statistics": {
        "total_files": 150,
        "total_functions": 300,
        "total_classes": 50,
        "total_imports": 200,
        "total_symbols": 500
    },
    "issues": {
        "by_severity": {...},
        "by_type": {...},
        "all": [...]
    },
    "function_contexts": {...},
    "dead_code_analysis": {...},
    "most_important_functions": {...},
    "call_graph_metrics": {...},
    "summaries": {...}
}
```

## SDK Integration

### Graph-sitter (Codegen SDK) Features

The module leverages the Codegen SDK's graph-sitter capabilities:

1. **Pre-computed Graph Structure**: Uses rustworkx + Python for efficient graph operations
2. **Symbol Relationships**: Leverages pre-computed dependencies and usages
3. **Fast Lookups**: Instant access to function calls, class hierarchies, import relationships
4. **Rich Context**: Detailed symbol, file, and codebase summaries

### Fallback Behavior

When the Codegen SDK is not available:
- Basic analysis still works with limited functionality
- Graceful degradation with informative error messages
- Fallback implementations for core data structures

## Validation

Run the validation script to test functionality:

```bash
cd backend
python validate_analysis.py
```

The validation script tests:
- Basic functionality (data structures, AI prompts)
- Analyzer initialization (local and GitHub repos)
- SDK availability and fallback behavior
- Complete analysis workflow

## Performance Considerations

### Optimizations
- **Pre-computed Relationships**: Uses SDK's pre-computed graph for O(1) lookups
- **Lazy Loading**: Only analyzes what's needed
- **Error Handling**: Graceful failure with partial results
- **Memory Efficient**: Streams large codebases without loading everything

### Scalability
- Handles large codebases efficiently
- Configurable analysis depth
- Parallel analysis capabilities (future enhancement)

## Error Handling

The analyzer includes comprehensive error handling:
- Individual analysis step failures don't stop the entire process
- Detailed error reporting with suggestions
- Graceful fallback when SDK is unavailable
- Validation of input parameters and codebase state

## Future Enhancements

Planned improvements:
- **Parallel Analysis**: Multi-threaded analysis for large codebases
- **Custom Rules**: User-defined analysis rules and patterns
- **Integration APIs**: REST API for remote analysis
- **Caching**: Incremental analysis with caching
- **Visualization**: Interactive analysis result visualization

## Dependencies

Required packages:
```
networkx>=3.4.1  # Graph analysis
codegen          # Codegen SDK (optional but recommended)
```

Optional packages for full functionality:
```
fastapi>=0.115.2  # API server
uvicorn==0.27.1   # ASGI server
requests>=2.32.3  # HTTP requests
pydantic>=2.10.0  # Data validation
```

## Contributing

When contributing to the analysis module:
1. Maintain backward compatibility
2. Add comprehensive error handling
3. Include validation tests
4. Update documentation
5. Follow the existing code style

## License

This module is part of the codebase-analytics project and follows the same licensing terms.
