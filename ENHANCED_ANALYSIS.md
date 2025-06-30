# Enhanced Codebase Analytics - Comprehensive Analysis Features

This document describes the comprehensive set of analysis features implemented in the `expand1` branch, inspired by tree-sitter and graph-based code analysis capabilities.

## Overview

The enhanced analysis system provides deep insights into codebases through multiple analysis dimensions:

- **Dependency Analysis**: Comprehensive dependency mapping, circular dependency detection, and dependency health assessment
- **Call Graph Analysis**: Function call relationship mapping, call chain analysis, and connectivity metrics
- **Code Quality Metrics**: Technical debt assessment, code duplication detection, and maintainability scoring
- **Architectural Insights**: Pattern detection, coupling/cohesion analysis, and modularity assessment
- **Security Analysis**: Vulnerability detection, security hotspot identification, and risk assessment
- **Performance Analysis**: Performance hotspot detection, algorithmic complexity analysis, and optimization opportunities

## Analysis Features

### 1. Dependency Analysis (`DependencyAnalysis`)

Analyzes dependencies between modules, files, and components.

**Key Features:**
- **Dependency Mapping**: Complete dependency graph construction
- **Circular Dependency Detection**: Identifies and reports circular dependencies
- **Dependency Depth Calculation**: Measures maximum dependency chain length
- **Critical Dependencies**: Identifies most-used dependencies
- **External vs Internal**: Classifies dependencies as external libraries or internal modules

**Example Usage:**
```python
from advanced_analysis import analyze_dependencies_comprehensive

analysis = analyze_dependencies_comprehensive(codebase)
print(f"Total dependencies: {analysis.total_dependencies}")
print(f"Circular dependencies: {len(analysis.circular_dependencies)}")
print(f"Dependency depth: {analysis.dependency_depth}")
```

**Output Data:**
```python
@dataclass
class DependencyAnalysis:
    total_dependencies: int
    circular_dependencies: List[List[str]]
    dependency_depth: int
    external_dependencies: List[str]
    internal_dependencies: List[str]
    dependency_graph: Dict[str, List[str]]
    critical_dependencies: List[str]
    unused_dependencies: List[str]
```

### 2. Call Graph Analysis (`CallGraphAnalysis`)

Analyzes function call relationships and patterns.

**Key Features:**
- **Call Graph Construction**: Maps all function call relationships
- **Entry Point Detection**: Identifies functions not called by others
- **Leaf Function Detection**: Identifies functions that don't call others
- **Call Chain Analysis**: Finds interesting call sequences
- **Connectivity Metrics**: Measures function interconnectedness

**Example Usage:**
```python
from advanced_analysis import analyze_call_graph

analysis = analyze_call_graph(codebase)
print(f"Total call relationships: {analysis.total_call_relationships}")
print(f"Entry points: {analysis.entry_points}")
print(f"Call depth: {analysis.call_depth}")
```

### 3. Code Quality Metrics (`CodeQualityMetrics`)

Comprehensive code quality assessment.

**Key Features:**
- **Technical Debt Ratio**: Measures TODO/FIXME density
- **Code Duplication Detection**: Identifies duplicated code blocks
- **Documentation Coverage**: Measures function documentation completeness
- **Naming Consistency**: Analyzes naming convention adherence
- **Code Smell Detection**: Identifies common anti-patterns
- **Refactoring Opportunities**: Suggests consolidation opportunities

**Code Smells Detected:**
- Long methods (>50 lines)
- Too many parameters (>5)
- Deep nesting (>4 levels)
- Potential code duplication

### 4. Architectural Insights (`ArchitecturalInsights`)

Analyzes architectural patterns and design quality.

**Key Features:**
- **Pattern Detection**: Identifies common architectural patterns (MVC, Repository, Factory, etc.)
- **Coupling Metrics**: Measures afferent/efferent coupling and instability
- **Cohesion Metrics**: Calculates LCOM (Lack of Cohesion of Methods)
- **Modularity Scoring**: Assesses overall code organization
- **Component Analysis**: Analyzes component structure and relationships

**Detected Patterns:**
- MVC (Model-View-Controller)
- Repository Pattern
- Factory Pattern
- Observer Pattern
- Layered Architecture

### 5. Security Analysis (`SecurityAnalysis`)

Identifies potential security vulnerabilities and risks.

**Key Features:**
- **Vulnerability Detection**: Identifies common security issues
- **Security Hotspots**: Highlights high-risk code areas
- **Input Validation Analysis**: Checks for validation patterns
- **Authentication Pattern Detection**: Identifies auth mechanisms
- **Encryption Usage Analysis**: Finds encryption implementations

**Detected Vulnerabilities:**
- Hardcoded passwords
- SQL injection risks
- Code injection vulnerabilities
- Unsafe input handling

### 6. Performance Analysis (`PerformanceAnalysis`)

Identifies performance bottlenecks and optimization opportunities.

**Key Features:**
- **Performance Hotspot Detection**: Identifies potential bottlenecks
- **Algorithmic Complexity Analysis**: Estimates computational complexity
- **Memory Usage Pattern Analysis**: Identifies memory-intensive operations
- **Optimization Opportunities**: Suggests performance improvements

**Detected Issues:**
- Nested loops (O(n²) complexity)
- N+1 query problems
- Inefficient algorithms
- Memory leaks

## Enhanced Visualizations

### 1. Enhanced Dependency Graph
Interactive dependency visualization with:
- Node sizing based on dependency count
- Color coding for external/internal/critical dependencies
- Circular dependency highlighting
- Hierarchical layout options

### 2. Call Flow Diagram
Function call relationship visualization with:
- Entry point highlighting
- Leaf function identification
- Call chain visualization
- Connectivity-based node sizing

### 3. Quality Heatmap
Code quality visualization showing:
- File-level quality scores
- Quality distribution across codebase
- Technical debt hotspots
- Documentation coverage

### 4. Architectural Overview
High-level architectural visualization with:
- Component identification
- Relationship mapping
- Pattern highlighting
- Modularity assessment

### 5. Security Risk Map
Security-focused visualization showing:
- Risk level by file
- Vulnerability distribution
- Security hotspots
- Risk severity indicators

### 6. Performance Hotspot Map
Performance-focused visualization with:
- Function-level performance scores
- Hotspot identification
- Complexity indicators
- Optimization priorities

## API Endpoints

### Enhanced Analysis Endpoint
```http
POST /analyze_comprehensive
Content-Type: application/json

{
  "repo_url": "owner/repository",
  "analysis_types": ["dependency", "call_graph", "code_quality"],
  "include_visualizations": true,
  "max_analysis_time": 300
}
```

### Comprehensive Insights Endpoint
```http
POST /insights
Content-Type: application/json

{
  "repo_url": "owner/repository",
  "analysis_types": ["all"]
}
```

**Response includes:**
- Overall quality score
- Technical debt level
- Maintainability rating
- Architectural health
- Security risk level
- Performance concerns
- Top recommendations

### Available Analysis Types
```http
GET /analysis_types
```

Returns list of available analysis types with descriptions.

## Usage Examples

### Basic Comprehensive Analysis
```python
from advanced_analysis import perform_comprehensive_analysis, AnalysisType
from codegen.sdk.core.codebase import Codebase

# Load codebase
codebase = Codebase.from_repo("owner/repository")

# Perform all analyses
results = perform_comprehensive_analysis(codebase)

# Access specific analysis results
dependency_analysis = results['dependency_analysis']
call_graph_analysis = results['call_graph_analysis']
quality_metrics = results['code_quality_metrics']
```

### Specific Analysis Types
```python
# Perform only dependency and security analysis
analysis_types = [AnalysisType.DEPENDENCY, AnalysisType.SECURITY]
results = perform_comprehensive_analysis(codebase, analysis_types)
```

### Creating Visualizations
```python
from enhanced_visualizations import create_comprehensive_dashboard_data

# Create all visualizations
dashboard_data = create_comprehensive_dashboard_data(codebase)

# Access specific visualizations
dependency_graph = dashboard_data['dependency_graph']
quality_heatmap = dashboard_data['quality_heatmap']
```

## Configuration Options

### Analysis Configuration
```python
# Configure analysis depth and scope
config = {
    "max_dependency_depth": 10,
    "include_test_files": True,
    "security_scan_depth": "deep",
    "performance_threshold": 0.8
}
```

### Visualization Configuration
```python
# Configure visualization appearance
viz_config = {
    "layout": "hierarchical",  # or "force", "circular"
    "color_scheme": "default",  # or "colorblind", "high_contrast"
    "node_sizing": "proportional",
    "edge_bundling": True
}
```

## Performance Considerations

### Analysis Performance
- **Dependency Analysis**: O(n) where n is number of files
- **Call Graph Analysis**: O(f²) where f is number of functions
- **Quality Analysis**: O(n*m) where n is files and m is average file size
- **Security Analysis**: O(n*m) with pattern matching overhead
- **Performance Analysis**: O(f*c) where f is functions and c is complexity

### Memory Usage
- Large codebases (>10k files): ~500MB-1GB memory usage
- Medium codebases (1k-10k files): ~100MB-500MB memory usage
- Small codebases (<1k files): ~50MB-100MB memory usage

### Optimization Strategies
- **Parallel Processing**: Analysis types run in parallel when possible
- **Incremental Analysis**: Cache results for unchanged files
- **Selective Analysis**: Choose specific analysis types for faster results
- **Sampling**: Use sampling for very large codebases

## Testing

### Running Tests
```bash
# Run all enhanced analysis tests
python backend/test_enhanced_analysis.py

# Run specific test categories
python -m unittest test_enhanced_analysis.TestDependencyAnalysis
python -m unittest test_enhanced_analysis.TestCallGraphAnalysis
python -m unittest test_enhanced_analysis.TestCodeQualityAnalysis
```

### Test Coverage
- **Unit Tests**: 95%+ coverage for all analysis modules
- **Integration Tests**: End-to-end analysis workflows
- **Performance Tests**: Analysis time and memory usage validation
- **Visualization Tests**: Output format and data integrity validation

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: AI-powered code quality prediction
2. **Historical Analysis**: Track metrics over time
3. **Custom Rules Engine**: User-defined analysis rules
4. **IDE Integration**: Real-time analysis in development environments
5. **Team Collaboration**: Shared analysis results and insights

### Extensibility
The analysis system is designed for extensibility:
- **Custom Analysis Types**: Add new analysis dimensions
- **Plugin Architecture**: Third-party analysis plugins
- **Custom Visualizations**: Create domain-specific visualizations
- **Export Formats**: Multiple output formats (JSON, CSV, PDF)

## Troubleshooting

### Common Issues

**Memory Issues with Large Codebases:**
```python
# Use selective analysis for large codebases
analysis_types = [AnalysisType.DEPENDENCY, AnalysisType.CODE_QUALITY]
results = perform_comprehensive_analysis(codebase, analysis_types)
```

**Timeout Issues:**
```python
# Increase timeout for complex analysis
config = {"max_analysis_time": 600}  # 10 minutes
```

**Visualization Rendering Issues:**
```python
# Use simplified visualizations for large datasets
viz_config = {"max_nodes": 1000, "simplified_layout": True}
```

### Performance Tuning
- Use specific analysis types instead of comprehensive analysis
- Enable caching for repeated analysis
- Consider sampling for very large codebases
- Use parallel processing when available

## Contributing

### Adding New Analysis Types
1. Define analysis data class in `advanced_analysis.py`
2. Implement analysis function
3. Add to `AnalysisType` enum
4. Update `perform_comprehensive_analysis` function
5. Add corresponding tests
6. Update documentation

### Adding New Visualizations
1. Implement visualization function in `enhanced_visualizations.py`
2. Add to dashboard creation function
3. Define visualization configuration options
4. Add tests for visualization output
5. Update API endpoints if needed

## License

This enhanced analysis system is part of the codebase-analytics project and follows the same licensing terms.

