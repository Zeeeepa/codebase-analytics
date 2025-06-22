# Codebase Analytics Backend API

A **unified all-in-one** codebase analysis engine that provides deep insights into code quality, function contexts, dead code detection, and interactive visualizations. Everything consolidated into a single `analysis.py` file!

## 🚀 Features

### 🎯 Unified Analysis Engine (`analysis.py`) - ALL-IN-ONE
**This single file contains ALL analysis functionality consolidated from multiple modules:**

**Core Analysis Features:**
- **Comprehensive Error Detection**: 6 categories with 25+ issue types
- **Function Context Analysis**: Dependencies, call chains, parameters, and issues
- **Dead Code Analysis**: Blast radius calculation and impact assessment
- **Halstead Metrics**: Complete complexity metrics (n1, n2, N1, N2, vocabulary, length, volume, difficulty, effort)
- **Most Important Functions**: Detection using advanced algorithms
- **Entry Point Identification**: Automatic detection of application entry points

**Integrated Advanced Features:**
- **Mock Codebase Generation**: Realistic test data for demonstrations
- **Comprehensive Demo Mode**: Standalone execution with detailed output
- **Function Relationship Analysis**: Dependency mapping and circular dependency detection
- **Code Quality Scoring**: 0-100 scale quality assessment
- **Refactoring Recommendations**: Actionable improvement suggestions
- **Codegen SDK Integration**: With fallback mock implementations

**Demo Mode Usage:**
```bash
python analysis.py  # Run comprehensive demo with detailed analysis output
```

**Integrated API Server:**
- **GET /analyze/{username}/{repo}**: Complete comprehensive analysis
- **GET /visualize/{username}/{repo}**: Interactive visualization data
- **GET /health**: Health check endpoint
- **Caching System**: In-memory caching for performance
- **Enhanced Results**: Refactoring suggestions, impact analysis, cleanup recommendations

**Built-in Visualization Engine:**
- **Repository Tree**: Clickable tree with issue badges and symbol navigation
- **Issue Visualization**: Heatmaps and severity distribution
- **Dead Code Visualization**: Blast radius and impact analysis
- **Call Graph**: Interactive function call relationships
- **Dependency Graph**: Module and function dependencies
- **Metrics Charts**: Complexity, quality, and performance metrics
- **Code Quality Dashboard**: Overall health scoring and recommendations

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- Flask (Web API framework)
- Flask-CORS (Cross-origin resource sharing)
- NetworkX (Graph analysis for call chains)
- Python 3.8+ (Required for dataclasses and type hints)

## 🛠️ Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Zeeeepa/codebase-analytics.git
cd codebase-analytics/backend
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Start the API server**:
```bash
python analysis.py api
```

**OR run the comprehensive demo**:
```bash
python analysis.py
```

The server will start on `http://localhost:5000`

## 📊 API Usage

### 1. Analyze Repository
Get comprehensive analysis including function contexts, issues, dead code, and metrics.

```bash
curl http://localhost:5000/analyze/codegen-sh/graph-sitter
```

**Response Structure**:
```json
{
  "summary": {
    "total_files": 7,
    "total_functions": 5,
    "total_issues": 8,
    "critical_issues": 0,
    "major_issues": 1,
    "minor_issues": 7,
    "dead_code_items": 1,
    "entry_points": 1
  },
  "function_contexts": {
    "process_data": {
      "name": "process_data",
      "filepath": "src/main.py",
      "parameters": [{"name": "data", "type": "dict"}],
      "dependencies": [...],
      "usages": [...],
      "function_calls": ["validate_input", "calculate_total"],
      "called_by": ["main"],
      "max_call_chain": ["process_data", "validate_input", "check_format"],
      "issues": [...],
      "is_entry_point": false,
      "is_dead_code": false,
      "complexity_score": 15,
      "halstead_metrics": {...}
    }
  },
  "most_important_functions": {
    "most_calls": {"name": "main", "call_count": 2},
    "most_called": {"name": "validate_input", "usage_count": 2},
    "deepest_inheritance": {"name": "DataProcessor", "chain_depth": 2}
  },
  "dead_code_analysis": {
    "total_dead_functions": 1,
    "dead_code_items": [...]
  },
  "halstead_metrics": {
    "n1": 15, "n2": 12, "N1": 45, "N2": 38,
    "vocabulary": 27, "length": 83, "volume": 245.2,
    "difficulty": 15.6, "effort": 3825.1
  },
  "refactoring_suggestions": [...],
  "impact_analysis": {...},
  "cleanup_recommendations": [...]
}
```

### 2. Visualize Repository
Get interactive visualization data including repository tree, issue heatmaps, and call graphs.

```bash
curl http://localhost:5000/visualize/codegen-sh/graph-sitter
```

**Response Structure**:
```json
{
  "repository_tree": {
    "name": "codegen-sh/graph-sitter",
    "type": "repository",
    "expanded": true,
    "issue_counts": {"critical": 1, "major": 4, "minor": 15},
    "children": [...]
  },
  "issue_visualization": {
    "severity_distribution": {...},
    "file_heatmap": {...},
    "most_problematic_files": [...]
  },
  "call_graph": {
    "nodes": [...],
    "edges": [...],
    "interactive": true
  },
  "dead_code_visualization": {
    "blast_radius_visualization": {...}
  },
  "ui_components": {
    "layout": {...},
    "search_bar": {...},
    "interactive_features": {...}
  }
}
```

### 3. Health Check
Verify API status and availability.

```bash
curl http://localhost:5000/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "codebase-analytics-api"
}
```

## 🔧 Advanced Usage

### Custom Analysis Types
The analysis engine supports various analysis types:
- `CODE_QUALITY`: Code quality and style issues
- `DEPENDENCIES`: Dependency analysis and circular dependencies
- `PERFORMANCE`: Performance bottlenecks and inefficiencies
- `SECURITY`: Security vulnerabilities and risks
- `DEAD_CODE`: Unused code detection
- `COMPLEXITY`: Cyclomatic and Halstead complexity
- `ISSUES`: Comprehensive issue detection

### Function Context Analysis
Each function provides detailed context including:
- **Implementation**: Source code and file location
- **Dependencies**: Functions and modules this function depends on
- **Usages**: Where this function is called from
- **Call Chain**: Maximum call depth and chain analysis
- **Issues**: Specific problems detected in the function
- **Metrics**: Complexity scores and Halstead metrics

### Issue Categories
The system detects 6 main categories of issues:

1. **Implementation Errors**: Null references, type mismatches, undefined variables
2. **Function Issues**: Parameter problems, wrong counts, unused parameters
3. **Exception Handling**: Missing error handling, unsafe assertions
4. **Code Quality**: Duplication, inefficient patterns, magic numbers
5. **Formatting & Style**: Naming conventions, documentation, indentation
6. **Runtime Risks**: Division by zero, array bounds, infinite loops

## 🎯 Key Algorithms

### Most Important Functions Detection
```python
# Find function that makes the most calls
most_calls = max(codebase.functions, key=lambda f: len(f.function_calls))

# Find the most called function
most_called = max(codebase.functions, key=lambda f: len(f.call_sites))

# Find class with most inheritance
deepest_class = max(codebase.classes, key=lambda x: len(x.superclasses))
```

### Halstead Metrics Calculation
- **n1**: Number of distinct operators
- **n2**: Number of distinct operands
- **N1**: Total number of operators
- **N2**: Total number of operands
- **Vocabulary**: n1 + n2
- **Length**: N1 + N2
- **Volume**: Length × log₂(Vocabulary)
- **Difficulty**: (n1/2) × (N2/n2)
- **Effort**: Difficulty × Volume

### Dead Code Detection
Functions are considered dead code if:
- No call sites found
- No usages detected
- Not an entry point (main, __main__, run, start, execute)

## 🌐 Integration

### Graph-Sitter Integration
The system is designed to integrate with [graph-sitter](https://graph-sitter.com/) for:
- Multi-language parsing support
- AST-based analysis
- Precise symbol extraction
- Cross-language dependency tracking

### Web Frontend Integration
The visualization endpoints provide data structures optimized for:
- Interactive tree components
- Issue heatmap visualizations
- Call graph rendering
- Metrics dashboard display

## 🚀 Performance

- **Caching**: In-memory caching for repeated requests
- **Lazy Loading**: Analysis triggered only when needed
- **Efficient Algorithms**: Optimized graph traversal and analysis
- **Scalable Architecture**: Modular design for easy extension

## 🔍 Example Output

### Comprehensive Demo Mode
Run the unified analysis engine directly to see all features:

```bash
cd backend
python analysis.py
```

**Sample Demo Output:**
```
🚀 COMPREHENSIVE CODEBASE ANALYSIS DEMO
==================================================
📁 Loading codebase...
🔍 Performing comprehensive analysis...

📊 ANALYSIS SUMMARY:
------------------------------
📁 Total Files: 12
🔧 Total Functions: 8
🚨 Total Issues: 8
💀 Dead Code Items: 2
🎯 Entry Points: 1

🌟 MOST IMPORTANT FUNCTIONS:
-----------------------------------
📞 Makes Most Calls: main (Call Count: 2)
📈 Most Called: process_data (Usage Count: 1)
🌳 Deepest Inheritance: AdvancedProcessor (Chain Depth: 3)

📐 HALSTEAD COMPLEXITY METRICS:
-----------------------------------
📚 Vocabulary: 71
📏 Length: 168
📊 Volume: 1033.16
🎯 Difficulty: 17.02
⚡ Effort: 17582.46
```

### Repository Structure Visualization
```
codegen-sh/graph-sitter/
├── 📁 src/ [Total: 20 issues]
│   ├── 📁 graph_sitter/ [⚠️ Critical: 1] [👉 Major: 4] [🔍 Minor: 15]
│   │   ├── 📁 core/ [⚠️ Critical: 1]
│   │   │   └── 📄 codebase.py [⚠️ Critical: 1]
│   │   ├── 📁 python/ [👉 Major: 4] [🔍 Minor: 5]
│   │   │   ├── 📄 file.py [👉 Major: 4] [🔍 Minor: 3]
│   │   │   └── 📄 function.py [🔍 Minor: 2]
│   │   └── 📁 typescript/ [🔍 Minor: 10]
└── 📁 docs/
```

### Function Analysis Example
```json
{
  "name": "process_data",
  "complexity_score": 15,
  "halstead_metrics": {
    "volume": 245.2,
    "difficulty": 15.6,
    "effort": 3825.1
  },
  "issues": [
    {
      "type": "unused_parameter",
      "severity": "minor",
      "message": "Parameter 'unused_param' appears to be unused"
    }
  ],
  "max_call_chain": ["process_data", "validate_input", "check_format"]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the API health endpoint: `/health`
- Review the comprehensive analysis output for debugging

---

**Built with ❤️ for better code quality and maintainability**
