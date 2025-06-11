# 🚀 Advanced Codebase Analytics API

## Overview

This enhanced API provides **intelligent, context-aware repository analysis** with real-time issue detection. It goes beyond simple pattern matching to deliver **dynamic analysis based on actual code state** and semantic understanding.

## 🎯 Key Features

### ✨ **Intelligent Analysis**
- **Real-time Code Intelligence**: Dynamic issue detection based on actual AST analysis
- **Context-aware Error Identification**: Understands specific codebase patterns
- **Intelligent Severity Assessment**: Based on actual impact and usage patterns
- **Adaptive Reporting**: Highlights the most critical issues for each specific project

### 🧠 **Advanced Analysis Capabilities**
- **Live Dependency Analysis**: Shows actual import chains and circular dependencies
- **Runtime Pattern Detection**: Identifies anti-patterns specific to the codebase
- **Performance Bottleneck Identification**: Based on actual code complexity and usage
- **Security Vulnerability Scanning**: Context-aware threat assessment
- **Code Smell Detection**: Adapts to the project's architecture and patterns

### 🔍 **Instant Issue Identification**
- **Real-time AST Parsing**: Catches syntax errors, type mismatches, and logic flaws
- **Semantic Analysis**: Identifies incorrect implementations and architectural issues
- **Cross-file Analysis**: Detects inconsistencies and breaking changes
- **Best Practice Validation**: Tailored to the specific technology stack and project type

### 📊 **Intuitive Visualization**
- **Interactive Heat Maps**: Shows actual usage patterns and problem areas
- **Smart Categorization**: Issues based on their real impact on the codebase
- **Contextual Suggestions**: With specific code examples and fixes
- **Priority-based Reporting**: Focuses on issues that matter most for that specific project

## 🏗️ Architecture

### Core Components

1. **IntelligentCodeAnalyzer**: Main analysis engine with AI-powered insights
2. **Advanced Data Models**: Rich context information for code elements
3. **Pattern Detection Engine**: Language-specific security, performance, and maintainability patterns
4. **Visualization Engine**: Creates comprehensive charts and heat maps
5. **Quality Scoring System**: Calculates overall quality scores and grades

### Analysis Pipeline

```
Repository URL → Clone → Graph-sitter Analysis → Pattern Matching → 
Issue Detection → Context Generation → Quality Scoring → 
Visualization → Intelligent Insights → Response
```

## 📋 API Endpoints

### `GET /`
Returns API information and available features.

### `GET /health`
Health check endpoint with system status.

### `POST /analyze`
**Main analysis endpoint** - Performs comprehensive repository analysis.

#### Request Body:
```json
{
  "repo_url": "https://github.com/owner/repo",
  "analysis_depth": "comprehensive",
  "focus_areas": ["security", "performance", "maintainability"],
  "include_context": true,
  "max_issues": 200,
  "enable_ai_insights": true
}
```

#### Response:
```json
{
  "repo_url": "https://github.com/owner/repo",
  "analysis_id": "abc123def456",
  "timestamp": "2024-01-01T12:00:00Z",
  "analysis_duration": 45.2,
  "overall_quality_score": 78.5,
  "quality_grade": "B",
  "risk_assessment": "🟡 Medium Risk - 3 major issues detected",
  "key_findings": [
    "📁 Large codebase with 1,234 files - consider modularization",
    "🔒 5 security vulnerabilities detected",
    "⚡ 12 performance bottlenecks identified"
  ],
  "critical_recommendations": [
    "🚨 Address 2 critical issues immediately",
    "🔐 Review and fix 5 security vulnerabilities"
  ],
  "architecture_assessment": "REST API architecture with 8 entry points",
  "issues": [...],
  "security_analysis": {...},
  "performance_analysis": {...},
  "dependency_graph": {...},
  "inheritance_analysis": [...],
  "entry_points": {...},
  "usage_heatmap": [...],
  "metrics": {...},
  "repository_structure": {...},
  "visualizations": {...}
}
```

## 🔧 Data Models

### IntelligentIssue
Advanced issue detection with context and impact analysis:
- **ID**: Unique identifier
- **Type & Category**: Security, performance, maintainability, logic, style
- **Severity**: Critical, major, minor
- **Context**: Rich context information including dependencies and usage
- **Impact Analysis**: Detailed explanation of the issue's impact
- **Fix Suggestion**: Actionable recommendations
- **Confidence**: AI confidence score (0.0 to 1.0)

### CodeContext
Rich context information for code elements:
- **Element Type**: Function, class, method, variable
- **Location**: File path and line numbers
- **Complexity**: Cyclomatic complexity score
- **Dependencies**: What this element depends on
- **Dependents**: What depends on this element
- **Usage Count**: How frequently it's used
- **Risk Score**: Calculated risk assessment

### AdvancedMetrics
Comprehensive code quality metrics:
- **Halstead Metrics**: Volume, difficulty, effort
- **Cyclomatic Complexity**: Average, total, maximum
- **Maintainability Index**: Average, min, max
- **Technical Debt Ratio**: Percentage of technical debt
- **Code Coverage Estimate**: Estimated test coverage
- **Duplication Percentage**: Code duplication analysis

## 🎨 Visualization Features

### Repository Structure Tree
```
📁✅ project-root/
├── 📁🔴 src/ [Total: 15 issues]
│   ├── 🔴 main.py [⚠️ Critical: 2] [👉 Major: 1]
│   ├── ✅ utils.py [No issues]
│   └── 📁🟡 components/ [👉 Major: 3]
└── 📁✅ tests/ [No issues]
```

### Issue Distribution Charts
- **By Severity**: Critical, Major, Minor
- **By Category**: Security, Performance, Maintainability, Style
- **Heat Map**: Usage frequency vs complexity
- **Risk Distribution**: High, Medium, Low risk areas

### Dependency Graph
- **Nodes**: Files, modules, functions
- **Edges**: Import relationships, function calls
- **Circular Dependencies**: Detected cycles
- **Critical Paths**: Most important dependency chains

## 🚀 Getting Started

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the API**:
```bash
python api.py
```

3. **Access Documentation**:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Example Usage

```python
import requests

# Analyze a repository
response = requests.post("http://localhost:8000/analyze", json={
    "repo_url": "https://github.com/owner/repo",
    "analysis_depth": "comprehensive",
    "focus_areas": ["security", "performance"],
    "max_issues": 100
})

analysis = response.json()
print(f"Quality Score: {analysis['overall_quality_score']}")
print(f"Grade: {analysis['quality_grade']}")
print(f"Risk: {analysis['risk_assessment']}")
```

## 🔒 Security Features

- **Rate Limiting**: 60 requests per minute per IP
- **Security Headers**: XSS protection, content type validation
- **Input Validation**: Repository URL validation
- **Timeout Protection**: Repository cloning timeouts
- **Error Handling**: Graceful degradation on failures

## 🎯 Advanced Features

### Graph-sitter Integration
When available, the API uses graph-sitter for:
- **AST-based Analysis**: Deep semantic understanding
- **Language-specific Patterns**: Tailored to each programming language
- **Context Generation**: Rich contextual information
- **Cross-reference Analysis**: Understanding symbol relationships

### Fallback Analysis
When graph-sitter is not available:
- **Pattern-based Detection**: Regex-based issue identification
- **Language Detection**: Automatic language identification
- **Basic Metrics**: Complexity and maintainability calculations
- **File Structure Analysis**: Repository organization assessment

## 📈 Performance

- **Parallel Processing**: Multi-threaded analysis
- **Caching**: Analysis result caching
- **Streaming**: Large repository support
- **Memory Management**: Efficient memory usage
- **Timeout Handling**: Prevents hanging operations

## 🔧 Configuration

### Environment Variables
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `MAX_REPO_SIZE`: Maximum repository size for analysis
- `ANALYSIS_TIMEOUT`: Maximum analysis time in seconds
- `CACHE_TTL`: Cache time-to-live in seconds

### Analysis Depth Levels
- **Quick**: Basic pattern matching and metrics
- **Standard**: Full pattern analysis with basic context
- **Comprehensive**: Complete analysis with rich context
- **Deep**: Maximum analysis depth with AI insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**🎯 The Enhanced Codebase Analytics API provides intelligent, real-time analysis that adapts to your specific codebase, delivering actionable insights that help you build better software.**
