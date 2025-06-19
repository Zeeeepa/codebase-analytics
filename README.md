# 🔍 Codebase Analytics

Comprehensive repository analysis with interactive exploration and issue detection.

## 🚀 Quick Start

### Start the Backend Server
```bash
cd backend
python api.py
```

The server will start on `http://localhost:8000` with the following endpoints:

## 🌐 API Endpoints

### 🎨 Interactive UI
```
http://localhost:8000/ui
```
- Modern web interface for repository analysis
- Interactive file tree with issue counts
- Click on files to see functions/classes
- Real-time analysis metrics

### 📊 Comprehensive Analysis API
```
GET /analyze/{owner}/{repo}
```
**Example:**
```bash
curl http://localhost:8000/analyze/Zeeeepa/codebase-analytics
```

**Returns:**
- Repository overview (files, functions, classes, symbols)
- Issue detection with severity classification
- Code quality metrics (maintainability, complexity, debt ratio)
- Dependency analysis (circular deps, external/internal)
- Call graph analysis (entry points, leaf functions, depth)
- Interactive repository structure
- Symbol details mapping

### 🖥️ CLI-Friendly Endpoint
```
GET /cli/{owner}/{repo}
```
**Example:**
```bash
curl http://localhost:8000/cli/Zeeeepa/codebase-analytics
```

**Returns formatted text output:**
```
🔍 CODEBASE ANALYSIS REPORT
==================================================
📊 Repository: Zeeeepa/codebase-analytics
🌐 URL: https://github.com/Zeeeepa/codebase-analytics

📈 OVERVIEW:
- Files: 60
- Functions: 145
- Classes: 20
- Symbols: 2724

🚨 ISSUES SUMMARY:
- Total Issues: 169
- Critical: 0
- Major: 53
- Minor: 116

📊 CODE QUALITY:
- Maintainability Index: 100.0
- Cyclomatic Complexity: 1.0
- Comment Density: 0.01
- Technical Debt Ratio: 0.25

🎯 ANALYSIS COMPLETE!
```

## 📂 Interactive Repository Structure

The UI provides a clickable repository structure with issue counts:

```
📁 backend/ [👉 Major: 53] [🔍 Minor: 116]
├── 📄 analysis.py [👉 Major: 45] [🔍 Minor: 98]
│   ├── ⚙️ Functions: analyze_codebase, detect_issues, analyze_code_quality...
│   ├── 🏗️ Classes: Issue, CodeQualityResult, DependencyAnalysis...
│   └── 🚨 Issues: Line 802: 'Unused parameter "import_string"'
├── 📄 api.py [👉 Major: 8] [🔍 Minor: 18]
└── 📄 visualization.py
```

**Interactive Features:**
- ✅ Click folders → Shows contents
- ✅ Click files → Shows symbol map (functions/classes)
- ✅ Click symbols → Shows parameters/context/issues
- ✅ Issue badges by severity with counts
- ✅ Real-time issue details

## 🔧 Analysis Features

### 🚨 Issue Detection
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

### 📊 Code Quality Metrics
- Maintainability Index
- Cyclomatic Complexity
- Comment Density
- Source Lines of Code
- Duplication Percentage
- Technical Debt Ratio

### 🔗 Dependency Analysis
- Total Dependencies
- Circular Dependencies
- External Dependencies
- Internal Dependencies
- Dependency Depth
- Critical Dependencies

### 📞 Call Graph Analysis
- Total Functions
- Entry Points
- Leaf Functions
- Maximum Call Depth
- Call Chains

## 🏗️ Architecture

### Backend Files
- `analysis.py` - Core analysis functions (50 functions, 20 classes)
- `api.py` - FastAPI server with all endpoints
- `visualization.py` - Visualization generation

### Frontend Files
- `frontend/interactive-analysis.html` - Interactive web UI

## 🎯 Usage Examples

### Start Server
```bash
cd backend
python api.py
```

### Access Interactive UI
Open browser to: `http://localhost:8000/ui`

### API Analysis
```bash
curl http://localhost:8000/analyze/Zeeeepa/codebase-analytics | jq
```

### CLI Analysis
```bash
curl http://localhost:8000/cli/Zeeeepa/codebase-analytics
```

## ✅ Features

- ✅ Comprehensive codebase analysis
- ✅ Interactive repository structure
- ✅ Issue detection with severity classification
- ✅ Code quality metrics
- ✅ Dependency and call graph analysis
- ✅ Modern web UI with real-time updates
- ✅ CLI-friendly endpoints for automation
- ✅ Visual representations of all analysis components

**Ready for comprehensive codebase analysis! 🎉**

