# 📊 Repository Analysis Report 📊

**Example Output from Enhanced Codebase Analytics API**

==================================================
📁 **Repository**: codegen-sh/graph-sitter  
📝 **Description**: Advanced codebase analysis with graph-sitter integration

📁 **Files**: 42  
🔄 **Functions**: 156  
📏 **Classes**: 23  

## Repository Structure:

```
codegen-sh/graph-sitter/
├── 📁 .codegen/
├── 📁 .github/
├── 📁 .vscode/
├── 📁 architecture/
├── 📁 docs/ [🐍codefiles: 8] [📏Lines: 1452]
│   ├── 📁 api-reference/ [🐍codefiles: 2] [📏Lines: 1452]
│   ├── 📁 building-with-graph-sitter/ [🐍codefiles: 4] [📏Lines: 1452]
│   └── 📁 changelog/
├── 📁 examples/
├── 📁 scripts/
├── 📁 src/ 
│   ├── 📁 codemods/
│   ├── 📁 graph_sitter/ [⚡ Issues: 20]
│   │   ├── 📁 ai/
│   │   ├── 📁 cli/
│   │   ├── 📁 code_generation/
│   │   ├── 📁 codebase/
│   │   │   ├── 📁 factory/
│   │   │   ├── 📁 flagging/
│   │   │   ├── 📁 io/
│   │   │   ├── 📁 node_classes/
│   │   │   └── 📁 progress/
│   │   ├── 📁 compiled/
│   │   ├── 📁 configs/
│   │   ├── 📁 core/ [⚠️ Critical: 1] 
│   │   │   └── 🐍 autocommit.py [⚠️ Critical: 1]
│   │   ├── 📁 extensions/
│   │   ├── 📁 git/
│   │   ├── 📁 gscli/
│   │   ├── 📁 output/
│   │   ├── 📁 python/ [⚠️ Critical: 1]
│   │   │   ├── 🐍 file.py [⚠️ Critical: 4] 
│   │   │   └── 🐍 function.py [⚠️ Critical: 1] 
│   │   ├── 📁 runner/
│   │   ├── 📁 shared/
│   │   ├── 📁 typescript/ [⚠️ Critical: 3] 
│   │   │   └── 🐍 symbol.py [⚠️ Critical: 3]
│   │   └── 📁 visualizations/
│   └── 📁 gsbuild/
└── 📁 tests/
    ├── 📁 integration/
    └── 📁 unit/
```

## 🔍 **CRITICAL ISSUES DETECTED** (7 critical issues found)

⚠️  **src/graph_sitter/core/autocommit.py:45** - High cyclomatic complexity: 28  
   📍 Function 'process_commits' has cyclomatic complexity of 28, which exceeds recommended threshold of 15  
   💡 Suggestion: Break down the function into smaller, more focused functions  

⚠️  **src/graph_sitter/python/file.py:123** - Long function: 87 lines  
   📍 Function 'parse_file_structure' is 87 lines long, exceeding recommended maximum of 50 lines  
   💡 Suggestion: Split function into smaller, single-responsibility functions  

⚠️  **src/graph_sitter/python/file.py:234** - Too many parameters: 9  
   📍 Function 'analyze_dependencies' has 9 parameters, exceeding recommended maximum of 7  
   💡 Suggestion: Group related parameters into a configuration object  

⚠️  **src/graph_sitter/python/file.py:456** - God class: 25 methods  
   📍 Class 'FileAnalyzer' has 25 methods, indicating it may have too many responsibilities  
   💡 Suggestion: Apply Single Responsibility Principle - split class into smaller classes  

⚠️  **src/graph_sitter/python/function.py:78** - Deep inheritance: 6 levels  
   📍 Class 'AdvancedFunctionAnalyzer' has inheritance depth of 6, which may indicate over-engineering  
   💡 Suggestion: Consider using composition over inheritance  

⚠️  **src/graph_sitter/typescript/symbol.py:156** - Hardcoded API key  
   📍 Potential security vulnerability: hardcoded API key detected  
   💡 Suggestion: Move sensitive data to environment variables  

⚠️  **src/graph_sitter/typescript/symbol.py:289** - Use of eval() function  
   📍 Security vulnerability: eval() function usage detected  
   💡 Suggestion: Replace eval() with safer alternatives like ast.literal_eval()  

## 🏗️ **INHERITANCE ANALYSIS**

**Classes with most inheritance:**

src/graph_sitter/python/file.py [⚕️FileAnalyzer] (Depth: 4)  
   └── Inherits from: BaseAnalyzer, MetricsCollector, CacheManager  

src/graph_sitter/python/function.py [⚕️AdvancedFunctionAnalyzer] (Depth: 6)  
   └── Inherits from: FunctionAnalyzer, ComplexityCalculator, MetricsProvider, BaseProcessor, CacheableEntity  

src/graph_sitter/typescript/symbol.py [⚕️TypeScriptSymbolProcessor] (Depth: 3)  
   └── Inherits from: SymbolProcessor, TypeAnalyzer  

## 🤖 **AUTOMATIC RESOLUTION SUGGESTIONS**

🔴 **Refactor complex function 'process_commits'** (Confidence: 85%)  
   • Extract repeated function calls into helper methods  
   • Group related parameters into configuration objects  

🔴 **Optimize dependencies in src/graph_sitter/core/autocommit.py** (Confidence: 90%)  
   • Remove unused imports using graph-sitter analysis  
   • Reorganize import statements  

🟡 **Improve maintainability in src/graph_sitter/python/file.py** (Confidence: 75%)  
   • Add comprehensive documentation  
   • Implement unit tests  

🟡 **Improve maintainability in src/graph_sitter/python/function.py** (Confidence: 75%)  
   • Add comprehensive documentation  
   • Implement unit tests  

🟢 **Clean up code smell in FileAnalyzer** (Confidence: 70%)  
   • Apply SOLID principles  
   • Remove duplicate code  

## 📈 **GRAPH-SITTER INTEGRATION INSIGHTS**

- **Pre-computed dependency graph** with 1,247 edges
- **Symbol usage analysis** across 42 files  
- **Multi-language support**: Python, TypeScript, React & JSX  
- **Advanced static analysis** for code manipulation operations  

### Key Graph-Sitter Benefits:
- ⚡ **Fast symbol lookups** - No parsing needed for dependency analysis
- 🔍 **Instant usage detection** - Pre-computed relationships enable rapid code analysis
- 🌐 **Cross-language consistency** - Unified interface across Python, TypeScript, and JSX
- 🛠️ **Reliable refactoring** - AST-based transformations ensure code correctness

## 🔧 **RECOMMENDED ACTIONS**

1. 🔴 **Address 7 critical issues immediately**
2. 🟡 **Refactor 12 complex functions**  
3. 🟠 **Review 3 classes with deep inheritance**
4. 🔧 **Leverage graph-sitter's pre-computed dependency graph for faster refactoring**
5. 📊 **Use graph-sitter's symbol usage analysis for safe code transformations**
6. 🚀 **Implement automated code fixes using graph-sitter's AST manipulation**

## 🎯 **AUTOMATIC RESOLUTION CAPABILITIES**

This enhanced analysis leverages **graph-sitter's advanced static analysis** to provide:

### 🔄 **Pre-computed Relationships**
- Instant dependency lookups without re-parsing
- Symbol usage tracking across the entire codebase
- Call graph analysis for impact assessment

### 🛠️ **Intelligent Refactoring Suggestions**
- **Function extraction** recommendations based on complexity analysis
- **Parameter grouping** suggestions using dependency patterns
- **Import optimization** using graph-sitter's import resolution

### 🎯 **Context-Aware Fixes**
- **Dependency injection** recommendations based on usage patterns
- **Circular dependency detection** using graph traversal
- **Dead code elimination** using reachability analysis

### 🚀 **Automated Code Transformations**
- **Safe renaming** using symbol usage analysis
- **Extract method** refactoring with dependency preservation
- **Import reorganization** with automatic conflict resolution

---

**Generated by**: Enhanced Codebase Analytics API v2.0.0  
**Analysis Duration**: 2.34 seconds  
**Graph-Sitter Version**: Latest  
**Timestamp**: 2025-07-11T04:47:00Z  

> 💡 **Pro Tip**: Use the `/codebase_analysis` endpoint for structured JSON data, or `/generate_report` for human-readable reports like this one!
