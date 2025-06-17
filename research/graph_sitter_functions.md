# Graph-Sitter Functions for Codebase Analysis

This document identifies functions and features from the graph-sitter tutorials that could be adapted for our codebase analysis system. These functions can be used with `from codegen import` instead of `from graph_sitter`.

## 1. Complexity Metrics

### 1.1 Cyclomatic Complexity

**Function**: `calculate_cyclomatic_complexity(function)`

**Description**: Calculates the cyclomatic complexity of a function by analyzing its control flow structure.

**Implementation Details**:
```python
def calculate_cyclomatic_complexity(function):
    def analyze_statement(statement):
        complexity = 0
        
        if isinstance(statement, IfBlockStatement):
            complexity += 1
            if hasattr(statement, "elif_statements"):
                complexity += len(statement.elif_statements)
        
        elif isinstance(statement, (ForLoopStatement, WhileStatement)):
            complexity += 1
        
        return complexity
```

**Calculation Method**:
- Base complexity of 1
- +1 for each if statement
- +1 for each elif statement
- +1 for each for loop
- +1 for each while loop
- +1 for each boolean operator (and, or) in conditions
- +1 for each except block in try-catch statements

**Adaptation for Our System**:
- Use the same calculation method
- Ensure compatibility with our codebase structure
- Add support for additional statement types if needed
- Integrate with our code quality analysis

### 1.2 Halstead Volume

**Function**: `calculate_halstead_volume(operators, operands)`

**Description**: Calculates the Halstead volume of code by analyzing operators and operands.

**Implementation Details**:
```python
def calculate_halstead_volume(operators, operands):
    n1 = len(set(operators))
    n2 = len(set(operands))
    
    N1 = len(operators)
    N2 = len(operands)
    
    N = N1 + N2
    n = n1 + n2
    
    if n > 0:
        volume = N * math.log2(n)
        return volume, N1, N2, n1, n2
    return 0, N1, N2, n1, n2
```

**Calculation Method**:
- n1 = number of unique operators
- n2 = number of unique operands
- N1 = total number of operators
- N2 = total number of operands
- N = N1 + N2
- n = n1 + n2
- Volume = N * log2(n)

**Adaptation for Our System**:
- Use the same calculation method
- Add a function to extract operators and operands from code
- Ensure compatibility with our codebase structure
- Integrate with our code quality analysis

### 1.3 Depth of Inheritance (DOI)

**Function**: `calculate_doi(cls)`

**Description**: Calculates the depth of inheritance for a class by counting the length of its superclasses list.

**Implementation Details**:
```python
def calculate_doi(cls):
    return len(cls.superclasses)
```

**Calculation Method**:
- Count the number of superclasses for a class

**Adaptation for Our System**:
- Use the same calculation method
- Ensure compatibility with our codebase structure
- Integrate with our architectural analysis

## 2. Maintainability Index

**Function**: `calculate_maintainability_index(halstead_volume, cyclomatic_complexity, loc)`

**Description**: Calculates the maintainability index for a function or module.

**Implementation Details**:
```python
def calculate_maintainability_index(halstead_volume, cyclomatic_complexity, loc):
    """Calculate the normalized maintainability index for a given function."""
    if loc <= 0:
        return 100
    
    try:
        raw_mi = (
            171
            - 5.2 * math.log(max(1, halstead_volume))
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(max(1, loc))
        )
        normalized_mi = max(0, min(100, raw_mi * 100 / 171))
        return int(normalized_mi)
    except (ValueError, TypeError):
        return 0
```

**Calculation Method**:
- MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(SLOC)
- Normalized MI = max(0, min(100, MI * 100 / 171))

**Adaptation for Our System**:
- Use the same calculation method
- Ensure compatibility with our codebase structure
- Integrate with our code quality analysis
- Add a function to get a qualitative rank for the maintainability index

## 3. Line Metrics

### 3.1 Lines of Code (LOC)

**Description**: Counts the total number of lines in the source code, including blank lines and comments.

**Adaptation for Our System**:
- Implement a function to count total lines in a file
- Ensure compatibility with our codebase structure
- Integrate with our code quality analysis

### 3.2 Logical Lines of Code (LLOC)

**Description**: Counts the number of lines of code that contain actual functional statements, excluding comments and blank lines.

**Adaptation for Our System**:
- Implement a function to count logical lines in a file
- Ensure compatibility with our codebase structure
- Integrate with our code quality analysis

### 3.3 Source Lines of Code (SLOC)

**Description**: Counts the number of lines containing actual code, excluding blank lines.

**Adaptation for Our System**:
- Implement a function to count source lines in a file
- Ensure compatibility with our codebase structure
- Integrate with our code quality analysis

### 3.4 Comment Density

**Description**: Calculates the proportion of comments in the codebase.

**Calculation Method**:
```
comment_density = (total_comments / total_loc * 100)
```

**Adaptation for Our System**:
- Implement a function to calculate comment density
- Ensure compatibility with our codebase structure
- Integrate with our code quality analysis

## 4. General Codebase Statistics

**Description**: Provides general statistics about the codebase, such as the number of files, functions, and classes.

**Implementation Details**:
```python
num_files = len(codebase.files(extensions="*"))
num_functions = len(codebase.functions)
num_classes = len(codebase.classes)
```

**Adaptation for Our System**:
- Use the same calculation method
- Ensure compatibility with our codebase structure
- Integrate with our analysis summary

## 5. Commit Activity

**Description**: Analyzes the commit history of the repository to track activity over time.

**Adaptation for Our System**:
- Implement a function to analyze commit history
- Count commits by month for the last 12 months
- Ensure compatibility with our codebase structure
- Integrate with our analysis summary

## 6. Neo4j Graph Visualization

**Description**: Exports codebase graphs to Neo4j for visualization and analysis.

**Implementation Details**:
```python
from graph_sitter import Codebase
from graph_sitter.extensions.graph.main import visualize_codebase

# parse codebase
codebase = Codebase("path/to/codebase")

# export to Neo4j
visualize_codebase(codebase, "bolt://localhost:7687", "neo4j", "password")
```

**Visualization Queries**:
- Class Hierarchy: `Match (s: Class )-[r: INHERITS_FROM*]-> (e:Class) RETURN s, e LIMIT 10`
- Methods Defined by Each Class: `Match (s: Class )-[r: DEFINES]-> (e:Method) RETURN s, e LIMIT 10`
- Function Calls: `Match (s: Func )-[r: CALLS]-> (e:Func) RETURN s, e LIMIT 10`
- Call Graph: `Match path = (:(Method|Func)) -[:CALLS*5..10]-> (:(Method|Func)) Return path LIMIT 20`

**Adaptation for Our System**:
- Implement a function to export codebase graphs to Neo4j
- Ensure compatibility with our codebase structure
- Integrate with our visualization module
- Provide pre-defined visualization queries

## 7. Deep Code Research Tools

**Description**: Tools for analyzing and explaining codebases using Codegen and LangChain.

**Implementation Details**:
```python
from graph_sitter import Codebase
from graph_sitter.extensions.langchain.agent import create_agent_with_tools
from graph_sitter.extensions.langchain.tools import (
    ListDirectoryTool,
    RevealSymbolTool,
    SearchTool,
    SemanticSearchTool,
    ViewFileTool,
)
```

**Tools**:
- `ViewFileTool`: Read and understand file contents
- `ListDirectoryTool`: Explore the codebase structure
- `SearchTool`: Find specific code patterns
- `SemanticSearchTool`: Search using natural language
- `RevealSymbolTool`: Analyze dependencies and usages

**Adaptation for Our System**:
- Implement similar tools for our codebase analysis
- Ensure compatibility with our codebase structure
- Integrate with our API module
- Consider adding LLM-powered analysis capabilities

## Summary of Adaptable Functions

1. **Complexity Metrics**:
   - `calculate_cyclomatic_complexity(function)`
   - `calculate_halstead_volume(operators, operands)`
   - `calculate_doi(cls)`

2. **Maintainability Index**:
   - `calculate_maintainability_index(halstead_volume, cyclomatic_complexity, loc)`

3. **Line Metrics**:
   - Functions for counting LOC, LLOC, SLOC
   - Function for calculating comment density

4. **General Codebase Statistics**:
   - Functions for counting files, functions, classes

5. **Commit Activity**:
   - Function for analyzing commit history

6. **Neo4j Graph Visualization**:
   - Function for exporting codebase graphs to Neo4j
   - Pre-defined visualization queries

7. **Deep Code Research Tools**:
   - Tools for exploring and analyzing codebases
   - LLM-powered analysis capabilities

These functions and features can be adapted for our codebase analysis system to provide comprehensive analysis capabilities.

