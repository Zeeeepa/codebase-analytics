#!/usr/bin/env python3
"""
Codemod to consolidate comprehensive_analysis.py, api.py, and analyzer.py
into a single unified API file using graph_sitter for AST manipulation.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import textwrap

# Try to import graph_sitter components
try:
    from graph_sitter.python import PythonParser
    from graph_sitter.core.interfaces.editable import Editable
    GRAPH_SITTER_AVAILABLE = True
except ImportError:
    GRAPH_SITTER_AVAILABLE = False
    print("Warning: graph_sitter not fully available, falling back to AST-based approach")

@dataclass
class FunctionInfo:
    """Information about a function to be consolidated."""
    name: str
    source_file: str
    ast_node: ast.FunctionDef
    dependencies: Set[str]
    is_duplicate: bool = False
    merge_priority: int = 0  # Higher priority functions take precedence

@dataclass
class ClassInfo:
    """Information about a class to be consolidated."""
    name: str
    source_file: str
    ast_node: ast.ClassDef
    dependencies: Set[str]
    is_duplicate: bool = False
    merge_priority: int = 0

@dataclass
class ImportInfo:
    """Information about imports to be consolidated."""
    module: str
    names: List[str]
    alias: Optional[str]
    source_file: str
    is_from_import: bool

class CodeConsolidator:
    """
    Consolidates three Python files into a single unified API file.
    Uses AST parsing and manipulation to merge functions, classes, and imports.
    """
    
    def __init__(self, files_to_merge: List[str], output_file: str):
        self.files_to_merge = files_to_merge
        self.output_file = output_file
        self.functions: Dict[str, FunctionInfo] = {}
        self.classes: Dict[str, ClassInfo] = {}
        self.imports: Dict[str, ImportInfo] = {}
        self.global_variables: Dict[str, Any] = {}
        self.file_asts: Dict[str, ast.Module] = {}
        
    def parse_files(self):
        """Parse all input files and extract their AST structures."""
        print("🔍 Parsing input files...")
        
        for file_path in self.files_to_merge:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
                
            print(f"  📄 Parsing {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            try:
                tree = ast.parse(content)
                self.file_asts[file_path] = tree
                self._extract_elements(tree, file_path)
            except SyntaxError as e:
                print(f"❌ Syntax error in {file_path}: {e}")
                continue
                
    def _extract_elements(self, tree: ast.Module, file_path: str):
        """Extract functions, classes, and imports from an AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._process_function(node, file_path)
            elif isinstance(node, ast.ClassDef):
                self._process_class(node, file_path)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._process_import(node, file_path)
                
    def _process_function(self, node: ast.FunctionDef, file_path: str):
        """Process a function definition."""
        func_name = node.name
        
        # Calculate dependencies (simplified - looks for function calls)
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                dependencies.add(child.func.id)
                
        # Determine merge priority based on file and function characteristics
        priority = self._calculate_function_priority(node, file_path)
        
        func_info = FunctionInfo(
            name=func_name,
            source_file=file_path,
            ast_node=node,
            dependencies=dependencies,
            merge_priority=priority
        )
        
        # Handle duplicates
        if func_name in self.functions:
            existing = self.functions[func_name]
            if priority > existing.merge_priority:
                existing.is_duplicate = True
                self.functions[func_name] = func_info
            else:
                func_info.is_duplicate = True
        else:
            self.functions[func_name] = func_info
            
    def _process_class(self, node: ast.ClassDef, file_path: str):
        """Process a class definition."""
        class_name = node.name
        
        # Calculate dependencies
        dependencies = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
                
        priority = self._calculate_class_priority(node, file_path)
        
        class_info = ClassInfo(
            name=class_name,
            source_file=file_path,
            ast_node=node,
            dependencies=dependencies,
            merge_priority=priority
        )
        
        # Handle duplicates
        if class_name in self.classes:
            existing = self.classes[class_name]
            if priority > existing.merge_priority:
                existing.is_duplicate = True
                self.classes[class_name] = class_info
            else:
                class_info.is_duplicate = True
        else:
            self.classes[class_name] = class_info
            
    def _process_import(self, node: ast.Import | ast.ImportFrom, file_path: str):
        """Process import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = ImportInfo(
                    module=alias.name,
                    names=[alias.name],
                    alias=alias.asname,
                    source_file=file_path,
                    is_from_import=False
                )
                self.imports[f"{alias.name}_{file_path}"] = import_info
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            import_info = ImportInfo(
                module=node.module or "",
                names=names,
                alias=None,
                source_file=file_path,
                is_from_import=True
            )
            key = f"{node.module}_{','.join(names)}_{file_path}"
            self.imports[key] = import_info
            
    def _calculate_function_priority(self, node: ast.FunctionDef, file_path: str) -> int:
        """Calculate merge priority for a function."""
        priority = 0
        
        # Priority based on file (analyzer.py > comprehensive_analysis.py > api.py)
        if "analyzer.py" in file_path:
            priority += 100
        elif "comprehensive_analysis.py" in file_path:
            priority += 50
        elif "api.py" in file_path:
            priority += 25
            
        # Priority based on function characteristics
        if node.decorator_list:  # Has decorators (likely API endpoints)
            priority += 20
        if len(node.body) > 10:  # Complex function
            priority += 10
        if node.name.startswith("_"):  # Private function
            priority -= 5
            
        return priority
        
    def _calculate_class_priority(self, node: ast.ClassDef, file_path: str) -> int:
        """Calculate merge priority for a class."""
        priority = 0
        
        # Priority based on file
        if "analyzer.py" in file_path:
            priority += 100
        elif "comprehensive_analysis.py" in file_path:
            priority += 50
        elif "api.py" in file_path:
            priority += 25
            
        # Priority based on class characteristics
        if len(node.body) > 5:  # Complex class
            priority += 10
        if any(isinstance(item, ast.FunctionDef) and item.name == "__init__" for item in node.body):
            priority += 5  # Has constructor
            
        return priority
        
    def resolve_conflicts(self):
        """Resolve naming conflicts and dependencies."""
        print("🔧 Resolving conflicts and dependencies...")
        
        # Mark duplicates
        for func_name, func_info in self.functions.items():
            if func_info.is_duplicate:
                print(f"  ⚠️  Duplicate function '{func_name}' from {func_info.source_file} (lower priority)")
                
        for class_name, class_info in self.classes.items():
            if class_info.is_duplicate:
                print(f"  ⚠️  Duplicate class '{class_name}' from {class_info.source_file} (lower priority)")
                
    def merge_imports(self) -> List[str]:
        """Merge and deduplicate imports."""
        print("📦 Merging imports...")
        
        # Group imports by module
        import_groups: Dict[str, Set[str]] = {}
        from_imports: Dict[str, Set[str]] = {}
        
        for import_info in self.imports.values():
            if import_info.is_from_import:
                module = import_info.module
                if module not in from_imports:
                    from_imports[module] = set()
                from_imports[module].update(import_info.names)
            else:
                for name in import_info.names:
                    if name not in import_groups:
                        import_groups[name] = set()
                    import_groups[name].add(name)
                    
        # Generate import statements
        import_lines = []
        
        # Standard library imports first
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'logging', 'argparse', 
            'pathlib', 'tempfile', 'subprocess', 'math', 're', 'collections',
            'typing', 'dataclasses', 'enum', 'abc'
        }
        
        # Regular imports
        for module in sorted(import_groups.keys()):
            if any(mod in module for mod in stdlib_modules):
                import_lines.append(f"import {module}")
                
        # Third-party imports
        for module in sorted(import_groups.keys()):
            if not any(mod in module for mod in stdlib_modules) and not module.startswith('.'):
                import_lines.append(f"import {module}")
                
        # From imports
        for module, names in sorted(from_imports.items()):
            if module:  # Skip relative imports for now
                sorted_names = sorted(names)
                if len(sorted_names) > 5:
                    # Multi-line import for many names
                    import_lines.append(f"from {module} import (")
                    for i, name in enumerate(sorted_names):
                        comma = "," if i < len(sorted_names) - 1 else ""
                        import_lines.append(f"    {name}{comma}")
                    import_lines.append(")")
                else:
                    import_lines.append(f"from {module} import {', '.join(sorted_names)}")
                    
        return import_lines
        
    def generate_unified_file(self):
        """Generate the unified API file."""
        print("🚀 Generating unified API file...")
        
        # Prepare the output content
        output_lines = []
        
        # File header
        output_lines.extend([
            '#!/usr/bin/env python3',
            '"""',
            'Unified Codebase Analytics API',
            '',
            'This file consolidates functionality from:',
            '- comprehensive_analysis.py: Deep codebase analysis with Codegen SDK',
            '- api.py: FastAPI web service with metrics calculation',  
            '- analyzer.py: Advanced issue management and transaction support',
            '',
            'Generated by consolidation_codemod.py',
            '"""',
            ''
        ])
        
        # Add imports
        import_lines = self.merge_imports()
        output_lines.extend(import_lines)
        output_lines.append('')
        
        # Add classes (non-duplicates only)
        output_lines.append('# =' * 40)
        output_lines.append('# CONSOLIDATED CLASSES')
        output_lines.append('# =' * 40)
        output_lines.append('')
        
        for class_name, class_info in self.classes.items():
            if not class_info.is_duplicate:
                output_lines.append(f"# From {class_info.source_file}")
                class_source = ast.unparse(class_info.ast_node)
                output_lines.append(class_source)
                output_lines.append('')
                
        # Add functions (non-duplicates only)
        output_lines.append('# =' * 40)
        output_lines.append('# CONSOLIDATED FUNCTIONS')
        output_lines.append('# =' * 40)
        output_lines.append('')
        
        for func_name, func_info in self.functions.items():
            if not func_info.is_duplicate:
                output_lines.append(f"# From {func_info.source_file}")
                func_source = ast.unparse(func_info.ast_node)
                output_lines.append(func_source)
                output_lines.append('')
                
        # Add main execution block
        output_lines.extend([
            '',
            'if __name__ == "__main__":',
            '    import uvicorn',
            '    import socket',
            '    ',
            '    def find_available_port(start_port=8000, max_port=8100):',
            '        """Find an available port starting from start_port"""',
            '        for port in range(start_port, max_port):',
            '            try:',
            '                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:',
            '                    s.bind(("0.0.0.0", port))',
            '                    return port',
            '            except OSError:',
            '                continue',
            '        raise RuntimeError(f"No available ports found between {start_port} and {max_port}")',
            '    ',
            '    # Find an available port',
            '    port = find_available_port()',
            '    print(f"🚀 Starting Unified Codebase Analytics API on http://localhost:{port}")',
            '    print(f"📚 API documentation available at http://localhost:{port}/docs")',
            '    ',
            '    # Initialize FastAPI app if available',
            '    try:',
            '        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)',
            '    except NameError:',
            '        print("❌ FastAPI app not found. Please check the consolidation.")',
            '        sys.exit(1)'
        ])
        
        # Write the output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            
        print(f"✅ Unified file generated: {self.output_file}")
        
    def generate_report(self):
        """Generate a consolidation report."""
        report_file = self.output_file.replace('.py', '_consolidation_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Codebase Consolidation Report\n\n")
            f.write(f"Generated: {os.path.basename(self.output_file)}\n\n")
            
            f.write("## Source Files\n")
            for file_path in self.files_to_merge:
                f.write(f"- {file_path}\n")
            f.write("\n")
            
            f.write("## Consolidated Elements\n\n")
            
            f.write("### Classes\n")
            for class_name, class_info in self.classes.items():
                status = "❌ DUPLICATE" if class_info.is_duplicate else "✅ INCLUDED"
                f.write(f"- `{class_name}` from {class_info.source_file} - {status}\n")
            f.write("\n")
            
            f.write("### Functions\n")
            for func_name, func_info in self.functions.items():
                status = "❌ DUPLICATE" if func_info.is_duplicate else "✅ INCLUDED"
                f.write(f"- `{func_name}` from {func_info.source_file} - {status}\n")
            f.write("\n")
            
            f.write("### Import Summary\n")
            f.write(f"Total unique imports: {len(self.imports)}\n\n")
            
            f.write("## Recommendations\n")
            f.write("1. Review duplicate functions/classes for potential feature merging\n")
            f.write("2. Test the unified API thoroughly\n")
            f.write("3. Update any external references to the original files\n")
            f.write("4. Consider refactoring for better organization\n")
            
        print(f"📊 Consolidation report generated: {report_file}")
        
    def consolidate(self):
        """Main consolidation process."""
        print("🔄 Starting code consolidation process...")
        
        # Step 1: Parse all files
        self.parse_files()
        
        # Step 2: Resolve conflicts
        self.resolve_conflicts()
        
        # Step 3: Generate unified file
        self.generate_unified_file()
        
        # Step 4: Generate report
        self.generate_report()
        
        print("✅ Consolidation complete!")
        
        # Print summary
        total_functions = len(self.functions)
        included_functions = sum(1 for f in self.functions.values() if not f.is_duplicate)
        total_classes = len(self.classes)
        included_classes = sum(1 for c in self.classes.values() if not c.is_duplicate)
        
        print(f"\n📈 Summary:")
        print(f"  Functions: {included_functions}/{total_functions} included")
        print(f"  Classes: {included_classes}/{total_classes} included")
        print(f"  Imports: {len(self.imports)} processed")


def main():
    """Main entry point for the consolidation codemod."""
    files_to_merge = [
        "backend/comprehensive_analysis.py",
        "backend/api.py", 
        "backend/analyzer.py"
    ]
    
    output_file = "backend/unified_api.py"
    
    # Check if all input files exist
    missing_files = [f for f in files_to_merge if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return 1
        
    # Create consolidator and run
    consolidator = CodeConsolidator(files_to_merge, output_file)
    consolidator.consolidate()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

