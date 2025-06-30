#!/usr/bin/env python3
"""
Comprehensive Codebase Analyzer
This script provides a full analysis of a codebase using the Codegen SDK.

Usage:
  python comprehensive_analysis.py --repo [REPO_URL_OR_PATH]

The analysis includes:
- Unused functions, classes, imports
- Parameter issues (unused, mismatches)
- Type annotation issues
- Circular dependencies
- Implementation errors
- And more
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional, Union

# Import essential summary functions from codebase_analysis
from codegen_on_oss.analyzers.codebase_analysis import (
    get_codebase_summary,
    get_file_summary,
    get_class_summary,
    get_function_summary,
    get_symbol_summary
)

# Set PYTHONPATH for Codegen SDK
codegen_path = "/home/l/codegen"
if codegen_path not in sys.path:
    sys.path.insert(0, codegen_path)
os.environ["PYTHONPATH"] = f"{codegen_path}:{os.environ.get('PYTHONPATH', '')}"

# Import Codegen SDK - this is required, no fallbacks
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.core.import_resolution import Import
from codegen.sdk.enums import EdgeType, SymbolType

class IssueType:
    """Types of issues that can be detected."""
    UNUSED_FUNCTION = "Unused function"
    UNUSED_CLASS = "Unused class"
    UNUSED_IMPORT = "Unused import"
    UNUSED_PARAMETER = "Unused parameter"
    PARAMETER_MISMATCH = "Parameter mismatch"
    MISSING_TYPE_ANNOTATION = "Missing type annotation"
    CIRCULAR_DEPENDENCY = "Circular dependency"
    IMPLEMENTATION_ERROR = "Implementation error"
    EMPTY_FUNCTION = "Empty function"
    UNREACHABLE_CODE = "Unreachable code"

class IssueSeverity:
    """Severity levels for issues."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class Issue:
    """Represents an issue found during codebase analysis."""
    def __init__(self, 
                 item: Any,
                 issue_type: str,
                 message: str,
                 severity: str = IssueSeverity.WARNING,
                 location: Optional[str] = None,
                 suggestion: Optional[str] = None):
        self.item = item
        self.type = issue_type
        self.message = message
        self.severity = severity
        self.location = self._get_location(item) if location is None else location
        self.suggestion = suggestion
        
    def _get_location(self, item: Any) -> str:
        """Get a string representation of the item's location."""
        if hasattr(item, 'file') and hasattr(item.file, 'path'):
            file_path = item.file.path
            if hasattr(item, 'line'):
                return f"{file_path}:{item.line}"
            return file_path
        elif hasattr(item, 'path'):
            return item.path
        else:
            return "Unknown location"
    
    def __str__(self) -> str:
        """Return a string representation of the issue."""
        base = f"[{self.severity.upper()}] {self.type}: {self.message} ({self.location})"
        if self.suggestion:
            base += f" - Suggestion: {self.suggestion}"
        return base

class ComprehensiveAnalyzer:
    """
    Comprehensive analyzer for codebases using the Codegen SDK.
    Implements deep analysis of code issues, dependencies, and metrics.
    """
    
    def __init__(self, repo_path_or_url: str):
        """
        Initialize the analyzer with a repository path or URL.
        
        Args:
            repo_path_or_url: Path to local repo or URL to GitHub repo
        """
        self.repo_path_or_url = repo_path_or_url
        self.issues: List[Issue] = []
        self.start_time = time.time()
        self.codebase = None
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the codebase.
        
        Returns:
            Dictionary with analysis results
        """
        print(f"Starting comprehensive analysis of {self.repo_path_or_url}...")
        
        # Initialize codebase
        try:
            print(f"Initializing codebase from {self.repo_path_or_url}")
            if self.repo_path_or_url.startswith(("http://", "https://")):
                # Extract repo name for GitHub URLs
                parts = self.repo_path_or_url.rstrip('/').split('/')
                repo_name = f"{parts[-2]}/{parts[-1]}"
                try:
                    self.codebase = Codebase.from_repo(repo_full_name=repo_name)
                    print(f"Successfully initialized codebase from GitHub repository: {repo_name}")
                except Exception as e:
                    print(f"Error initializing codebase from GitHub: {e}")
                    self.issues.append(Issue(
                        self.repo_path_or_url,
                        "Initialization Error",
                        f"Failed to initialize codebase from GitHub: {e}",
                        IssueSeverity.ERROR,
                        suggestion="Check your network connection and GitHub access permissions."
                    ))
                    return {
                        "error": f"Failed to initialize codebase: {str(e)}",
                        "success": False
                    }
            else:
                # Local path
                try:
                    self.codebase = Codebase(self.repo_path_or_url)
                    print(f"Successfully initialized codebase from local path: {self.repo_path_or_url}")
                except Exception as e:
                    print(f"Error initializing codebase from local path: {e}")
                    self.issues.append(Issue(
                        self.repo_path_or_url,
                        "Initialization Error",
                        f"Failed to initialize codebase from local path: {e}",
                        IssueSeverity.ERROR,
                        suggestion="Ensure the path exists and contains valid source code."
                    ))
                    return {
                        "error": f"Failed to initialize codebase: {str(e)}",
                        "success": False
                    }
            
            # Check if codebase was initialized correctly
            if not hasattr(self.codebase, 'files') or not self.codebase.files:
                self.issues.append(Issue(
                    self.repo_path_or_url,
                    "Empty Codebase",
                    "Codebase was initialized but contains no files",
                    IssueSeverity.ERROR,
                    suggestion="Check if the repository contains supported language files."
                ))
                print("Warning: Codebase contains no files")
            
            # Run all analyses - with error handling for each step
            try:
                self._analyze_dead_code()
            except Exception as e:
                print(f"Error in dead code analysis: {e}")
                self.issues.append(Issue(
                    self.repo_path_or_url, 
                    "Analysis Error", 
                    f"Dead code analysis failed: {e}",
                    IssueSeverity.ERROR
                ))
                
            try:
                self._analyze_parameter_issues()
            except Exception as e:
                print(f"Error in parameter analysis: {e}")
                self.issues.append(Issue(
                    self.repo_path_or_url, 
                    "Analysis Error", 
                    f"Parameter analysis failed: {e}",
                    IssueSeverity.ERROR
                ))
                
            try:
                self._analyze_type_annotations()
            except Exception as e:
                print(f"Error in type annotation analysis: {e}")
                self.issues.append(Issue(
                    self.repo_path_or_url, 
                    "Analysis Error", 
                    f"Type annotation analysis failed: {e}",
                    IssueSeverity.ERROR
                ))
                
            try:
                self._analyze_circular_dependencies()
            except Exception as e:
                print(f"Error in circular dependency analysis: {e}")
                self.issues.append(Issue(
                    self.repo_path_or_url, 
                    "Analysis Error", 
                    f"Circular dependency analysis failed: {e}",
                    IssueSeverity.ERROR
                ))
                
            try:
                self._analyze_implementation_issues()
            except Exception as e:
                print(f"Error in implementation issue analysis: {e}")
                self.issues.append(Issue(
                    self.repo_path_or_url, 
                    "Analysis Error", 
                    f"Implementation issue analysis failed: {e}",
                    IssueSeverity.ERROR
                ))
            
            # Generate final report
            return self._generate_report()
            
        except Exception as e:
            print(f"Error analyzing codebase: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "success": False
            }
    
    def _analyze_dead_code(self):
        """Find and log unused code (functions, classes, imports)."""
        # Find unused functions
        for func in self.codebase.functions:
            if not func.usages:
                self.issues.append(Issue(
                    func, 
                    IssueType.UNUSED_FUNCTION,
                    f"Unused function: {func.name}",
                    IssueSeverity.WARNING,
                    suggestion="Consider removing this unused function or documenting why it's needed"
                ))
        
        # Find unused classes
        for cls in self.codebase.classes:
            if not cls.usages:
                self.issues.append(Issue(
                    cls,
                    IssueType.UNUSED_CLASS,
                    f"Unused class: {cls.name}",
                    IssueSeverity.WARNING,
                    suggestion="Consider removing this unused class or documenting why it's needed"
                ))
        
        # Find unused imports
        for file in self.codebase.files:
            for imp in file.imports:
                if not imp.usages:
                    self.issues.append(Issue(
                        imp,
                        IssueType.UNUSED_IMPORT,
                        f"Unused import: {imp.source if hasattr(imp, 'source') else str(imp)}",
                        IssueSeverity.INFO,
                        suggestion="Remove this unused import"
                    ))
    
    def _analyze_parameter_issues(self):
        """Find and log parameter issues (unused, mismatches)."""
        for func in self.codebase.functions:
            # Check for unused parameters
            for param in func.parameters:
                # Skip 'self' in methods
                if param.name == 'self' and func.is_method:
                    continue
                    
                # Check if parameter is used in function body
                param_dependencies = [dep.name for dep in func.dependencies if hasattr(dep, 'name')]
                if param.name not in param_dependencies:
                    self.issues.append(Issue(
                        func,
                        IssueType.UNUSED_PARAMETER,
                        f"Function '{func.name}' has unused parameter: {param.name}",
                        IssueSeverity.INFO,
                        suggestion=f"Consider removing the unused parameter '{param.name}' if it's not needed"
                    ))
            
            # Check call sites for parameter mismatches
            for call in func.call_sites:
                if hasattr(call, 'args') and hasattr(func, 'parameters'):
                    expected_params = set(p.name for p in func.parameters if not p.is_optional and p.name != 'self')
                    actual_params = set()
                    
                    # Extract parameter names from call arguments
                    if hasattr(call, 'args'):
                        for arg in call.args:
                            if hasattr(arg, 'parameter_name') and arg.parameter_name:
                                actual_params.add(arg.parameter_name)
                    
                    # Find missing parameters
                    missing = expected_params - actual_params
                    if missing:
                        # Skip if function has **kwargs
                        has_kwargs = any(p.name.startswith('**') for p in func.parameters)
                        if not has_kwargs:
                            self.issues.append(Issue(
                                call,
                                IssueType.PARAMETER_MISMATCH,
                                f"Call to '{func.name}' is missing parameters: {', '.join(missing)}",
                                IssueSeverity.ERROR,
                                suggestion="Add the missing parameters to the function call"
                            ))
    
    def _analyze_type_annotations(self):
        """Find and log missing type annotations."""
        for func in self.codebase.functions:
            # Skip if function is in a type-annotated file
            file_path = str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else ''
            if any(file_ext in file_path for file_ext in ['.pyi']):
                continue
                
            # Check return type
            if not func.return_type and not func.name.startswith('__'):
                self.issues.append(Issue(
                    func,
                    IssueType.MISSING_TYPE_ANNOTATION,
                    f"Function '{func.name}' is missing return type annotation",
                    IssueSeverity.INFO,
                    suggestion="Add a return type annotation to improve type safety"
                ))
            
            # Check parameter types
            params_without_type = [p.name for p in func.parameters 
                                 if not p.type and p.name != 'self' and not p.name.startswith('*')]
            if params_without_type:
                self.issues.append(Issue(
                    func,
                    IssueType.MISSING_TYPE_ANNOTATION,
                    f"Function '{func.name}' has parameters without type annotations: {', '.join(params_without_type)}",
                    IssueSeverity.INFO,
                    suggestion="Add type annotations to all parameters"
                ))
    
    def _analyze_circular_dependencies(self):
        """Find and log circular dependencies."""
        circular_deps = {}
        
        # Basic implementation to detect file-level circular dependencies
        for file in self.codebase.files:
            visited = set()
            path = []
            self._check_circular_deps(file, visited, path, circular_deps)
        
        # Log circular dependencies
        for file_path, cycles in circular_deps.items():
            for cycle in cycles:
                cycle_str = " -> ".join([f.path for f in cycle])
                self.issues.append(Issue(
                    file_path,
                    IssueType.CIRCULAR_DEPENDENCY,
                    f"Circular dependency detected: {cycle_str}",
                    IssueSeverity.ERROR,
                    suggestion="Refactor the code to break the circular dependency"
                ))
    
    def _check_circular_deps(self, file, visited, path, circular_deps):
        """Helper method to check for circular dependencies using DFS."""
        if file in path:
            # Found a cycle
            cycle = path[path.index(file):] + [file]
            if file.path not in circular_deps:
                circular_deps[file.path] = []
            circular_deps[file.path].append(cycle)
            return
            
        if file in visited:
            return
            
        visited.add(file)
        path.append(file)
        
        # Check imports
        for imp in file.imports:
            if hasattr(imp, 'resolved_module') and imp.resolved_module:
                self._check_circular_deps(imp.resolved_module, visited, path.copy(), circular_deps)
        
        path.pop()
    
    def _analyze_implementation_issues(self):
        """Find and log implementation issues (empty functions, etc.)."""
        for func in self.codebase.functions:
            # Skip dunder methods and abstract methods
            if func.name.startswith('__') and func.name.endswith('__'):
                continue
            
            # Check for empty function bodies
            if not func.body or not func.body.strip():
                # Skip if it's a method overridden from parent class
                is_override = False
                if hasattr(func, 'parent') and isinstance(func.parent, Class):
                    for parent_class in func.parent.parents:
                        if any(m.name == func.name for m in parent_class.methods):
                            is_override = True
                            break
                
                if not is_override:
                    self.issues.append(Issue(
                        func,
                        IssueType.EMPTY_FUNCTION,
                        f"Function '{func.name}' has an empty body",
                        IssueSeverity.WARNING,
                        suggestion="Implement the function or remove it if it's not needed"
                    ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the analysis results.
        
        Returns:
            Dictionary with analysis results
        """
        analysis_duration = time.time() - self.start_time
        
        # Get statistics
        stats = {
            "total_files": len(list(self.codebase.files)),
            "total_functions": len(list(self.codebase.functions)),
            "total_classes": len(list(self.codebase.classes)),
            "total_imports": len(list(self.codebase.imports)),
            "total_issues": len(self.issues),
            "analysis_duration": analysis_duration
        }
        
        # Get issues by severity
        issues_by_severity = {}
        for issue in self.issues:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)
            
        # Get issues by type
        issues_by_type = {}
        for issue in self.issues:
            if issue.type not in issues_by_type:
                issues_by_type[issue.type] = []
            issues_by_type[issue.type].append(issue)
        
        # Prepare report
        report = {
            "success": True,
            "repo": self.repo_path_or_url,
            "timestamp": datetime.now().isoformat(),
            "duration": analysis_duration,
            "stats": stats,
            "issues": {
                "all": [str(issue) for issue in self.issues],
                "by_severity": {sev: [str(issue) for issue in issues] 
                                for sev, issues in issues_by_severity.items()},
                "by_type": {type_: [str(issue) for issue in issues] 
                            for type_, issues in issues_by_type.items()}
            },
            # Detailed counts
            "counts": {
                "unused_functions": len(issues_by_type.get(IssueType.UNUSED_FUNCTION, [])),
                "unused_classes": len(issues_by_type.get(IssueType.UNUSED_CLASS, [])),
                "unused_imports": len(issues_by_type.get(IssueType.UNUSED_IMPORT, [])),
                "parameter_issues": len(issues_by_type.get(IssueType.UNUSED_PARAMETER, [])) + 
                                   len(issues_by_type.get(IssueType.PARAMETER_MISMATCH, [])),
                "type_annotation_issues": len(issues_by_type.get(IssueType.MISSING_TYPE_ANNOTATION, [])),
                "circular_dependencies": len(issues_by_type.get(IssueType.CIRCULAR_DEPENDENCY, [])),
                "implementation_issues": len(issues_by_type.get(IssueType.IMPLEMENTATION_ERROR, [])) +
                                        len(issues_by_type.get(IssueType.EMPTY_FUNCTION, []))
            }
        }
        
        # Print the report to console
        self._print_report(report)
        
        # Save report to file
        self._save_report(report)
        
        return report
    
    def _print_report(self, report: Dict[str, Any]):
        """Print the analysis report to the console."""
        print("\n" + "=" * 80)
        print(f"🔍 COMPREHENSIVE CODEBASE ANALYSIS: {self.repo_path_or_url}")
        print("=" * 80)
        print(f"Analysis Time: {report['timestamp']}")
        print(f"Analysis Duration: {report['duration']:.2f} seconds")
        
        # Print stats
        print("\n" + "-" * 40)
        print("CODEBASE STATISTICS:")
        print("-" * 40)
        print(f"Total Files: {report['stats']['total_files']}")
        print(f"Total Functions: {report['stats']['total_functions']}")
        print(f"Total Classes: {report['stats']['total_classes']}")
        print(f"Total Imports: {report['stats']['total_imports']}")
        print(f"Total Issues: {report['stats']['total_issues']}")
        
        # Print issue counts by type
        print("\n" + "-" * 40)
        print("ISSUES BY TYPE:")
        print("-" * 40)
        for issue_type, count in report['counts'].items():
            print(f"{issue_type.replace('_', ' ').title()}: {count}")
        
        # Print issue counts by severity
        print("\n" + "-" * 40)
        print("ISSUES BY SEVERITY:")
        print("-" * 40)
        for severity, issues in report['issues']['by_severity'].items():
            print(f"{severity.upper()}: {len(issues)}")
        
        # Print top issues
        print("\n" + "-" * 40)
        print("TOP ISSUES:")
        print("-" * 40)
        for i, issue in enumerate(report['issues']['all'][:10], 1):
            print(f"{i}. {issue}")
        
        if len(report['issues']['all']) > 10:
            print(f"... and {len(report['issues']['all']) - 10} more issues")
    
    def _save_report(self, report: Dict[str, Any]):
        """Save the analysis report to a file."""
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_analysis_{timestamp}.txt"
        detailed_filename = f"detailed_analysis_{timestamp}.txt"
        
        print(f"\nSaving full analysis to {filename}...")
        with open(filename, "w") as f:
            f.write(f"COMPREHENSIVE CODEBASE ANALYSIS: {self.repo_path_or_url}\n")
            f.write(f"Analysis Time: {report['timestamp']}\n")
            f.write(f"Analysis Duration: {report['duration']:.2f} seconds\n\n")
            
            # Write stats
            f.write("CODEBASE STATISTICS:\n")
            f.write(f"Total Files: {report['stats']['total_files']}\n")
            f.write(f"Total Functions: {report['stats']['total_functions']}\n")
            f.write(f"Total Classes: {report['stats']['total_classes']}\n")
            f.write(f"Total Imports: {report['stats']['total_imports']}\n")
            f.write(f"Total Issues: {report['stats']['total_issues']}\n\n")
            
            # Write issue counts by type
            f.write("ISSUES BY TYPE:\n")
            for issue_type, count in report['counts'].items():
                f.write(f"{issue_type.replace('_', ' ').title()}: {count}\n")
            f.write("\n")
            
            # Write issue counts by severity
            f.write("ISSUES BY SEVERITY:\n")
            for severity, issues in report['issues']['by_severity'].items():
                f.write(f"{severity.upper()}: {len(issues)}\n")
            f.write("\n")
            
            # Write all issues
            f.write("ALL ISSUES:\n")
            for i, issue in enumerate(report['issues']['all'], 1):
                f.write(f"{i}. {issue}\n")
            
            # Add note about detailed summaries
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("NOTE: Detailed summaries of codebase elements are available in a separate file\n")
            f.write(f"See: {detailed_filename}\n")
            f.write("=" * 80 + "\n")
        
        # Save detailed summaries to a separate file
        try:
            print(f"Saving detailed analysis to {detailed_filename}...")
            self._save_detailed_summaries(detailed_filename)
            print(f"Detailed summaries saved to {detailed_filename}")
        except Exception as e:
            print(f"Error saving detailed summaries: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Analysis results saved to {filename}")
    
    def _save_detailed_summaries(self, filename: str):
        """Save detailed summaries of the codebase, files, classes, and functions."""
        with open(filename, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(" " * 20 + "COMPREHENSIVE CODEBASE ANALYSIS DETAILS\n")
            f.write(" " * 20 + f"Repository: {self.repo_path_or_url}\n")
            f.write(" " * 20 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
            # Write codebase summary
            try:
                f.write("=" * 80 + "\n")
                f.write(" " * 30 + "CODEBASE SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(get_codebase_summary(self.codebase))
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error generating codebase summary: {str(e)}\n\n")
            
            # Write file summaries
            try:
                f.write("=" * 80 + "\n")
                f.write(" " * 30 + "FILE SUMMARIES\n")
                f.write("=" * 80 + "\n")
                files = list(self.codebase.files)
                for i, file in enumerate(files):
                    f.write(f"File {i+1}/{len(files)}: {getattr(file, 'path', 'Unknown path')}\n")
                    f.write("-" * 60 + "\n")
                    try:
                        f.write(get_file_summary(file))
                    except Exception as e:
                        f.write(f"Error generating file summary: {str(e)}\n")
                    f.write("\n\n")
            except Exception as e:
                f.write(f"Error processing file summaries: {str(e)}\n\n")
            
            # Write class summaries
            try:
                f.write("=" * 80 + "\n")
                f.write(" " * 30 + "CLASS SUMMARIES\n")
                f.write("=" * 80 + "\n")
                classes = list(self.codebase.classes)
                if classes:
                    for i, cls in enumerate(classes):
                        f.write(f"Class {i+1}/{len(classes)}: {getattr(cls, 'name', 'Unknown class')}\n")
                        f.write("-" * 60 + "\n")
                        try:
                            f.write(get_class_summary(cls))
                        except Exception as e:
                            f.write(f"Error generating class summary: {str(e)}\n")
                        f.write("\n\n")
                else:
                    f.write("No classes found in the codebase.\n\n")
            except Exception as e:
                f.write(f"Error processing class summaries: {str(e)}\n\n")
            
            # Write function summaries (limit to top 30 for large codebases)
            try:
                f.write("=" * 80 + "\n")
                f.write(" " * 30 + "FUNCTION SUMMARIES\n")
                f.write("=" * 80 + "\n")
                functions = list(self.codebase.functions)
                if functions:
                    max_funcs = min(30, len(functions))
                    f.write(f"Showing {max_funcs} of {len(functions)} functions\n\n")
                    for i, func in enumerate(functions[:max_funcs]):
                        f.write(f"Function {i+1}/{max_funcs}: {getattr(func, 'name', 'Unknown function')}\n")
                        f.write("-" * 60 + "\n")
                        try:
                            f.write(get_function_summary(func))
                        except Exception as e:
                            f.write(f"Error generating function summary: {str(e)}\n")
                        f.write("\n\n")
                    
                    if len(functions) > max_funcs:
                        f.write(f"... and {len(functions) - max_funcs} more functions not shown\n\n")
                else:
                    f.write("No functions found in the codebase.\n\n")
            except Exception as e:
                f.write(f"Error processing function summaries: {str(e)}\n\n")
                
            # Add symbol usage summary for important symbols
            try:
                f.write("=" * 80 + "\n")
                f.write(" " * 30 + "KEY SYMBOL USAGE SUMMARIES\n")
                f.write("=" * 80 + "\n")
                
                # Get top 10 most used symbols
                symbols = list(self.codebase.symbols)
                if symbols:
                    # Sort symbols by usage count (if available)
                    try:
                        symbols_with_usage = [(s, len(getattr(s, 'usages', []))) for s in symbols]
                        sorted_symbols = [s for s, _ in sorted(symbols_with_usage, key=lambda x: x[1], reverse=True)]
                        top_symbols = sorted_symbols[:10]
                    except:
                        # Fall back to first 10 symbols if sorting fails
                        top_symbols = symbols[:10]
                        
                    for i, symbol in enumerate(top_symbols):
                        f.write(f"Symbol {i+1}/{len(top_symbols)}: {getattr(symbol, 'name', 'Unknown symbol')}\n")
                        f.write("-" * 60 + "\n")
                        try:
                            f.write(get_symbol_summary(symbol))
                        except Exception as e:
                            f.write(f"Error generating symbol summary: {str(e)}\n")
                        f.write("\n\n")
                else:
                    f.write("No symbols found for detailed analysis.\n\n")
            except Exception as e:
                f.write(f"Error processing symbol summaries: {str(e)}\n\n")
                
            f.write("=" * 80 + "\n")
            f.write(" " * 25 + "END OF DETAILED CODEBASE ANALYSIS\n")
            f.write("=" * 80 + "\n")

def main():
    """Run the comprehensive analyzer from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze a codebase comprehensively using the Codegen SDK"
    )
    parser.add_argument(
        "--repo", 
        default="./",
        help="Repository URL or local path to analyze"
    )
    args = parser.parse_args()
    
    analyzer = ComprehensiveAnalyzer(args.repo)
    analyzer.analyze()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())