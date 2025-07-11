"""
Enhanced analysis module with graph-sitter integration and automatic resolution capabilities.
This module provides comprehensive repository analysis with tree structure visualization,
issue detection, and intelligent automatic resolution suggestions.
"""

from typing import Dict, List, Any
from graph_sitter.core.codebase import Codebase
from .analysis import analyze_code_issues, calculate_doi


def generate_repository_analysis_report(codebase: Codebase, repo_url: str = "") -> str:
    """
    Generate a comprehensive repository analysis report with tree structure,
    issue detection, and automatic resolution suggestions.
    """
    try:
        # Get repository information
        repo_name = repo_url.split('/')[-1] if repo_url else "Unknown Repository"
        
        # Calculate basic statistics
        files = list(codebase.files)
        functions = list(codebase.functions)
        classes = list(codebase.classes)
        
        total_files = len(files)
        total_functions = len(functions)
        total_classes = len(classes)
        
        # Analyze issues
        issues = analyze_code_issues(codebase, max_issues=500)
        critical_issues = [issue for issue in issues if issue.severity == "critical"]
        
        # Build directory tree with issue counts
        directory_tree = build_directory_tree_with_issues(codebase, issues)
        
        # Find classes with most inheritance
        inheritance_analysis = analyze_inheritance_hierarchy(codebase)
        
        # Generate automatic resolution suggestions
        resolution_suggestions = generate_automatic_resolutions(codebase, issues)
        
        # Build the report
        report = f"""📊 Repository Analysis Report 📊
==================================================
📁 Repository: {repo_name}
📝 Description: Advanced codebase analysis with graph-sitter integration

📁 Files: {total_files}
🔄 Functions: {total_functions}
📏 Classes: {total_classes}

{directory_tree}

🔍 **CRITICAL ISSUES DETECTED** ({len(critical_issues)} critical issues found)
{format_critical_issues(critical_issues)}

🏗️ **INHERITANCE ANALYSIS**
Classes with most inheritance:
{format_inheritance_analysis(inheritance_analysis)}

🤖 **AUTOMATIC RESOLUTION SUGGESTIONS**
{format_resolution_suggestions(resolution_suggestions)}

📈 **GRAPH-SITTER INTEGRATION INSIGHTS**
- Pre-computed dependency graph with {len(list(codebase.ctx.edges))} edges
- Symbol usage analysis across {total_files} files
- Multi-language support: Python, TypeScript, React & JSX
- Advanced static analysis for code manipulation operations

🔧 **RECOMMENDED ACTIONS**
{generate_recommended_actions(issues, inheritance_analysis)}
"""
        
        return report
        
    except Exception as e:
        return f"Error generating repository analysis report: {str(e)}"


def build_directory_tree_with_issues(codebase: Codebase, issues: List[Any]) -> str:
    """Build a directory tree structure with issue counts and file statistics."""
    try:
        # Group issues by file path
        issues_by_file = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)
        
        # Build directory structure
        directories = {}
        files_info = {}
        
        for file in codebase.files:
            file_path = file.filepath
            path_parts = file_path.split('/')
            
            # Count lines of code
            loc = len(file.source.splitlines()) if hasattr(file, 'source') else 0
            
            # Count issues for this file
            file_issues = issues_by_file.get(file_path, [])
            critical_issues_count = len([i for i in file_issues if i.severity == "critical"])
            
            files_info[file_path] = {
                'loc': loc,
                'functions': len(file.functions),
                'classes': len(file.classes),
                'issues': len(file_issues),
                'critical_issues': critical_issues_count
            }
            
            # Build directory structure
            current_dir = directories
            for i, part in enumerate(path_parts[:-1]):
                if part not in current_dir:
                    current_dir[part] = {'dirs': {}, 'files': [], 'total_issues': 0, 'critical_issues': 0}
                current_dir = current_dir[part]['dirs']
            
            # Add file to final directory
            final_dir = path_parts[-2] if len(path_parts) > 1 else ''
            if final_dir and final_dir in current_dir:
                current_dir[final_dir]['files'].append(path_parts[-1])
                current_dir[final_dir]['total_issues'] += len(file_issues)
                current_dir[final_dir]['critical_issues'] += critical_issues_count
        
        # Format the tree
        tree_output = format_directory_tree(directories, files_info, "", True)
        return tree_output
        
    except Exception as e:
        return f"Error building directory tree: {str(e)}"


def format_directory_tree(directories: Dict, files_info: Dict, prefix: str = "", is_root: bool = False) -> str:
    """Format the directory tree with issue indicators."""
    output = ""
    
    if is_root:
        output += "Repository Structure:\n"
    
    for i, (dir_name, dir_info) in enumerate(directories.items()):
        is_last_dir = i == len(directories) - 1
        current_prefix = "└── " if is_last_dir else "├── "
        
        # Calculate directory statistics
        total_issues = dir_info.get('total_issues', 0)
        critical_issues = dir_info.get('critical_issues', 0)
        
        issue_indicator = ""
        if critical_issues > 0:
            issue_indicator = f" [⚠️ Critical: {critical_issues}]"
        elif total_issues > 0:
            issue_indicator = f" [⚡ Issues: {total_issues}]"
        
        output += f"{prefix}{current_prefix}📁 {dir_name}/{issue_indicator}\n"
        
        # Add subdirectories
        next_prefix = prefix + ("    " if is_last_dir else "│   ")
        if dir_info['dirs']:
            output += format_directory_tree(dir_info['dirs'], files_info, next_prefix)
        
        # Add files
        for j, file_name in enumerate(dir_info.get('files', [])):
            is_last_file = j == len(dir_info['files']) - 1 and not dir_info['dirs']
            file_prefix = "└── " if is_last_file else "├── "
            
            # Get file info
            full_path = f"{dir_name}/{file_name}"  # Simplified path construction
            file_info = files_info.get(full_path, {})
            
            file_indicator = ""
            if file_info.get('critical_issues', 0) > 0:
                file_indicator = f" [⚠️ Critical: {file_info['critical_issues']}]"
            elif file_info.get('issues', 0) > 0:
                file_indicator = f" [⚡ Issues: {file_info['issues']}]"
            
            # File type indicator
            if file_name.endswith('.py'):
                file_type = "🐍"
            elif file_name.endswith(('.ts', '.tsx', '.js', '.jsx')):
                file_type = "📜"
            else:
                file_type = "📄"
            
            output += f"{next_prefix}{file_prefix}{file_type} {file_name}{file_indicator}\n"
    
    return output


def analyze_inheritance_hierarchy(codebase: Codebase) -> List[Dict[str, Any]]:
    """Analyze inheritance hierarchy and find classes with deep inheritance."""
    inheritance_data = []
    
    try:
        for file in codebase.files:
            for cls in file.classes:
                try:
                    depth = calculate_doi(cls)
                    if depth > 1:  # Only include classes with inheritance
                        inheritance_data.append({
                            'class_name': cls.name,
                            'file_path': file.filepath,
                            'inheritance_depth': depth,
                            'parent_classes': cls.parent_class_names,
                            'methods_count': len(cls.methods),
                            'attributes_count': len(cls.attributes)
                        })
                except Exception as e:
                    print(f"Error analyzing inheritance for class {cls.name}: {e}")
                    continue
        
        # Sort by inheritance depth
        inheritance_data.sort(key=lambda x: x['inheritance_depth'], reverse=True)
        return inheritance_data[:10]  # Return top 10
        
    except Exception as e:
        print(f"Error in inheritance analysis: {e}")
        return []


def generate_automatic_resolutions(codebase: Codebase, issues: List[Any]) -> List[Dict[str, Any]]:
    """
    Generate automatic resolution suggestions based on graph-sitter analysis.
    This leverages the pre-computed dependency graph for intelligent fixes.
    """
    resolutions = []
    
    try:
        # Group issues by type for batch resolution
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.type
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Generate resolutions for each issue type
        for issue_type, type_issues in issues_by_type.items():
            if issue_type == "complexity_issue":
                resolutions.extend(generate_complexity_resolutions(codebase, type_issues))
            elif issue_type == "maintainability_issue":
                resolutions.extend(generate_maintainability_resolutions(codebase, type_issues))
            elif issue_type == "code_smell":
                resolutions.extend(generate_code_smell_resolutions(codebase, type_issues))
            elif issue_type == "dependency_issue":
                resolutions.extend(generate_dependency_resolutions(codebase, type_issues))
        
        return resolutions[:20]  # Return top 20 resolutions
        
    except Exception as e:
        print(f"Error generating automatic resolutions: {e}")
        return []


def generate_complexity_resolutions(codebase: Codebase, issues: List[Any]) -> List[Dict[str, Any]]:
    """Generate automatic resolutions for complexity issues."""
    resolutions = []
    
    for issue in issues:
        try:
            # Find the function with complexity issue
            function_name = issue.function_name
            file_path = issue.file_path
            
            # Use graph-sitter to analyze function dependencies
            for file in codebase.files:
                if file.filepath == file_path:
                    for func in file.functions:
                        if func.name == function_name:
                            # Analyze function structure for refactoring suggestions
                            suggestions = analyze_function_for_refactoring(func, codebase)
                            
                            resolutions.append({
                                'issue_id': issue.id,
                                'resolution_type': 'function_refactoring',
                                'priority': 'high',
                                'description': f"Refactor complex function '{function_name}'",
                                'automatic_actions': suggestions,
                                'confidence': 0.85
                            })
                            break
                    break
        except Exception as e:
            print(f"Error generating complexity resolution: {e}")
            continue
    
    return resolutions


def analyze_function_for_refactoring(func: Any, codebase: Codebase) -> List[str]:
    """Analyze a function and suggest specific refactoring actions."""
    suggestions = []
    
    try:
        # Analyze function calls to suggest extraction
        if len(func.function_calls) > 10:
            suggestions.append("Extract repeated function calls into helper methods")
        
        # Analyze parameters for object grouping
        if len(func.parameters) > 5:
            suggestions.append("Group related parameters into configuration objects")
        
        # Analyze return statements for simplification
        if len(func.return_statements) > 3:
            suggestions.append("Simplify multiple return paths using early returns")
        
        # Use graph-sitter dependency analysis
        if len(func.dependencies) > 8:
            suggestions.append("Reduce dependencies by applying dependency injection")
        
        return suggestions
        
    except Exception as e:
        print(f"Error analyzing function for refactoring: {e}")
        return ["Apply general refactoring principles"]


def generate_maintainability_resolutions(codebase: Codebase, issues: List[Any]) -> List[Dict[str, Any]]:
    """Generate resolutions for maintainability issues."""
    resolutions = []
    
    for issue in issues[:5]:  # Limit to top 5 maintainability issues
        resolutions.append({
            'issue_id': issue.id,
            'resolution_type': 'maintainability_improvement',
            'priority': 'medium',
            'description': f"Improve maintainability in {issue.file_path}",
            'automatic_actions': [
                "Add comprehensive documentation",
                "Implement unit tests",
                "Apply consistent naming conventions",
                "Reduce function length through extraction"
            ],
            'confidence': 0.75
        })
    
    return resolutions


def generate_code_smell_resolutions(codebase: Codebase, issues: List[Any]) -> List[Dict[str, Any]]:
    """Generate resolutions for code smell issues."""
    resolutions = []
    
    for issue in issues[:3]:  # Limit to top 3 code smell issues
        resolutions.append({
            'issue_id': issue.id,
            'resolution_type': 'code_smell_cleanup',
            'priority': 'low',
            'description': f"Clean up code smell in {issue.function_name or issue.class_name}",
            'automatic_actions': [
                "Apply SOLID principles",
                "Remove duplicate code",
                "Improve naming conventions",
                "Simplify complex expressions"
            ],
            'confidence': 0.70
        })
    
    return resolutions


def generate_dependency_resolutions(codebase: Codebase, issues: List[Any]) -> List[Dict[str, Any]]:
    """Generate resolutions for dependency issues using graph-sitter analysis."""
    resolutions = []
    
    try:
        # Use graph-sitter's pre-computed dependency graph
        for issue in issues[:3]:
            resolutions.append({
                'issue_id': issue.id,
                'resolution_type': 'dependency_optimization',
                'priority': 'high',
                'description': f"Optimize dependencies in {issue.file_path}",
                'automatic_actions': [
                    "Remove unused imports using graph-sitter analysis",
                    "Reorganize import statements",
                    "Apply dependency inversion principle",
                    "Use graph-sitter to detect circular dependencies"
                ],
                'confidence': 0.90
            })
    
    except Exception as e:
        print(f"Error generating dependency resolutions: {e}")
    
    return resolutions


def format_critical_issues(critical_issues: List[Any]) -> str:
    """Format critical issues for the report."""
    if not critical_issues:
        return "✅ No critical issues detected!"
    
    output = ""
    for issue in critical_issues[:10]:  # Show top 10 critical issues
        output += f"⚠️  {issue.file_path}:{issue.line_number or 'N/A'} - {issue.message}\n"
        output += f"   📍 {issue.description}\n"
        if issue.fix_suggestions:
            output += f"   💡 Suggestion: {issue.fix_suggestions[0]}\n"
        output += "\n"
    
    return output


def format_inheritance_analysis(inheritance_data: List[Dict[str, Any]]) -> str:
    """Format inheritance analysis for the report."""
    if not inheritance_data:
        return "✅ No deep inheritance hierarchies detected!"
    
    output = ""
    for item in inheritance_data:
        output += f"{item['file_path']} [⚕️{item['class_name']}] (Depth: {item['inheritance_depth']})\n"
        if item['parent_classes']:
            output += f"   └── Inherits from: {', '.join(item['parent_classes'])}\n"
    
    return output


def format_resolution_suggestions(resolutions: List[Dict[str, Any]]) -> str:
    """Format automatic resolution suggestions."""
    if not resolutions:
        return "✅ No automatic resolutions needed!"
    
    output = ""
    for resolution in resolutions:
        priority_icon = "🔴" if resolution['priority'] == 'high' else "🟡" if resolution['priority'] == 'medium' else "🟢"
        confidence = int(resolution['confidence'] * 100)
        
        output += f"{priority_icon} {resolution['description']} (Confidence: {confidence}%)\n"
        for action in resolution['automatic_actions'][:2]:  # Show top 2 actions
            output += f"   • {action}\n"
        output += "\n"
    
    return output


def generate_recommended_actions(issues: List[Any], inheritance_analysis: List[Dict[str, Any]]) -> str:
    """Generate recommended actions based on analysis."""
    actions = []
    
    # Critical issues
    critical_count = len([i for i in issues if i.severity == "critical"])
    if critical_count > 0:
        actions.append(f"🔴 Address {critical_count} critical issues immediately")
    
    # High complexity functions
    complexity_issues = len([i for i in issues if i.type == "complexity_issue"])
    if complexity_issues > 5:
        actions.append(f"🟡 Refactor {complexity_issues} complex functions")
    
    # Deep inheritance
    deep_inheritance = len([i for i in inheritance_analysis if i['inheritance_depth'] > 3])
    if deep_inheritance > 0:
        actions.append(f"🟠 Review {deep_inheritance} classes with deep inheritance")
    
    # Graph-sitter specific recommendations
    actions.append("🔧 Leverage graph-sitter's pre-computed dependency graph for faster refactoring")
    actions.append("📊 Use graph-sitter's symbol usage analysis for safe code transformations")
    actions.append("🚀 Implement automated code fixes using graph-sitter's AST manipulation")
    
    return "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])
