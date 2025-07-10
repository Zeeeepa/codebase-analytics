"""
Advanced code analysis features including call graphs, multi-view analysis,
advanced metrics, security analysis, and performance optimization.
"""

from typing import Dict, List, Any, Set
from collections import defaultdict, Counter
import re
import hashlib
from pathlib import Path
from datetime import datetime

# Graph-sitter imports
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.class_definition import Class
from graph_sitter.core.function import Function
from graph_sitter.core.file import SourceFile
from graph_sitter.core.symbol import Symbol


class CodeAnalysisError(Exception):
    """Custom exception for code analysis errors"""
    pass


class CodebaseCache:
    """Cache for AST and metrics to improve performance"""
    def __init__(self):
        self.ast_cache = {}
        self.metric_cache = {}
        self.call_graph_cache = {}
        
    def get_or_compute_ast(self, file: SourceFile) -> Dict:
        """Get AST from cache or compute if not available"""
        if file.filepath not in self.ast_cache:
            self.ast_cache[file.filepath] = self.parse_file_ast(file)
        return self.ast_cache[file.filepath]
    
    def parse_file_ast(self, file: SourceFile) -> Dict:
        """Parse file AST with error handling"""
        try:
            return {
                "file_path": file.filepath,
                "functions": [{"name": f.name, "line": getattr(f, 'line_number', None)} for f in file.functions],
                "classes": [{"name": c.name, "line": getattr(c, 'line_number', None)} for c in file.classes],
                "imports": [{"name": i.name} for i in file.imports]
            }
        except Exception as e:
            raise CodeAnalysisError(f"Failed to parse AST for {file.filepath}: {str(e)}")


# Global cache instance
codebase_cache = CodebaseCache()


def validate_source_code(file: SourceFile) -> bool:
    """Validate source code for syntax errors before analysis"""
    try:
        codebase_cache.get_or_compute_ast(file)
        return True
    except Exception as e:
        raise CodeAnalysisError(f"Syntax error in {file.filepath}: {str(e)}")


def generate_call_graph(codebase: Codebase) -> Dict[str, Any]:
    """Generate a comprehensive call graph for the entire codebase."""
    from .api import calculate_cyclomatic_complexity, get_operators_and_operands, calculate_halstead_volume
    
    graph_data = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "total_functions": 0,
            "total_calls": 0,
            "max_complexity": 0,
            "languages": set()
        }
    }
    
    function_map = {}  # Map function names to IDs for edge creation
    
    for file in codebase.files:
        # Determine language from file extension
        ext = Path(file.filepath).suffix.lower()
        graph_data["metadata"]["languages"].add(ext)
        
        for function in file.functions:
            try:
                complexity = calculate_cyclomatic_complexity(function)
                operators, operands = get_operators_and_operands(function)
                volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
                
                function_id = f"{file.filepath}:{function.name}"
                function_map[function.name] = function_id
                
                # Add node for each function
                graph_data["nodes"].append({
                    "id": function_id,
                    "name": function.name,
                    "type": "function",
                    "file": file.filepath,
                    "language": ext,
                    "metrics": {
                        "complexity": complexity,
                        "halstead_volume": volume,
                        "parameters": len(function.parameters),
                        "return_statements": len(function.return_statements)
                    }
                })
                
                graph_data["metadata"]["total_functions"] += 1
                graph_data["metadata"]["max_complexity"] = max(
                    graph_data["metadata"]["max_complexity"], complexity
                )
                
                # Add edges for function calls
                for call in function.function_calls:
                    target_id = function_map.get(call.name, f"external:{call.name}")
                    graph_data["edges"].append({
                        "source": function_id,
                        "target": target_id,
                        "type": "calls",
                        "call_name": call.name
                    })
                    graph_data["metadata"]["total_calls"] += 1
                    
            except Exception as e:
                print(f"Error processing function {function.name}: {e}")
                continue
    
    graph_data["metadata"]["languages"] = list(graph_data["metadata"]["languages"])
    return graph_data


def generate_code_views(file: SourceFile) -> Dict[str, Any]:
    """Generate multiple views of code structure including AST, CFG, and DFG."""
    views = {
        "ast": generate_ast_view(file),
        "cfg": generate_control_flow_graph(file),
        "dfg": generate_data_flow_graph(file),
        "metadata": {
            "file_path": file.filepath,
            "file_size": len(file.source),
            "language": Path(file.filepath).suffix.lower()
        }
    }
    return views


def generate_ast_view(file: SourceFile) -> Dict[str, Any]:
    """Generate Abstract Syntax Tree view"""
    from .api import calculate_cyclomatic_complexity
    
    try:
        ast_data = {
            "functions": [],
            "classes": [],
            "imports": [],
            "global_variables": []
        }
        
        for func in file.functions:
            ast_data["functions"].append({
                "name": func.name,
                "line_number": getattr(func, 'line_number', None),
                "parameters": [p.name for p in func.parameters],
                "return_type": getattr(func, 'return_type', None),
                "decorators": [d.name for d in func.decorators] if hasattr(func, 'decorators') else [],
                "complexity": calculate_cyclomatic_complexity(func)
            })
        
        for cls in file.classes:
            ast_data["classes"].append({
                "name": cls.name,
                "line_number": getattr(cls, 'line_number', None),
                "methods": [m.name for m in cls.methods],
                "attributes": [a.name for a in cls.attributes],
                "parent_classes": cls.parent_class_names
            })
        
        for imp in file.imports:
            ast_data["imports"].append({
                "name": imp.name,
                "module": getattr(imp, 'module', None),
                "alias": getattr(imp, 'alias', None)
            })
        
        for var in file.global_vars:
            ast_data["global_variables"].append({
                "name": var.name,
                "type": getattr(var, 'type', None)
            })
        
        return ast_data
        
    except Exception as e:
        return {"error": f"Failed to generate AST view: {str(e)}"}


def generate_control_flow_graph(file: SourceFile) -> Dict[str, Any]:
    """Generate Control Flow Graph"""
    from .api import calculate_cyclomatic_complexity
    
    cfg_data = {
        "functions": {},
        "complexity_analysis": {}
    }
    
    try:
        for func in file.functions:
            func_cfg = {
                "entry_points": [],
                "exit_points": [],
                "branches": [],
                "loops": [],
                "complexity": calculate_cyclomatic_complexity(func)
            }
            
            # Analyze control flow patterns in source
            if hasattr(func, 'source'):
                source = func.source.lower()
                
                # Count control flow structures
                func_cfg["branches"] = [
                    {"type": "if", "count": source.count('if ')},
                    {"type": "elif", "count": source.count('elif ')},
                    {"type": "else", "count": source.count('else')},
                    {"type": "try", "count": source.count('try:')},
                    {"type": "except", "count": source.count('except')}
                ]
                
                func_cfg["loops"] = [
                    {"type": "for", "count": source.count('for ')},
                    {"type": "while", "count": source.count('while ')}
                ]
                
                func_cfg["exit_points"] = [
                    {"type": "return", "count": source.count('return ')},
                    {"type": "raise", "count": source.count('raise ')},
                    {"type": "break", "count": source.count('break')},
                    {"type": "continue", "count": source.count('continue')}
                ]
            
            cfg_data["functions"][func.name] = func_cfg
            cfg_data["complexity_analysis"][func.name] = func_cfg["complexity"]
        
        return cfg_data
        
    except Exception as e:
        return {"error": f"Failed to generate CFG: {str(e)}"}


def generate_data_flow_graph(file: SourceFile) -> Dict[str, Any]:
    """Generate Data Flow Graph"""
    dfg_data = {
        "variables": {},
        "data_dependencies": [],
        "assignments": []
    }
    
    try:
        for func in file.functions:
            func_variables = {
                "parameters": [p.name for p in func.parameters],
                "local_vars": [],
                "assignments": [],
                "usages": []
            }
            
            # Analyze variable usage patterns
            if hasattr(func, 'source'):
                source = func.source
                
                # Simple pattern matching for assignments
                assignment_pattern = r'(\w+)\s*=\s*'
                assignments = re.findall(assignment_pattern, source)
                func_variables["assignments"] = list(set(assignments))
                
                # Variable usage pattern
                for param in func_variables["parameters"]:
                    usage_count = source.count(param)
                    if usage_count > 1:  # More than just the parameter declaration
                        func_variables["usages"].append({
                            "variable": param,
                            "usage_count": usage_count - 1
                        })
            
            dfg_data["variables"][func.name] = func_variables
        
        return dfg_data
        
    except Exception as e:
        return {"error": f"Failed to generate DFG: {str(e)}"}


def calculate_advanced_metrics(codebase: Codebase) -> Dict[str, Any]:
    """Calculate advanced code metrics including complexity, dependencies, and quality."""
    from .api import calculate_cyclomatic_complexity, get_operators_and_operands, calculate_halstead_volume, calculate_maintainability_index
    
    metrics = {
        "complexity": {
            "cyclomatic": {},
            "cognitive": {},
            "halstead": {}
        },
        "dependencies": {
            "internal": [],
            "external": [],
            "circular": []
        },
        "quality": {
            "maintainability_index": {},
            "code_smells": [],
            "technical_debt": {}
        },
        "summary": {
            "total_files": 0,
            "total_functions": 0,
            "total_classes": 0,
            "avg_complexity": 0,
            "high_complexity_functions": 0
        }
    }
    
    total_complexity = 0
    function_count = 0
    
    for file in codebase.files:
        try:
            file_metrics = analyze_file_metrics(file)
            
            metrics["complexity"]["cyclomatic"][file.filepath] = file_metrics["cyclomatic"]
            metrics["complexity"]["cognitive"][file.filepath] = file_metrics["cognitive"]
            metrics["complexity"]["halstead"][file.filepath] = file_metrics["halstead"]
            metrics["quality"]["maintainability_index"][file.filepath] = file_metrics["maintainability"]
            
            # Accumulate for summary
            metrics["summary"]["total_files"] += 1
            metrics["summary"]["total_functions"] += len(file.functions)
            metrics["summary"]["total_classes"] += len(file.classes)
            
            for func in file.functions:
                complexity = calculate_cyclomatic_complexity(func)
                total_complexity += complexity
                function_count += 1
                
                if complexity > 15:
                    metrics["summary"]["high_complexity_functions"] += 1
                    metrics["quality"]["code_smells"].append({
                        "type": "high_complexity",
                        "file": file.filepath,
                        "function": func.name,
                        "complexity": complexity
                    })
            
            # Analyze dependencies
            for imp in file.imports:
                if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'filepath'):
                    metrics["dependencies"]["internal"].append({
                        "from": file.filepath,
                        "to": imp.imported_symbol.filepath,
                        "symbol": imp.name
                    })
                else:
                    metrics["dependencies"]["external"].append({
                        "from": file.filepath,
                        "module": imp.name
                    })
                    
        except Exception as e:
            print(f"Error analyzing file {file.filepath}: {e}")
            continue
    
    # Calculate summary statistics
    if function_count > 0:
        metrics["summary"]["avg_complexity"] = total_complexity / function_count
    
    # Detect circular dependencies
    metrics["dependencies"]["circular"] = detect_circular_dependencies(
        metrics["dependencies"]["internal"]
    )
    
    return metrics


def analyze_file_metrics(file: SourceFile) -> Dict[str, Any]:
    """Analyze metrics for a single file"""
    from .api import calculate_cyclomatic_complexity, get_operators_and_operands, calculate_halstead_volume, calculate_maintainability_index
    
    file_metrics = {
        "cyclomatic": {},
        "cognitive": {},
        "halstead": {},
        "maintainability": {}
    }
    
    try:
        total_complexity = 0
        total_volume = 0
        total_mi = 0
        
        for func in file.functions:
            complexity = calculate_cyclomatic_complexity(func)
            operators, operands = get_operators_and_operands(func)
            volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
            
            loc = len(func.source.splitlines()) if hasattr(func, 'source') else 0
            mi_score = calculate_maintainability_index(volume, complexity, loc)
            
            file_metrics["cyclomatic"][func.name] = complexity
            file_metrics["cognitive"][func.name] = complexity  # Simplified
            file_metrics["halstead"][func.name] = volume
            file_metrics["maintainability"][func.name] = mi_score
            
            total_complexity += complexity
            total_volume += volume
            total_mi += mi_score
        
        # File-level averages
        func_count = len(file.functions)
        if func_count > 0:
            file_metrics["avg_complexity"] = total_complexity / func_count
            file_metrics["avg_volume"] = total_volume / func_count
            file_metrics["avg_maintainability"] = total_mi / func_count
        
        return file_metrics
        
    except Exception as e:
        return {"error": f"Failed to analyze file metrics: {str(e)}"}


def detect_circular_dependencies(internal_deps: List[Dict]) -> List[Dict]:
    """Detect circular dependencies in the codebase"""
    circular_deps = []
    
    try:
        # Build dependency graph
        dep_graph = defaultdict(set)
        for dep in internal_deps:
            dep_graph[dep["from"]].add(dep["to"])
        
        # Simple cycle detection using DFS
        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dep_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    circular_deps.append({
                        "cycle": cycle,
                        "length": len(cycle) - 1
                    })
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        visited = set()
        for node in dep_graph:
            if node not in visited:
                has_cycle(node, visited, set(), [])
        
        return circular_deps[:10]  # Limit results
        
    except Exception as e:
        return [{"error": f"Failed to detect circular dependencies: {str(e)}"}]


def analyze_security_patterns(codebase: Codebase) -> Dict[str, Any]:
    """Analyze security patterns and potential vulnerabilities"""
    security_analysis = {
        "vulnerabilities": [],
        "security_smells": [],
        "call_graph_anomalies": [],
        "dependency_risks": [],
        "summary": {
            "total_vulnerabilities": 0,
            "critical_issues": 0,
            "high_risk_files": []
        }
    }
    
    try:
        # Security patterns to detect
        security_patterns = [
            (r'eval\s*\(', "Code injection via eval()", "critical"),
            (r'exec\s*\(', "Code execution via exec()", "critical"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection", "high"),
            (r'pickle\.loads?\s*\(', "Unsafe deserialization", "high"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "high"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "high"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret", "medium"),
            (r'sql.*\+.*\+', "Potential SQL injection", "medium"),
            (r'open\s*\(.*["\']w["\']', "File write operation", "low"),
            (r'requests\.get\s*\(.*verify\s*=\s*False', "SSL verification disabled", "medium")
        ]
        
        file_risk_scores = {}
        
        for file in codebase.files:
            file_vulnerabilities = []
            file_risk_score = 0
            
            for pattern, description, severity in security_patterns:
                matches = re.finditer(pattern, file.source, re.IGNORECASE)
                for match in matches:
                    line_num = file.source[:match.start()].count('\n') + 1
                    
                    vulnerability = {
                        "type": "security_vulnerability",
                        "severity": severity,
                        "description": description,
                        "file": file.filepath,
                        "line": line_num,
                        "matched_text": match.group(),
                        "pattern": pattern
                    }
                    
                    file_vulnerabilities.append(vulnerability)
                    security_analysis["vulnerabilities"].append(vulnerability)
                    
                    # Calculate risk score
                    risk_points = {"critical": 10, "high": 5, "medium": 2, "low": 1}
                    file_risk_score += risk_points.get(severity, 0)
                    
                    if severity == "critical":
                        security_analysis["summary"]["critical_issues"] += 1
            
            if file_vulnerabilities:
                file_risk_scores[file.filepath] = {
                    "risk_score": file_risk_score,
                    "vulnerability_count": len(file_vulnerabilities)
                }
        
        # Analyze call graph for anomalies
        try:
            call_graph = generate_call_graph(codebase)
            security_analysis["call_graph_anomalies"] = detect_call_anomalies(call_graph)
        except Exception as e:
            security_analysis["call_graph_anomalies"] = [{"error": str(e)}]
        
        # Identify high-risk files
        sorted_files = sorted(file_risk_scores.items(), key=lambda x: x[1]["risk_score"], reverse=True)
        security_analysis["summary"]["high_risk_files"] = [
            {"file": file, "risk_score": data["risk_score"], "vulnerabilities": data["vulnerability_count"]}
            for file, data in sorted_files[:10]
        ]
        
        security_analysis["summary"]["total_vulnerabilities"] = len(security_analysis["vulnerabilities"])
        
        return security_analysis
        
    except Exception as e:
        return {"error": f"Security analysis failed: {str(e)}"}


def detect_call_anomalies(call_graph: Dict[str, Any]) -> List[Dict]:
    """Detect anomalies in call graph patterns"""
    anomalies = []
    
    try:
        # Analyze call patterns
        call_counts = defaultdict(int)
        function_calls = defaultdict(set)
        
        for edge in call_graph["edges"]:
            call_counts[edge["target"]] += 1
            function_calls[edge["source"]].add(edge["target"])
        
        # Detect highly called functions (potential bottlenecks)
        avg_calls = sum(call_counts.values()) / len(call_counts) if call_counts else 0
        for func, count in call_counts.items():
            if count > avg_calls * 3:  # 3x average
                anomalies.append({
                    "type": "high_call_frequency",
                    "function": func,
                    "call_count": count,
                    "description": f"Function called {count} times (avg: {avg_calls:.1f})"
                })
        
        # Detect functions with too many outgoing calls
        for func, calls in function_calls.items():
            if len(calls) > 10:
                anomalies.append({
                    "type": "high_fan_out",
                    "function": func,
                    "outgoing_calls": len(calls),
                    "description": f"Function makes {len(calls)} different calls"
                })
        
        return anomalies[:20]  # Limit results
        
    except Exception as e:
        return [{"error": f"Failed to detect call anomalies: {str(e)}"}]

