"""
Function Context Analysis
Comprehensive function context tracking with dependencies and call chains
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import networkx as nx


@dataclass
class FunctionContext:
    """Complete context for a function with all relationships"""
    name: str
    filepath: str
    line_start: int
    line_end: int
    source: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    usages: List[Dict[str, Any]] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    class_name: Optional[str] = None
    max_call_chain: List[str] = field(default_factory=list)
    issues: List[Any] = field(default_factory=list)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    is_entry_point: bool = False
    is_dead_code: bool = False
    call_depth: int = 0
    fan_in: int = 0  # Number of functions calling this one
    fan_out: int = 0  # Number of functions this one calls
    coupling_score: float = 0.0
    cohesion_score: float = 0.0


class FunctionContextAnalyzer:
    """Analyzer for building comprehensive function contexts"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.function_contexts = {}
        self.call_graph = nx.DiGraph()
        
    def analyze_all_functions(self) -> Dict[str, FunctionContext]:
        """Analyze all functions and build comprehensive contexts"""
        print("📊 Building function contexts...")
        
        # First pass: Create basic contexts
        for function in self.codebase.functions:
            context = self._create_basic_context(function)
            self.function_contexts[function.name] = context
        
        # Second pass: Build call graph
        self._build_call_graph()
        
        # Third pass: Calculate advanced metrics
        self._calculate_advanced_metrics()
        
        # Fourth pass: Identify patterns
        self._identify_function_patterns()
        
        print(f"✅ Analyzed {len(self.function_contexts)} functions")
        return self.function_contexts
    
    def _create_basic_context(self, function) -> FunctionContext:
        """Create basic function context"""
        context = FunctionContext(
            name=function.name,
            filepath=function.filepath,
            line_start=function.start_point[0] if hasattr(function, 'start_point') else 0,
            line_end=function.end_point[0] if hasattr(function, 'end_point') else 0,
            source=function.source if hasattr(function, 'source') else ""
        )
        
        # Extract parameters
        if hasattr(function, 'parameters'):
            for param in function.parameters:
                context.parameters.append({
                    "name": param.name if hasattr(param, 'name') else str(param),
                    "type": param.type if hasattr(param, 'type') else None,
                    "default": param.default if hasattr(param, 'default') else None,
                    "is_optional": hasattr(param, 'default') and param.default is not None
                })
        
        # Extract return type
        if hasattr(function, 'return_type'):
            context.return_type = function.return_type
        
        # Extract dependencies
        if hasattr(function, 'dependencies'):
            for dep in function.dependencies:
                context.dependencies.append({
                    "name": dep.name if hasattr(dep, 'name') else str(dep),
                    "source": dep.source if hasattr(dep, 'source') else "",
                    "filepath": dep.filepath if hasattr(dep, 'filepath') else "",
                    "type": "dependency"
                })
        
        # Extract usages
        if hasattr(function, 'usages'):
            for usage in function.usages:
                if hasattr(usage, 'usage_symbol'):
                    context.usages.append({
                        "source": usage.usage_symbol.source,
                        "filepath": usage.usage_symbol.filepath,
                        "line": usage.usage_symbol.start_point[0] if hasattr(usage.usage_symbol, 'start_point') else 0,
                        "context": "usage"
                    })
        
        # Extract function calls
        if hasattr(function, 'function_calls'):
            context.function_calls = [
                call.name for call in function.function_calls 
                if hasattr(call, 'name')
            ]
        
        # Extract called by
        if hasattr(function, 'call_sites'):
            context.called_by = [
                call.parent_function.name 
                for call in function.call_sites 
                if hasattr(call, 'parent_function') and hasattr(call.parent_function, 'name')
            ]
        
        # Get class name if applicable
        if hasattr(function, 'parent_class') and function.parent_class:
            context.class_name = function.parent_class.name
        
        # Check if entry point
        context.is_entry_point = self._is_entry_point(function)
        
        # Calculate basic complexity
        context.complexity_metrics = self._calculate_basic_complexity(function)
        
        return context
    
    def _build_call_graph(self):
        """Build call graph for advanced analysis"""
        for name, context in self.function_contexts.items():
            # Add node
            self.call_graph.add_node(name, context=context)
            
            # Add edges for function calls
            for called_func in context.function_calls:
                if called_func in self.function_contexts:
                    self.call_graph.add_edge(name, called_func)
        
        # Update contexts with graph metrics
        for name, context in self.function_contexts.items():
            if name in self.call_graph:
                context.fan_out = self.call_graph.out_degree(name)
                context.fan_in = self.call_graph.in_degree(name)
                context.call_depth = self._calculate_call_depth(name)
                context.max_call_chain = self._get_max_call_chain(name)
    
    def _calculate_call_depth(self, function_name: str) -> int:
        """Calculate maximum call depth from this function"""
        try:
            if function_name not in self.call_graph:
                return 0
            
            # Use DFS to find maximum depth
            visited = set()
            
            def dfs(node, depth):
                if node in visited:
                    return depth  # Avoid cycles
                visited.add(node)
                
                max_depth = depth
                for successor in self.call_graph.successors(node):
                    max_depth = max(max_depth, dfs(successor, depth + 1))
                
                visited.remove(node)
                return max_depth
            
            return dfs(function_name, 0)
        except:
            return 0
    
    def _get_max_call_chain(self, function_name: str) -> List[str]:
        """Get the longest call chain starting from this function"""
        try:
            if function_name not in self.call_graph:
                return [function_name]
            
            # Find longest path using DFS
            longest_path = [function_name]
            visited = set()
            
            def dfs(node, current_path):
                nonlocal longest_path
                
                if node in visited:
                    return current_path
                
                visited.add(node)
                
                if len(current_path) > len(longest_path):
                    longest_path = current_path.copy()
                
                for successor in self.call_graph.successors(node):
                    dfs(successor, current_path + [successor])
                
                visited.remove(node)
                return current_path
            
            dfs(function_name, [function_name])
            return longest_path
        except:
            return [function_name]
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced metrics for all functions"""
        for name, context in self.function_contexts.items():
            # Coupling score (based on dependencies and calls)
            context.coupling_score = self._calculate_coupling_score(context)
            
            # Cohesion score (based on internal relationships)
            context.cohesion_score = self._calculate_cohesion_score(context)
            
            # Update complexity metrics
            context.complexity_metrics.update(self._calculate_advanced_complexity(context))
    
    def _calculate_coupling_score(self, context: FunctionContext) -> float:
        """Calculate coupling score (0-100, lower is better)"""
        # Factors: dependencies, function calls, parameters, fan-in/fan-out
        dependency_factor = len(context.dependencies) * 2
        calls_factor = len(context.function_calls) * 1.5
        params_factor = len(context.parameters) * 0.5
        fan_factor = (context.fan_in + context.fan_out) * 1
        
        coupling = dependency_factor + calls_factor + params_factor + fan_factor
        return min(100, coupling)
    
    def _calculate_cohesion_score(self, context: FunctionContext) -> float:
        """Calculate cohesion score (0-100, higher is better)"""
        # Simple heuristic based on function length and complexity
        if not context.source:
            return 50.0
        
        lines = len(context.source.split('\n'))
        complexity = context.complexity_metrics.get('cyclomatic_complexity', 1)
        
        # Ideal function: 10-30 lines, complexity 1-5
        line_score = 100 - abs(lines - 20) * 2  # Penalty for deviation from 20 lines
        complexity_score = 100 - (complexity - 3) * 10  # Penalty for high complexity
        
        cohesion = (line_score + complexity_score) / 2
        return max(0, min(100, cohesion))
    
    def _calculate_basic_complexity(self, function) -> Dict[str, Any]:
        """Calculate basic complexity metrics"""
        metrics = {
            'cyclomatic_complexity': 1,  # Default
            'lines_of_code': 0,
            'parameters_count': 0,
            'return_statements': 0
        }
        
        if hasattr(function, 'source') and function.source:
            # Lines of code
            metrics['lines_of_code'] = len(function.source.split('\n'))
            
            # Simple cyclomatic complexity (count decision points)
            decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
            complexity = 1  # Base complexity
            for keyword in decision_keywords:
                complexity += function.source.count(f' {keyword} ')
                complexity += function.source.count(f'\n{keyword} ')
                complexity += function.source.count(f'({keyword} ')
            
            metrics['cyclomatic_complexity'] = complexity
            
            # Return statements
            metrics['return_statements'] = function.source.count('return ')
        
        # Parameters
        if hasattr(function, 'parameters'):
            metrics['parameters_count'] = len(function.parameters)
        
        return metrics
    
    def _calculate_advanced_complexity(self, context: FunctionContext) -> Dict[str, Any]:
        """Calculate advanced complexity metrics"""
        advanced_metrics = {}
        
        # Nesting depth
        if context.source:
            advanced_metrics['max_nesting_depth'] = self._calculate_nesting_depth(context.source)
        
        # Call complexity (based on call chain)
        advanced_metrics['call_complexity'] = len(context.max_call_chain)
        
        # Dependency complexity
        advanced_metrics['dependency_complexity'] = len(context.dependencies)
        
        # Interface complexity (parameters + return complexity)
        interface_complexity = len(context.parameters)
        if context.return_type and context.return_type != 'None':
            interface_complexity += 1
        advanced_metrics['interface_complexity'] = interface_complexity
        
        return advanced_metrics
    
    def _calculate_nesting_depth(self, source: str) -> int:
        """Calculate maximum nesting depth in function"""
        lines = source.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
            
            # Calculate indentation level
            indent_level = (len(line) - len(stripped)) // 4  # Assuming 4-space indentation
            
            # Check for control structures
            if any(stripped.startswith(keyword) for keyword in ['if', 'for', 'while', 'try', 'with', 'def', 'class']):
                current_depth = indent_level + 1
                max_depth = max(max_depth, current_depth)
        
        return max_depth
    
    def _identify_function_patterns(self):
        """Identify common function patterns and anti-patterns"""
        for name, context in self.function_contexts.items():
            patterns = []
            
            # God function (too many responsibilities)
            if (context.complexity_metrics.get('cyclomatic_complexity', 0) > 20 or
                context.complexity_metrics.get('lines_of_code', 0) > 100):
                patterns.append('god_function')
            
            # Dead code
            if context.fan_in == 0 and not context.is_entry_point:
                patterns.append('dead_code')
                context.is_dead_code = True
            
            # Hub function (called by many)
            if context.fan_in > 10:
                patterns.append('hub_function')
            
            # Utility function (calls many others)
            if context.fan_out > 15:
                patterns.append('utility_function')
            
            # Pure function (no external dependencies)
            if len(context.dependencies) == 0 and len(context.function_calls) == 0:
                patterns.append('pure_function')
            
            # Entry point
            if context.is_entry_point:
                patterns.append('entry_point')
            
            # Complex interface
            if len(context.parameters) > 7:
                patterns.append('complex_interface')
            
            context.complexity_metrics['patterns'] = patterns
    
    def _is_entry_point(self, function) -> bool:
        """Check if function is an entry point"""
        entry_point_patterns = [
            'main', '__main__', 'run', 'start', 'execute', 'init',
            'setup', 'configure', 'app', 'server', 'cli', 'handler',
            'endpoint', 'route', 'view', 'controller'
        ]
        
        return any(pattern in function.name.lower() for pattern in entry_point_patterns)
    
    def get_function_relationships(self, function_name: str) -> Dict[str, Any]:
        """Get detailed relationships for a specific function"""
        if function_name not in self.function_contexts:
            return {}
        
        context = self.function_contexts[function_name]
        
        return {
            "direct_dependencies": context.dependencies,
            "direct_usages": context.usages,
            "calls_made": context.function_calls,
            "called_by": context.called_by,
            "call_chain": context.max_call_chain,
            "fan_in": context.fan_in,
            "fan_out": context.fan_out,
            "coupling_score": context.coupling_score,
            "cohesion_score": context.cohesion_score,
            "patterns": context.complexity_metrics.get('patterns', [])
        }
    
    def get_critical_functions(self) -> List[Dict[str, Any]]:
        """Get functions that are critical to the system"""
        critical_functions = []
        
        for name, context in self.function_contexts.items():
            # Critical if: high fan-in, entry point, or in long call chains
            is_critical = (
                context.fan_in > 5 or
                context.is_entry_point or
                len(context.max_call_chain) > 10 or
                'hub_function' in context.complexity_metrics.get('patterns', [])
            )
            
            if is_critical:
                critical_functions.append({
                    "name": name,
                    "filepath": context.filepath,
                    "fan_in": context.fan_in,
                    "fan_out": context.fan_out,
                    "call_depth": context.call_depth,
                    "complexity": context.complexity_metrics.get('cyclomatic_complexity', 0),
                    "patterns": context.complexity_metrics.get('patterns', []),
                    "is_entry_point": context.is_entry_point
                })
        
        # Sort by criticality score
        critical_functions.sort(key=lambda x: x['fan_in'] + x['call_depth'], reverse=True)
        return critical_functions

