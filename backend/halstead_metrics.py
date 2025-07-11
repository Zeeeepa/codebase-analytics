"""
Halstead Complexity Metrics Calculator
Comprehensive Halstead metrics for quantitative complexity analysis
"""

import re
import ast
from typing import Dict, List, Any, Tuple
from collections import Counter
from dataclasses import dataclass


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics for a code unit"""
    n1: int  # Number of distinct operators
    n2: int  # Number of distinct operands
    N1: int  # Total number of operators
    N2: int  # Total number of operands
    vocabulary: int  # n1 + n2
    length: int  # N1 + N2
    calculated_length: float  # n1 * log2(n1) + n2 * log2(n2)
    volume: float  # length * log2(vocabulary)
    difficulty: float  # (n1/2) * (N2/n2)
    effort: float  # difficulty * volume
    time: float  # effort / 18 (seconds)
    bugs: float  # volume / 3000 (estimated bugs)


class HalsteadCalculator:
    """Calculator for Halstead complexity metrics"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.python_operators = {
            # Arithmetic operators
            '+', '-', '*', '/', '//', '%', '**',
            # Assignment operators
            '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
            # Comparison operators
            '==', '!=', '<', '>', '<=', '>=',
            # Logical operators
            'and', 'or', 'not',
            # Bitwise operators
            '&', '|', '^', '~', '<<', '>>',
            # Membership operators
            'in', 'not in',
            # Identity operators
            'is', 'is not',
            # Control flow
            'if', 'elif', 'else', 'for', 'while', 'break', 'continue',
            'try', 'except', 'finally', 'raise', 'with', 'as',
            'def', 'class', 'return', 'yield', 'yield from',
            'import', 'from', 'global', 'nonlocal',
            'lambda', 'pass', 'del', 'assert',
            # Brackets and delimiters
            '(', ')', '[', ']', '{', '}', ',', ':', ';', '.', '->'
        }
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate Halstead metrics for entire codebase"""
        print("📊 Calculating Halstead complexity metrics...")
        
        file_metrics = {}
        function_metrics = {}
        overall_metrics = {
            'operators': Counter(),
            'operands': Counter()
        }
        
        # Calculate metrics for each file
        for file in self.codebase.files:
            if hasattr(file, 'source') and file.source:
                file_halstead = self._calculate_file_metrics(file)
                file_metrics[file.filepath] = file_halstead
                
                # Aggregate for overall metrics
                overall_metrics['operators'].update(file_halstead.operators)
                overall_metrics['operands'].update(file_halstead.operands)
        
        # Calculate metrics for each function
        for function in self.codebase.functions:
            if hasattr(function, 'source') and function.source:
                func_halstead = self._calculate_function_metrics(function)
                function_metrics[function.name] = func_halstead
        
        # Calculate overall codebase metrics
        overall_halstead = self._calculate_halstead_from_counters(
            overall_metrics['operators'],
            overall_metrics['operands']
        )
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(file_metrics, function_metrics)
        
        result = {
            'overall_metrics': self._halstead_to_dict(overall_halstead),
            'file_metrics': {path: self._halstead_to_dict(metrics) for path, metrics in file_metrics.items()},
            'function_metrics': {name: self._halstead_to_dict(metrics) for name, metrics in function_metrics.items()},
            'summary_statistics': summary_stats,
            'complexity_distribution': self._calculate_complexity_distribution(function_metrics),
            'quality_indicators': self._calculate_quality_indicators(overall_halstead, function_metrics)
        }
        
        print(f"✅ Calculated Halstead metrics for {len(file_metrics)} files and {len(function_metrics)} functions")
        return result
    
    def _calculate_file_metrics(self, file) -> HalsteadMetrics:
        """Calculate Halstead metrics for a single file"""
        operators, operands = self._extract_operators_operands(file.source)
        return self._calculate_halstead_from_counters(operators, operands)
    
    def _calculate_function_metrics(self, function) -> HalsteadMetrics:
        """Calculate Halstead metrics for a single function"""
        operators, operands = self._extract_operators_operands(function.source)
        return self._calculate_halstead_from_counters(operators, operands)
    
    def _extract_operators_operands(self, source_code: str) -> Tuple[Counter, Counter]:
        """Extract operators and operands from source code"""
        operators = Counter()
        operands = Counter()
        
        try:
            # Parse the source code into AST
            tree = ast.parse(source_code)
            
            # Walk the AST to extract operators and operands
            for node in ast.walk(tree):
                self._process_ast_node(node, operators, operands)
            
        except SyntaxError:
            # Fallback to regex-based extraction for malformed code
            operators, operands = self._extract_with_regex(source_code)
        
        return operators, operands
    
    def _process_ast_node(self, node: ast.AST, operators: Counter, operands: Counter):
        """Process an AST node to extract operators and operands"""
        # Operators
        if isinstance(node, ast.Add):
            operators['+'] += 1
        elif isinstance(node, ast.Sub):
            operators['-'] += 1
        elif isinstance(node, ast.Mult):
            operators['*'] += 1
        elif isinstance(node, ast.Div):
            operators['/'] += 1
        elif isinstance(node, ast.FloorDiv):
            operators['//'] += 1
        elif isinstance(node, ast.Mod):
            operators['%'] += 1
        elif isinstance(node, ast.Pow):
            operators['**'] += 1
        elif isinstance(node, ast.Eq):
            operators['=='] += 1
        elif isinstance(node, ast.NotEq):
            operators['!='] += 1
        elif isinstance(node, ast.Lt):
            operators['<'] += 1
        elif isinstance(node, ast.LtE):
            operators['<='] += 1
        elif isinstance(node, ast.Gt):
            operators['>'] += 1
        elif isinstance(node, ast.GtE):
            operators['>='] += 1
        elif isinstance(node, ast.And):
            operators['and'] += 1
        elif isinstance(node, ast.Or):
            operators['or'] += 1
        elif isinstance(node, ast.Not):
            operators['not'] += 1
        elif isinstance(node, ast.In):
            operators['in'] += 1
        elif isinstance(node, ast.NotIn):
            operators['not in'] += 1
        elif isinstance(node, ast.Is):
            operators['is'] += 1
        elif isinstance(node, ast.IsNot):
            operators['is not'] += 1
        elif isinstance(node, ast.Assign):
            operators['='] += 1
        elif isinstance(node, ast.AugAssign):
            # Handle augmented assignments (+=, -=, etc.)
            op_map = {
                ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/=',
                ast.FloorDiv: '//=', ast.Mod: '%=', ast.Pow: '**='
            }
            op_symbol = op_map.get(type(node.op), '=')
            operators[op_symbol] += 1
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            # Control flow operators
            operators[node.__class__.__name__.lower()] += 1
        elif isinstance(node, ast.FunctionDef):
            operators['def'] += 1
        elif isinstance(node, ast.ClassDef):
            operators['class'] += 1
        elif isinstance(node, ast.Return):
            operators['return'] += 1
        elif isinstance(node, ast.Import):
            operators['import'] += 1
        elif isinstance(node, ast.ImportFrom):
            operators['from'] += 1
        
        # Operands
        if isinstance(node, ast.Name):
            # Variable names
            operands[node.id] += 1
        elif isinstance(node, ast.Constant):
            # Constants (numbers, strings, etc.)
            if isinstance(node.value, (int, float)):
                operands[f'NUM_{node.value}'] += 1
            elif isinstance(node.value, str):
                operands[f'STR_{len(node.value)}'] += 1  # Use length to avoid storing actual strings
            else:
                operands[f'CONST_{type(node.value).__name__}'] += 1
        elif isinstance(node, ast.Attribute):
            # Attribute access (obj.attr)
            operands[node.attr] += 1
    
    def _extract_with_regex(self, source_code: str) -> Tuple[Counter, Counter]:
        """Fallback regex-based extraction for malformed code"""
        operators = Counter()
        operands = Counter()
        
        # Extract operators using regex
        for operator in self.python_operators:
            if operator.isalpha():
                # Word operators (and, or, not, etc.)
                pattern = r'\b' + re.escape(operator) + r'\b'
            else:
                # Symbol operators
                pattern = re.escape(operator)
            
            matches = re.findall(pattern, source_code)
            if matches:
                operators[operator] += len(matches)
        
        # Extract operands (identifiers and literals)
        # Variable names
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = re.findall(var_pattern, source_code)
        for var in variables:
            if var not in self.python_operators:
                operands[var] += 1
        
        # Numbers
        num_pattern = r'\b\d+\.?\d*\b'
        numbers = re.findall(num_pattern, source_code)
        for num in numbers:
            operands[f'NUM_{num}'] += 1
        
        # Strings (simplified)
        str_pattern = r'["\'][^"\']*["\']'
        strings = re.findall(str_pattern, source_code)
        for i, string in enumerate(strings):
            operands[f'STR_{len(string)}'] += 1
        
        return operators, operands
    
    def _calculate_halstead_from_counters(self, operators: Counter, operands: Counter) -> HalsteadMetrics:
        """Calculate Halstead metrics from operator and operand counters"""
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        N1 = sum(operators.values())  # Total operators
        N2 = sum(operands.values())   # Total operands
        
        if n1 == 0 and n2 == 0:
            # Empty code
            return HalsteadMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        vocabulary = n1 + n2
        length = N1 + N2
        
        # Calculated length (Halstead's formula)
        import math
        calculated_length = 0
        if n1 > 0:
            calculated_length += n1 * math.log2(n1)
        if n2 > 0:
            calculated_length += n2 * math.log2(n2)
        
        # Volume
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        
        # Difficulty
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        
        # Effort
        effort = difficulty * volume
        
        # Time (in seconds, assuming 18 mental discriminations per second)
        time = effort / 18 if effort > 0 else 0
        
        # Bugs (Halstead's formula for delivered bugs)
        bugs = volume / 3000 if volume > 0 else 0
        
        return HalsteadMetrics(
            n1=n1, n2=n2, N1=N1, N2=N2,
            vocabulary=vocabulary, length=length,
            calculated_length=calculated_length,
            volume=volume, difficulty=difficulty,
            effort=effort, time=time, bugs=bugs
        )
    
    def _halstead_to_dict(self, metrics: HalsteadMetrics) -> Dict[str, Any]:
        """Convert HalsteadMetrics to dictionary"""
        return {
            'distinct_operators': metrics.n1,
            'distinct_operands': metrics.n2,
            'total_operators': metrics.N1,
            'total_operands': metrics.N2,
            'vocabulary': metrics.vocabulary,
            'length': metrics.length,
            'calculated_length': round(metrics.calculated_length, 2),
            'volume': round(metrics.volume, 2),
            'difficulty': round(metrics.difficulty, 2),
            'effort': round(metrics.effort, 2),
            'time_seconds': round(metrics.time, 2),
            'estimated_bugs': round(metrics.bugs, 4)
        }
    
    def _calculate_summary_statistics(self, file_metrics: Dict[str, HalsteadMetrics], 
                                    function_metrics: Dict[str, HalsteadMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics across all metrics"""
        all_volumes = []
        all_difficulties = []
        all_efforts = []
        all_bugs = []
        
        # Collect metrics from functions (more granular)
        for metrics in function_metrics.values():
            all_volumes.append(metrics.volume)
            all_difficulties.append(metrics.difficulty)
            all_efforts.append(metrics.effort)
            all_bugs.append(metrics.bugs)
        
        if not all_volumes:
            return {}
        
        return {
            'average_volume': round(sum(all_volumes) / len(all_volumes), 2),
            'max_volume': round(max(all_volumes), 2),
            'min_volume': round(min(all_volumes), 2),
            'average_difficulty': round(sum(all_difficulties) / len(all_difficulties), 2),
            'max_difficulty': round(max(all_difficulties), 2),
            'min_difficulty': round(min(all_difficulties), 2),
            'average_effort': round(sum(all_efforts) / len(all_efforts), 2),
            'total_estimated_bugs': round(sum(all_bugs), 4),
            'functions_analyzed': len(function_metrics),
            'files_analyzed': len(file_metrics)
        }
    
    def _calculate_complexity_distribution(self, function_metrics: Dict[str, HalsteadMetrics]) -> Dict[str, Any]:
        """Calculate distribution of complexity across functions"""
        if not function_metrics:
            return {}
        
        volumes = [m.volume for m in function_metrics.values()]
        difficulties = [m.difficulty for m in function_metrics.values()]
        
        # Categorize by volume
        low_volume = len([v for v in volumes if v < 100])
        medium_volume = len([v for v in volumes if 100 <= v < 500])
        high_volume = len([v for v in volumes if v >= 500])
        
        # Categorize by difficulty
        low_difficulty = len([d for d in difficulties if d < 10])
        medium_difficulty = len([d for d in difficulties if 10 <= d < 30])
        high_difficulty = len([d for d in difficulties if d >= 30])
        
        return {
            'volume_distribution': {
                'low': low_volume,
                'medium': medium_volume,
                'high': high_volume
            },
            'difficulty_distribution': {
                'low': low_difficulty,
                'medium': medium_difficulty,
                'high': high_difficulty
            },
            'total_functions': len(function_metrics)
        }
    
    def _calculate_quality_indicators(self, overall_metrics: HalsteadMetrics, 
                                    function_metrics: Dict[str, HalsteadMetrics]) -> Dict[str, Any]:
        """Calculate quality indicators based on Halstead metrics"""
        if not function_metrics:
            return {}
        
        # Find functions with concerning metrics
        high_volume_functions = []
        high_difficulty_functions = []
        high_effort_functions = []
        
        for name, metrics in function_metrics.items():
            if metrics.volume > 1000:  # High volume threshold
                high_volume_functions.append({
                    'name': name,
                    'volume': round(metrics.volume, 2)
                })
            
            if metrics.difficulty > 50:  # High difficulty threshold
                high_difficulty_functions.append({
                    'name': name,
                    'difficulty': round(metrics.difficulty, 2)
                })
            
            if metrics.effort > 50000:  # High effort threshold
                high_effort_functions.append({
                    'name': name,
                    'effort': round(metrics.effort, 2)
                })
        
        # Sort by metric value
        high_volume_functions.sort(key=lambda x: x['volume'], reverse=True)
        high_difficulty_functions.sort(key=lambda x: x['difficulty'], reverse=True)
        high_effort_functions.sort(key=lambda x: x['effort'], reverse=True)
        
        return {
            'high_volume_functions': high_volume_functions[:10],  # Top 10
            'high_difficulty_functions': high_difficulty_functions[:10],
            'high_effort_functions': high_effort_functions[:10],
            'overall_maintainability': self._calculate_maintainability_index(overall_metrics),
            'quality_score': self._calculate_quality_score(overall_metrics, function_metrics)
        }
    
    def _calculate_maintainability_index(self, metrics: HalsteadMetrics) -> float:
        """Calculate maintainability index based on Halstead metrics"""
        if metrics.volume == 0:
            return 100.0
        
        # Simplified maintainability index
        # MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        # Where V = volume, G = cyclomatic complexity, LOC = lines of code
        # Since we don't have G and LOC here, we'll use a simplified version
        
        import math
        mi = max(0, 171 - 5.2 * math.log(metrics.volume) - 0.1 * metrics.difficulty)
        return round(mi, 2)
    
    def _calculate_quality_score(self, overall_metrics: HalsteadMetrics, 
                               function_metrics: Dict[str, HalsteadMetrics]) -> float:
        """Calculate overall quality score (0-100)"""
        if not function_metrics:
            return 50.0
        
        # Base score
        score = 100.0
        
        # Penalize high average difficulty
        avg_difficulty = sum(m.difficulty for m in function_metrics.values()) / len(function_metrics)
        difficulty_penalty = min(avg_difficulty / 2, 30)
        
        # Penalize high average volume
        avg_volume = sum(m.volume for m in function_metrics.values()) / len(function_metrics)
        volume_penalty = min(avg_volume / 100, 25)
        
        # Penalize high estimated bugs
        total_bugs = sum(m.bugs for m in function_metrics.values())
        bugs_penalty = min(total_bugs * 10, 20)
        
        quality_score = score - difficulty_penalty - volume_penalty - bugs_penalty
        return max(0, min(100, round(quality_score, 2)))

