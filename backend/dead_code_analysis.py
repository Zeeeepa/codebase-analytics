"""
Dead Code Analysis with Blast Radius Calculation
Intelligent dead code detection with impact assessment
"""

from typing import Dict, List, Any, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DeadCodeItem:
    """Represents a dead code item"""
    name: str
    type: str  # function, class, variable, import
    filepath: str
    line_number: int
    reason: str
    confidence: float
    blast_radius: List[str]
    safe_to_remove: bool


class DeadCodeAnalyzer:
    """Analyzer for dead code detection with blast radius calculation"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.dead_code_items = []
        self.usage_map = defaultdict(set)
        self.dependency_map = defaultdict(set)
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze dead code and return comprehensive results"""
        print("💀 Analyzing dead code...")
        
        # Build usage and dependency maps
        self._build_usage_maps()
        
        # Find dead code
        self._find_dead_functions()
        self._find_dead_classes()
        self._find_dead_variables()
        self._find_dead_imports()
        
        # Calculate blast radius for each item
        self._calculate_blast_radius()
        
        # Generate analysis results
        results = self._generate_analysis_results()
        
        print(f"✅ Dead code analysis complete: {len(self.dead_code_items)} items found")
        return results
    
    def _build_usage_maps(self):
        """Build maps of symbol usage and dependencies"""
        # Map function usages
        for function in self.codebase.functions:
            if hasattr(function, 'usages'):
                for usage in function.usages:
                    if hasattr(usage, 'usage_symbol'):
                        self.usage_map[function.name].add(usage.usage_symbol.filepath)
            
            if hasattr(function, 'call_sites'):
                for call_site in function.call_sites:
                    if hasattr(call_site, 'parent_function'):
                        self.usage_map[function.name].add(call_site.parent_function.name)
        
        # Map dependencies
        for function in self.codebase.functions:
            if hasattr(function, 'dependencies'):
                for dep in function.dependencies:
                    if hasattr(dep, 'name'):
                        self.dependency_map[function.name].add(dep.name)
    
    def _find_dead_functions(self):
        """Find unused functions"""
        for function in self.codebase.functions:
            # Check if function has any usages
            if function.name not in self.usage_map or len(self.usage_map[function.name]) == 0:
                # Check if it's not an entry point
                if not self._is_entry_point(function):
                    confidence = 0.8
                    
                    # Lower confidence if it's a public method or has special naming
                    if (function.name.startswith('_') or 
                        function.name in ['__init__', '__str__', '__repr__'] or
                        'test' in function.name.lower()):
                        confidence = 0.6
                    
                    dead_item = DeadCodeItem(
                        name=function.name,
                        type='function',
                        filepath=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        reason='No usages found',
                        confidence=confidence,
                        blast_radius=[],
                        safe_to_remove=confidence > 0.7
                    )
                    
                    self.dead_code_items.append(dead_item)
    
    def _find_dead_classes(self):
        """Find unused classes"""
        for cls in self.codebase.classes:
            # Check if class has any usages
            if cls.name not in self.usage_map or len(self.usage_map[cls.name]) == 0:
                # Check if it's not a base class or has special methods
                confidence = 0.7
                
                if hasattr(cls, 'methods'):
                    # Lower confidence if it has special methods
                    special_methods = ['__init__', '__str__', '__repr__', '__call__']
                    if any(method.name in special_methods for method in cls.methods if hasattr(method, 'name')):
                        confidence = 0.5
                
                dead_item = DeadCodeItem(
                    name=cls.name,
                    type='class',
                    filepath=cls.filepath,
                    line_number=cls.start_point[0] if hasattr(cls, 'start_point') else 0,
                    reason='No instantiations or references found',
                    confidence=confidence,
                    blast_radius=[],
                    safe_to_remove=confidence > 0.6
                )
                
                self.dead_code_items.append(dead_item)
    
    def _find_dead_variables(self):
        """Find unused variables (simplified analysis)"""
        # This would require more sophisticated AST analysis
        # For now, we'll implement a basic version
        pass
    
    def _find_dead_imports(self):
        """Find unused imports"""
        for file in self.codebase.files:
            if hasattr(file, 'imports') and hasattr(file, 'source'):
                for imp in file.imports:
                    if hasattr(imp, 'module_name'):
                        module_name = imp.module_name
                        
                        # Check if import is used in the file
                        if module_name not in file.source:
                            dead_item = DeadCodeItem(
                                name=module_name,
                                type='import',
                                filepath=file.filepath,
                                line_number=getattr(imp, 'line_number', 1),
                                reason='Import not used in file',
                                confidence=0.9,
                                blast_radius=[],
                                safe_to_remove=True
                            )
                            
                            self.dead_code_items.append(dead_item)
    
    def _is_entry_point(self, function) -> bool:
        """Check if function is an entry point"""
        entry_patterns = [
            'main', '__main__', 'run', 'start', 'execute', 'init',
            'setup', 'configure', 'app', 'server', 'cli', 'handler',
            'endpoint', 'route', 'view', 'controller', 'test_'
        ]
        
        return any(pattern in function.name.lower() for pattern in entry_patterns)
    
    def _calculate_blast_radius(self):
        """Calculate blast radius for each dead code item"""
        for item in self.dead_code_items:
            blast_radius = self._calculate_item_blast_radius(item)
            item.blast_radius = blast_radius
            
            # Adjust safety based on blast radius
            if len(blast_radius) > 5:
                item.safe_to_remove = False
                item.confidence *= 0.7
    
    def _calculate_item_blast_radius(self, item: DeadCodeItem) -> List[str]:
        """Calculate blast radius for a specific dead code item"""
        blast_radius = []
        
        if item.type == 'function':
            # Find functions that would be affected if this function is removed
            
            # 1. Functions that this function calls (might become orphaned)
            if item.name in self.dependency_map:
                for dep in self.dependency_map[item.name]:
                    # Check if this dependency would become orphaned
                    if self._would_become_orphaned(dep, item.name):
                        blast_radius.append(f"function:{dep}")
            
            # 2. Functions that call this function (would break)
            for func_name, usages in self.usage_map.items():
                if item.name in usages:
                    blast_radius.append(f"caller:{func_name}")
        
        elif item.type == 'class':
            # Find code that would be affected if this class is removed
            
            # 1. Methods of this class
            for function in self.codebase.functions:
                if (hasattr(function, 'parent_class') and 
                    function.parent_class and 
                    function.parent_class.name == item.name):
                    blast_radius.append(f"method:{function.name}")
            
            # 2. Code that uses this class
            for func_name, usages in self.usage_map.items():
                if item.name in usages:
                    blast_radius.append(f"user:{func_name}")
        
        elif item.type == 'import':
            # Find code that might be affected by removing this import
            # This is usually safe, but we'll check for dynamic usage
            pass
        
        return blast_radius
    
    def _would_become_orphaned(self, function_name: str, removing_function: str) -> bool:
        """Check if a function would become orphaned if another function is removed"""
        if function_name not in self.usage_map:
            return True
        
        usages = self.usage_map[function_name]
        # If the only usage is from the function being removed, it would become orphaned
        return len(usages) == 1 and removing_function in usages
    
    def calculate_blast_radius(self) -> Dict[str, List[str]]:
        """Calculate blast radius for all dead code items"""
        blast_radius_map = {}
        
        for item in self.dead_code_items:
            blast_radius_map[f"{item.type}:{item.name}"] = item.blast_radius
        
        return blast_radius_map
    
    def _generate_analysis_results(self) -> Dict[str, Any]:
        """Generate comprehensive analysis results"""
        # Group by type
        by_type = defaultdict(list)
        for item in self.dead_code_items:
            by_type[item.type].append(item)
        
        # Calculate statistics
        total_items = len(self.dead_code_items)
        safe_to_remove = len([item for item in self.dead_code_items if item.safe_to_remove])
        high_confidence = len([item for item in self.dead_code_items if item.confidence > 0.8])
        
        # Find items with largest blast radius
        largest_blast_radius = sorted(
            self.dead_code_items, 
            key=lambda x: len(x.blast_radius), 
            reverse=True
        )[:10]
        
        # Calculate potential savings
        potential_loc_savings = self._calculate_potential_savings()
        
        return {
            'summary': {
                'total_dead_code_items': total_items,
                'safe_to_remove': safe_to_remove,
                'high_confidence_items': high_confidence,
                'potential_loc_savings': potential_loc_savings
            },
            'by_type': {
                'functions': len(by_type['function']),
                'classes': len(by_type['class']),
                'variables': len(by_type['variable']),
                'imports': len(by_type['import'])
            },
            'detailed_items': [self._serialize_dead_code_item(item) for item in self.dead_code_items],
            'high_impact_items': [
                self._serialize_dead_code_item(item) 
                for item in largest_blast_radius
            ],
            'safe_removal_candidates': [
                self._serialize_dead_code_item(item) 
                for item in self.dead_code_items 
                if item.safe_to_remove
            ],
            'removal_recommendations': self._generate_removal_recommendations()
        }
    
    def _calculate_potential_savings(self) -> int:
        """Calculate potential lines of code savings"""
        total_savings = 0
        
        for item in self.dead_code_items:
            if item.type == 'function':
                # Estimate function size
                for function in self.codebase.functions:
                    if function.name == item.name:
                        if hasattr(function, 'source'):
                            total_savings += len(function.source.split('\n'))
                        else:
                            total_savings += 10  # Estimate
                        break
            elif item.type == 'class':
                # Estimate class size
                for cls in self.codebase.classes:
                    if cls.name == item.name:
                        if hasattr(cls, 'source'):
                            total_savings += len(cls.source.split('\n'))
                        else:
                            total_savings += 20  # Estimate
                        break
            elif item.type == 'import':
                total_savings += 1  # One line per import
        
        return total_savings
    
    def _serialize_dead_code_item(self, item: DeadCodeItem) -> Dict[str, Any]:
        """Serialize dead code item for JSON output"""
        return {
            'name': item.name,
            'type': item.type,
            'filepath': item.filepath,
            'line_number': item.line_number,
            'reason': item.reason,
            'confidence': round(item.confidence, 2),
            'blast_radius': item.blast_radius,
            'blast_radius_size': len(item.blast_radius),
            'safe_to_remove': item.safe_to_remove
        }
    
    def _generate_removal_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized removal recommendations"""
        recommendations = []
        
        # Sort by safety and confidence
        sorted_items = sorted(
            self.dead_code_items,
            key=lambda x: (x.safe_to_remove, x.confidence, -len(x.blast_radius)),
            reverse=True
        )
        
        for i, item in enumerate(sorted_items[:20]):  # Top 20 recommendations
            priority = 'high' if item.safe_to_remove and item.confidence > 0.8 else 'medium'
            if len(item.blast_radius) > 3:
                priority = 'low'
            
            recommendations.append({
                'rank': i + 1,
                'item': self._serialize_dead_code_item(item),
                'priority': priority,
                'action': f"Remove {item.type} '{item.name}' from {item.filepath}",
                'risk_level': 'low' if item.safe_to_remove else 'medium'
            })
        
        return recommendations

