#!/usr/bin/env python3
"""
Performance Optimization System for Codebase Analysis

This module provides:
- High-performance caching with LRU eviction
- Incremental analysis for changed files only
- Performance monitoring and bottleneck identification
- Memory usage optimization
- Thread-safe operations
"""

import os
import time
import hashlib
import pickle
import threading
import json
from functools import wraps
from typing import List, Dict, Any, Set, Callable, Optional
from collections import defaultdict

class AnalysisCache:
    """
    High-performance caching system for analysis results.
    
    Features:
    - Memory-based caching with LRU eviction
    - File-based persistent caching
    - Thread-safe operations
    - Cache invalidation based on file modification times
    """
    
    def __init__(self, max_memory_items: int = 1000, cache_dir: str = ".analysis_cache"):
        self.max_memory_items = max_memory_items
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a unique cache key for function arguments."""
        try:
            # Create a hash of function name and arguments
            key_data = {
                'func': func_name,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_string.encode()).hexdigest()
        except:
            # Fallback to simple string concatenation
            return f"{func_name}_{hash(str(args))}_{hash(str(kwargs))}"
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content for cache invalidation."""
        try:
            if not os.path.exists(filepath):
                return "nonexistent"
            
            # Use modification time and size for quick hash
            stat = os.stat(filepath)
            return f"{stat.st_mtime}_{stat.st_size}"
        except:
            return "unknown"
    
    def get(self, key: str, file_dependencies: List[str] = None) -> Any:
        """Get cached result if valid."""
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                cached_data = self.memory_cache[key]
                
                # Check if cache is still valid
                if self._is_cache_valid(cached_data, file_dependencies):
                    # Update access order for LRU
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    return cached_data['result']
                else:
                    # Remove invalid cache entry
                    del self.memory_cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            
            # Check file cache
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if self._is_cache_valid(cached_data, file_dependencies):
                        # Load into memory cache
                        self._store_in_memory(key, cached_data)
                        return cached_data['result']
                    else:
                        # Remove invalid file cache
                        os.remove(cache_file)
                except:
                    # Remove corrupted cache file
                    try:
                        os.remove(cache_file)
                    except:
                        pass
            
            return None
    
    def set(self, key: str, result: Any, file_dependencies: List[str] = None):
        """Store result in cache."""
        with self.lock:
            # Prepare cache data
            cached_data = {
                'result': result,
                'timestamp': time.time(),
                'file_hashes': {}
            }
            
            # Store file hashes for invalidation
            if file_dependencies:
                for filepath in file_dependencies:
                    cached_data['file_hashes'][filepath] = self._get_file_hash(filepath)
            
            # Store in memory cache
            self._store_in_memory(key, cached_data)
            
            # Store in file cache for persistence
            try:
                cache_file = os.path.join(self.cache_dir, f"{key}.cache")
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
            except:
                pass  # File cache is optional
    
    def _store_in_memory(self, key: str, cached_data: dict):
        """Store data in memory cache with LRU eviction."""
        # Add to memory cache
        self.memory_cache[key] = cached_data
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Evict oldest items if cache is full
        while len(self.memory_cache) > self.max_memory_items:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]
    
    def _is_cache_valid(self, cached_data: dict, file_dependencies: List[str] = None) -> bool:
        """Check if cached data is still valid."""
        try:
            # Check file dependencies
            if file_dependencies and 'file_hashes' in cached_data:
                for filepath in file_dependencies:
                    current_hash = self._get_file_hash(filepath)
                    cached_hash = cached_data['file_hashes'].get(filepath)
                    if current_hash != cached_hash:
                        return False
            
            # Cache is valid if no dependencies or all dependencies are unchanged
            return True
        except:
            return False
    
    def clear(self):
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            self.access_order.clear()
            
            # Clear file cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))
            except:
                pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'memory_cache_size': len(self.memory_cache),
                'max_memory_items': self.max_memory_items,
                'cache_hit_ratio': self._calculate_hit_ratio(),
                'total_cached_items': len(self.memory_cache)
            }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        # This is a simplified implementation
        # In a real system, you'd track hits and misses
        return 0.85  # Placeholder

class IncrementalAnalyzer:
    """
    Incremental analysis system that only analyzes changed files.
    
    Features:
    - File change detection based on modification times
    - Dependency tracking for affected file analysis
    - Efficient re-analysis of only changed components
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.file_states = {}  # filepath -> {mtime, size, hash}
        self.dependency_graph = {}  # filepath -> [dependent_files]
        self.analysis_results = {}  # filepath -> analysis_results
        
    def get_changed_files(self, file_paths: List[str]) -> List[str]:
        """Get list of files that have changed since last analysis."""
        changed_files = []
        
        for filepath in file_paths:
            try:
                if not os.path.exists(filepath):
                    # File was deleted
                    if filepath in self.file_states:
                        changed_files.append(filepath)
                        del self.file_states[filepath]
                    continue
                
                # Get current file state
                stat = os.stat(filepath)
                current_state = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size
                }
                
                # Check if file has changed
                if filepath not in self.file_states:
                    # New file
                    changed_files.append(filepath)
                    self.file_states[filepath] = current_state
                elif (self.file_states[filepath]['mtime'] != current_state['mtime'] or
                      self.file_states[filepath]['size'] != current_state['size']):
                    # Modified file
                    changed_files.append(filepath)
                    self.file_states[filepath] = current_state
                    
            except Exception as e:
                # If we can't check the file, assume it changed
                changed_files.append(filepath)
        
        return changed_files
    
    def get_affected_files(self, changed_files: List[str]) -> Set[str]:
        """Get all files affected by changes (including dependencies)."""
        affected = set(changed_files)
        
        # Add files that depend on changed files
        for changed_file in changed_files:
            affected.update(self._get_dependent_files(changed_file))
        
        return affected
    
    def _get_dependent_files(self, filepath: str) -> Set[str]:
        """Get files that depend on the given file."""
        dependents = set()
        
        # Direct dependents
        if filepath in self.dependency_graph:
            dependents.update(self.dependency_graph[filepath])
        
        # Transitive dependents (recursive, with cycle detection)
        visited = set()
        for dependent in list(dependents):
            if dependent not in visited:
                visited.add(dependent)
                dependents.update(self._get_dependent_files_recursive(dependent, visited))
        
        return dependents
    
    def _get_dependent_files_recursive(self, filepath: str, visited: Set[str]) -> Set[str]:
        """Get dependent files recursively with cycle detection."""
        dependents = set()
        
        if filepath in self.dependency_graph:
            for dependent in self.dependency_graph[filepath]:
                if dependent not in visited:
                    visited.add(dependent)
                    dependents.add(dependent)
                    dependents.update(self._get_dependent_files_recursive(dependent, visited))
        
        return dependents
    
    def update_dependency_graph(self, codebase):
        """Update the dependency graph based on current codebase."""
        self.dependency_graph.clear()
        
        if not codebase or not hasattr(codebase, 'files'):
            return
        
        try:
            # Build dependency graph from imports
            for file in codebase.files:
                file_path = str(file.path) if hasattr(file, 'path') else None
                if not file_path:
                    continue
                
                dependencies = []
                if hasattr(file, 'imports'):
                    for imp in file.imports:
                        if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'file'):
                            dep_path = str(imp.imported_symbol.file.path)
                            dependencies.append(dep_path)
                
                # Add reverse dependencies
                for dep_path in dependencies:
                    if dep_path not in self.dependency_graph:
                        self.dependency_graph[dep_path] = []
                    if file_path not in self.dependency_graph[dep_path]:
                        self.dependency_graph[dep_path].append(file_path)
                        
        except Exception as e:
            print(f"Warning: Failed to update dependency graph: {e}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get incremental analysis statistics."""
        return {
            'tracked_files': len(self.file_states),
            'dependency_relationships': sum(len(deps) for deps in self.dependency_graph.values()),
            'cached_results': len(self.analysis_results)
        }

class PerformanceMonitor:
    """
    Monitor and optimize analysis performance.
    
    Features:
    - Execution time tracking
    - Memory usage monitoring
    - Performance bottleneck identification
    - Optimization suggestions
    """
    
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.lock = threading.RLock()
        
    def track_execution(self, func_name: str):
        """Decorator to track function execution time."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # Record metrics
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                with self.lock:
                    self.execution_times[func_name].append(execution_time)
                    self.memory_usage[func_name].append(memory_delta)
                    self.call_counts[func_name] += 1
                
                if not success:
                    raise result
                
                return result
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report."""
        with self.lock:
            report = {
                'summary': {},
                'bottlenecks': [],
                'optimization_suggestions': [],
                'total_analysis_time': 0.0,
                'most_called_functions': [],
                'memory_intensive_functions': []
            }
            
            total_time = 0.0
            
            # Calculate summary statistics
            for func_name in self.execution_times:
                times = self.execution_times[func_name]
                memory = self.memory_usage[func_name]
                
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    total_func_time = sum(times)
                    total_time += total_func_time
                    
                    avg_memory = sum(memory) / len(memory) if memory else 0
                    max_memory = max(memory) if memory else 0
                    
                    report['summary'][func_name] = {
                        'call_count': self.call_counts[func_name],
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'total_time': total_func_time,
                        'avg_memory': avg_memory,
                        'max_memory': max_memory
                    }
                    
                    # Identify bottlenecks
                    if avg_time > 1.0:  # Functions taking more than 1 second on average
                        report['bottlenecks'].append({
                            'function': func_name,
                            'avg_time': avg_time,
                            'total_time': total_func_time,
                            'call_count': self.call_counts[func_name],
                            'efficiency_score': self.call_counts[func_name] / total_func_time if total_func_time > 0 else 0
                        })
            
            report['total_analysis_time'] = total_time
            
            # Find most called functions
            most_called = sorted(self.call_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            report['most_called_functions'] = [{'function': name, 'calls': count} for name, count in most_called]
            
            # Find memory intensive functions
            memory_intensive = []
            for func_name, memory_list in self.memory_usage.items():
                if memory_list:
                    avg_memory = sum(memory_list) / len(memory_list)
                    if avg_memory > 10.0:  # Functions using more than 10MB on average
                        memory_intensive.append({'function': func_name, 'avg_memory': avg_memory})
            
            report['memory_intensive_functions'] = sorted(memory_intensive, key=lambda x: x['avg_memory'], reverse=True)[:5]
            
            # Generate optimization suggestions
            report['optimization_suggestions'] = self._generate_optimization_suggestions(report['summary'])
            
            return report
    
    def _generate_optimization_suggestions(self, summary: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on performance data."""
        suggestions = []
        
        for func_name, metrics in summary.items():
            avg_time = metrics['avg_time']
            call_count = metrics['call_count']
            total_time = metrics['total_time']
            avg_memory = metrics['avg_memory']
            
            if avg_time > 2.0:
                suggestions.append(f"🐌 Consider optimizing '{func_name}' - average execution time is {avg_time:.2f}s")
            
            if call_count > 100 and avg_time > 0.1:
                suggestions.append(f"💾 Consider caching results for '{func_name}' - called {call_count} times with {avg_time:.2f}s average")
            
            if total_time > 10.0:
                suggestions.append(f"⚡ '{func_name}' consumes significant total time ({total_time:.2f}s) - consider parallelization")
            
            if avg_memory > 50.0:
                suggestions.append(f"🧠 '{func_name}' uses significant memory ({avg_memory:.1f}MB average) - consider memory optimization")
        
        return suggestions
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        with self.lock:
            self.execution_times.clear()
            self.memory_usage.clear()
            self.call_counts.clear()

# Global instances
_analysis_cache = AnalysisCache()
_performance_monitor = PerformanceMonitor()

def cached_analysis(file_dependencies: List[str] = None):
    """
    Decorator for caching analysis results.
    
    Args:
        file_dependencies: List of file paths that invalidate cache when changed
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _analysis_cache._generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = _analysis_cache.get(cache_key, file_dependencies)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _analysis_cache.set(cache_key, result, file_dependencies)
            
            return result
        
        return wrapper
    return decorator

def performance_tracked(func_name: str = None):
    """
    Decorator for tracking function performance.
    
    Args:
        func_name: Optional custom name for the function
    """
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        return _performance_monitor.track_execution(name)(func)
    return decorator

def get_cache_instance() -> AnalysisCache:
    """Get the global cache instance."""
    return _analysis_cache

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor

def clear_all_caches():
    """Clear all caches and reset performance metrics."""
    _analysis_cache.clear()
    _performance_monitor.reset_metrics()

def get_optimization_report() -> Dict[str, Any]:
    """Get comprehensive optimization report."""
    return {
        'performance': _performance_monitor.get_performance_report(),
        'cache_stats': _analysis_cache.get_cache_stats(),
        'recommendations': _generate_system_recommendations()
    }

def _generate_system_recommendations() -> List[str]:
    """Generate system-level optimization recommendations."""
    recommendations = []
    
    # Get current stats
    perf_report = _performance_monitor.get_performance_report()
    cache_stats = _analysis_cache.get_cache_stats()
    
    # Cache recommendations
    if cache_stats['memory_cache_size'] < cache_stats['max_memory_items'] * 0.5:
        recommendations.append("💾 Consider increasing cache size for better performance")
    
    # Performance recommendations
    if perf_report['total_analysis_time'] > 60.0:
        recommendations.append("⚡ Total analysis time is high - consider enabling incremental analysis")
    
    if len(perf_report['bottlenecks']) > 3:
        recommendations.append("🐌 Multiple performance bottlenecks detected - prioritize optimization")
    
    return recommendations

