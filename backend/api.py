#!/usr/bin/env python3
"""
Consolidated API Server for Codebase Analytics
Provides HTTP endpoints for analysis and visualization services
Graph-sitter compliant with comprehensive analysis features
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading
import time

# Import our consolidated analysis module
from analysis import (
    analyze_codebase, 
    get_function_context, 
    get_issue_context, 
    get_symbol_context,
    get_interactive_symbol_data,
    AnalysisResult
)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory cache for analysis results
analysis_cache = {}
cache_timestamps = {}
CACHE_EXPIRY_HOURS = 24

# Background analysis queue
analysis_queue = []
analysis_lock = threading.Lock()

def is_cache_valid(cache_key: str) -> bool:
    """Check if cached result is still valid"""
    if cache_key not in cache_timestamps:
        return False
    
    cache_time = cache_timestamps[cache_key]
    expiry_time = cache_time + timedelta(hours=CACHE_EXPIRY_HOURS)
    return datetime.now() < expiry_time

def clean_expired_cache():
    """Remove expired cache entries"""
    current_time = datetime.now()
    expired_keys = []
    
    for key, timestamp in cache_timestamps.items():
        if current_time > timestamp + timedelta(hours=CACHE_EXPIRY_HOURS):
            expired_keys.append(key)
    
    for key in expired_keys:
        analysis_cache.pop(key, None)
        cache_timestamps.pop(key, None)
    
    logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": len(analysis_cache),
        "queue_size": len(analysis_queue)
    })

@app.route('/analyze/<path:repo_path>', methods=['GET'])
def analyze_repository(repo_path: str):
    """
    MAIN ANALYSIS ENDPOINT - Complete comprehensive analysis
    Returns: ALL function contexts, issues, entry points, etc.
    
    Query parameters:
    - refresh: Force refresh of cached results
    - include_source: Include source code in function definitions
    """
    try:
        # Clean expired cache periodically
        if len(analysis_cache) > 100:
            clean_expired_cache()
        
        # Check cache first
        cache_key = f"analysis:{repo_path}"
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if not refresh and cache_key in analysis_cache and is_cache_valid(cache_key):
            logger.info(f"Returning cached analysis for {repo_path}")
            return jsonify(analysis_cache[cache_key])
        
        # Validate repository path
        if not os.path.exists(repo_path):
            return jsonify({"error": f"Repository path '{repo_path}' not found"}), 404
        
        if not os.path.isdir(repo_path):
            return jsonify({"error": f"Path '{repo_path}' is not a directory"}), 400
        
        logger.info(f"Starting analysis of {repo_path}")
        
        # Perform comprehensive analysis
        analysis_result = analyze_codebase(repo_path)
        
        # Convert to API response format
        response_data = {
            "status": "success",
            "analysis_timestamp": analysis_result.analysis_timestamp,
            "repository_info": {
                "path": analysis_result.repository_path,
                "total_files": analysis_result.total_files,
                "total_lines": analysis_result.total_lines,
                "programming_languages": analysis_result.programming_languages
            },
            "summary": {
                "total_functions": len(analysis_result.all_functions),
                "total_entry_points": len(analysis_result.all_entry_points),
                "total_issues": len(analysis_result.all_issues),
                "issues_by_severity": _group_issues_by_severity(analysis_result.all_issues)
            },
            "functions": [func.to_dict() for func in analysis_result.all_functions],
            "entry_points": [ep.to_dict() for ep in analysis_result.all_entry_points],
            "issues": [issue.to_dict() for issue in analysis_result.all_issues],
            "dependency_graph": analysis_result.dependency_graph,
            "symbol_table": analysis_result.symbol_table
        }
        
        # Optionally exclude source code to reduce response size
        include_source = request.args.get('include_source', 'true').lower() == 'true'
        if not include_source:
            for func in response_data["functions"]:
                func.pop('source_code', None)
        
        # Cache results
        analysis_cache[cache_key] = response_data
        cache_timestamps[cache_key] = datetime.now()
        
        logger.info(f"Analysis complete for {repo_path}: {len(analysis_result.all_functions)} functions, {len(analysis_result.all_entry_points)} entry points")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing {repo_path}: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/function/<path:repo_path>/<function_name>', methods=['GET'])
def get_function_details(repo_path: str, function_name: str):
    """
    Get detailed context for a specific function
    Returns: Function definition, dependencies, issues, related entry points
    """
    try:
        # Check if we have cached analysis
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            # Trigger analysis if not cached
            analysis_result = analyze_codebase(repo_path)
        else:
            # Reconstruct analysis result from cache
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        # Get function context
        function_context = get_function_context(analysis_result, function_name)
        
        if not function_context:
            return jsonify({"error": f"Function '{function_name}' not found"}), 404
        
        return jsonify({
            "status": "success",
            "function_name": function_name,
            "context": function_context
        })
        
    except Exception as e:
        logger.error(f"Error getting function context for {function_name}: {str(e)}")
        return jsonify({"error": f"Failed to get function context: {str(e)}"}), 500

@app.route('/issues/<path:repo_path>', methods=['GET'])
def get_repository_issues(repo_path: str):
    """
    Get all issues for a repository with optional filtering
    
    Query parameters:
    - severity: Filter by severity (critical, major, minor, info)
    - type: Filter by issue type
    - file: Filter by file path
    """
    try:
        # Check cache
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            analysis_result = analyze_codebase(repo_path)
        else:
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        # Apply filters
        issues = analysis_result.all_issues
        
        severity_filter = request.args.get('severity')
        if severity_filter:
            issues = [issue for issue in issues if issue.severity.value == severity_filter]
        
        type_filter = request.args.get('type')
        if type_filter:
            issues = [issue for issue in issues if issue.type.value == type_filter]
        
        file_filter = request.args.get('file')
        if file_filter:
            issues = [issue for issue in issues if file_filter in issue.file_path]
        
        # Group issues by file for better organization
        issues_by_file = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue.to_dict())
        
        return jsonify({
            "status": "success",
            "total_issues": len(issues),
            "issues_by_severity": _group_issues_by_severity(issues),
            "issues_by_file": issues_by_file,
            "filters_applied": {
                "severity": severity_filter,
                "type": type_filter,
                "file": file_filter
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting issues for {repo_path}: {str(e)}")
        return jsonify({"error": f"Failed to get issues: {str(e)}"}), 500

@app.route('/entry-points/<path:repo_path>', methods=['GET'])
def get_entry_points(repo_path: str):
    """
    Get ALL entry points for a repository
    Returns: Complete list of entry points with their contexts
    """
    try:
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            analysis_result = analyze_codebase(repo_path)
        else:
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        # Group entry points by type
        entry_points_by_type = {}
        for ep in analysis_result.all_entry_points:
            ep_type = ep.type
            if ep_type not in entry_points_by_type:
                entry_points_by_type[ep_type] = []
            entry_points_by_type[ep_type].append(ep.to_dict())
        
        return jsonify({
            "status": "success",
            "total_entry_points": len(analysis_result.all_entry_points),
            "entry_points_by_type": entry_points_by_type,
            "all_entry_points": [ep.to_dict() for ep in analysis_result.all_entry_points]
        })
        
    except Exception as e:
        logger.error(f"Error getting entry points for {repo_path}: {str(e)}")
        return jsonify({"error": f"Failed to get entry points: {str(e)}"}), 500

@app.route('/symbols/<path:repo_path>', methods=['GET'])
def get_interactive_symbols(repo_path: str):
    """
    Get interactive symbol data for symbol selection
    Returns: All symbols with metadata for interactive selection
    """
    try:
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            analysis_result = analyze_codebase(repo_path)
        else:
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        # Get interactive symbol data
        symbol_data = get_interactive_symbol_data(analysis_result)
        
        return jsonify({
            "status": "success",
            **symbol_data
        })
        
    except Exception as e:
        logger.error(f"Error getting symbols for {repo_path}: {str(e)}")
        return jsonify({"error": f"Failed to get symbols: {str(e)}"}), 500

@app.route('/symbol-context/<path:repo_path>/<symbol_name>', methods=['GET'])
def get_symbol_context_endpoint(repo_path: str, symbol_name: str):
    """
    Get comprehensive context for a specific symbol (interactive selection)
    Returns: Complete symbol context with relationships and issues
    """
    try:
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            analysis_result = analyze_codebase(repo_path)
        else:
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        # Get optional file path filter
        file_path = request.args.get('file_path')
        
        # Get symbol context
        context = get_symbol_context(analysis_result, symbol_name, file_path)
        
        if context:
            return jsonify({
                "status": "success",
                "symbol_context": context
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": f"Symbol '{symbol_name}' not found"
            }), 404
        
    except Exception as e:
        logger.error(f"Error getting symbol context for {symbol_name} in {repo_path}: {str(e)}")
        return jsonify({"error": f"Failed to get symbol context: {str(e)}"}), 500

@app.route('/visualize/<path:repo_path>', methods=['GET'])
def get_visualization_data(repo_path: str):
    """
    MAIN VISUALIZATION ENDPOINT - Interactive visualization data
    Returns: Repository tree, issue counts, symbol trees, interactive UI data
    """
    try:
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            analysis_result = analyze_codebase(repo_path)
        else:
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        # Generate visualization data
        visualization_data = {
            "repository_tree": _generate_repository_tree(analysis_result),
            "dependency_graph": _generate_dependency_graph_viz(analysis_result),
            "issue_heatmap": _generate_issue_heatmap(analysis_result),
            "function_complexity_chart": _generate_complexity_chart(analysis_result),
            "entry_points_map": _generate_entry_points_map(analysis_result),
            "symbol_relationships": _generate_symbol_relationships(analysis_result)
        }
        
        return jsonify({
            "status": "success",
            "visualization_data": visualization_data,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "repository_path": repo_path,
                "total_nodes": len(analysis_result.all_functions) + len(analysis_result.all_entry_points)
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating visualization for {repo_path}: {str(e)}")
        return jsonify({"error": f"Failed to generate visualization: {str(e)}"}), 500

@app.route('/search/<path:repo_path>', methods=['GET'])
def search_codebase(repo_path: str):
    """
    Search functions, entry points, and issues
    
    Query parameters:
    - q: Search query
    - type: Search type (functions, entry_points, issues, all)
    """
    try:
        query = request.args.get('q', '').lower()
        search_type = request.args.get('type', 'all')
        
        if not query:
            return jsonify({"error": "Search query 'q' parameter is required"}), 400
        
        cache_key = f"analysis:{repo_path}"
        
        if cache_key not in analysis_cache or not is_cache_valid(cache_key):
            analysis_result = analyze_codebase(repo_path)
        else:
            cached_data = analysis_cache[cache_key]
            analysis_result = _reconstruct_analysis_result(cached_data)
        
        results = {
            "functions": [],
            "entry_points": [],
            "issues": []
        }
        
        # Search functions
        if search_type in ['functions', 'all']:
            for func in analysis_result.all_functions:
                if (query in func.name.lower() or 
                    (func.docstring and query in func.docstring.lower()) or
                    query in func.source_code.lower()):
                    results["functions"].append(func.to_dict())
        
        # Search entry points
        if search_type in ['entry_points', 'all']:
            for ep in analysis_result.all_entry_points:
                if (query in ep.name.lower() or 
                    query in ep.description.lower()):
                    results["entry_points"].append(ep.to_dict())
        
        # Search issues
        if search_type in ['issues', 'all']:
            for issue in analysis_result.all_issues:
                if (query in issue.message.lower() or 
                    query in issue.file_path.lower()):
                    results["issues"].append(issue.to_dict())
        
        total_results = len(results["functions"]) + len(results["entry_points"]) + len(results["issues"])
        
        return jsonify({
            "status": "success",
            "query": query,
            "search_type": search_type,
            "total_results": total_results,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error searching {repo_path}: {str(e)}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    return jsonify({
        "total_entries": len(analysis_cache),
        "cache_keys": list(analysis_cache.keys()),
        "timestamps": {k: v.isoformat() for k, v in cache_timestamps.items()},
        "queue_size": len(analysis_queue)
    })

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear analysis cache"""
    global analysis_cache, cache_timestamps
    
    cleared_count = len(analysis_cache)
    analysis_cache.clear()
    cache_timestamps.clear()
    
    return jsonify({
        "status": "success",
        "message": f"Cleared {cleared_count} cache entries"
    })

# Helper functions
def _group_issues_by_severity(issues: List) -> Dict[str, int]:
    """Group issues by severity level"""
    severity_counts = {"critical": 0, "major": 0, "minor": 0, "info": 0}
    
    for issue in issues:
        severity = issue.severity.value if hasattr(issue, 'severity') else issue.get('severity', 'info')
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    return severity_counts

def _reconstruct_analysis_result(cached_data: Dict[str, Any]) -> AnalysisResult:
    """Reconstruct AnalysisResult from cached data (simplified)"""
    # This is a simplified reconstruction for demo purposes
    # In a full implementation, you'd properly deserialize all objects
    from analysis import AnalysisResult, FunctionDefinition, EntryPoint, CodeIssue
    
    # For now, return a minimal result that works with the API
    class MockAnalysisResult:
        def __init__(self, data):
            self.all_functions = []
            self.all_entry_points = []
            self.all_issues = []
            self.dependency_graph = data.get('dependency_graph', {})
            self.symbol_table = data.get('symbol_table', {})
    
    return MockAnalysisResult(cached_data)

def _generate_repository_tree(analysis_result) -> Dict[str, Any]:
    """Generate interactive repository tree structure"""
    # Build file tree from analysis results
    file_tree = {
        "name": "Repository",
        "type": "directory",
        "path": "/",
        "expanded": True,
        "children": []
    }
    
    # This would be implemented based on the file paths in the analysis
    # For now, return a basic structure
    return file_tree

def _generate_dependency_graph_viz(analysis_result) -> Dict[str, Any]:
    """Generate dependency graph visualization data"""
    nodes = []
    edges = []
    
    # Convert dependency graph to visualization format
    for source, targets in analysis_result.dependency_graph.items():
        nodes.append({"id": source, "label": source, "type": "function"})
        for target in targets:
            edges.append({"source": source, "target": target})
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def _generate_issue_heatmap(analysis_result) -> Dict[str, Any]:
    """Generate issue heatmap data"""
    file_issue_counts = {}
    
    for issue in analysis_result.all_issues:
        file_path = issue.file_path if hasattr(issue, 'file_path') else issue.get('file_path', 'unknown')
        if file_path not in file_issue_counts:
            file_issue_counts[file_path] = 0
        file_issue_counts[file_path] += 1
    
    return {
        "file_issue_counts": file_issue_counts,
        "max_issues": max(file_issue_counts.values()) if file_issue_counts else 0
    }

def _generate_complexity_chart(analysis_result) -> Dict[str, Any]:
    """Generate function complexity chart data"""
    complexity_data = []
    
    for func in analysis_result.all_functions:
        complexity_data.append({
            "name": func.name if hasattr(func, 'name') else func.get('name', 'unknown'),
            "complexity": func.complexity_score if hasattr(func, 'complexity_score') else func.get('complexity_score', 1),
            "lines": (func.line_end - func.line_start) if hasattr(func, 'line_end') else 10
        })
    
    return {
        "functions": complexity_data,
        "average_complexity": sum(f["complexity"] for f in complexity_data) / len(complexity_data) if complexity_data else 0
    }

def _generate_entry_points_map(analysis_result) -> Dict[str, Any]:
    """Generate entry points map"""
    entry_points_by_type = {}
    
    for ep in analysis_result.all_entry_points:
        ep_type = ep.type if hasattr(ep, 'type') else ep.get('type', 'unknown')
        if ep_type not in entry_points_by_type:
            entry_points_by_type[ep_type] = []
        
        entry_points_by_type[ep_type].append({
            "name": ep.name if hasattr(ep, 'name') else ep.get('name', 'unknown'),
            "file": ep.file_path if hasattr(ep, 'file_path') else ep.get('file_path', 'unknown'),
            "line": ep.line_number if hasattr(ep, 'line_number') else ep.get('line_number', 1)
        })
    
    return entry_points_by_type

def _generate_symbol_relationships(analysis_result) -> Dict[str, Any]:
    """Generate symbol relationship data"""
    return {
        "symbols": list(analysis_result.symbol_table.keys()),
        "relationships": analysis_result.dependency_graph
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Codebase Analytics API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Codebase Analytics API server on {args.host}:{args.port}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )
