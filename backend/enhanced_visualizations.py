#!/usr/bin/env python3
"""
Enhanced Visualization Module

This module provides advanced visualization capabilities for comprehensive
codebase analysis, including dependency graphs, call flow diagrams,
architectural views, and quality heatmaps.
"""

import json
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import asdict

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase

# Import analysis modules
from advanced_analysis import (
    DependencyAnalysis,
    CallGraphAnalysis,
    CodeQualityMetrics,
    ArchitecturalInsights,
    perform_comprehensive_analysis,
    AnalysisType
)

def create_enhanced_dependency_graph(codebase: Codebase, analysis: DependencyAnalysis = None) -> Dict[str, Any]:
    """Create an enhanced dependency graph visualization."""
    if analysis is None:
        results = perform_comprehensive_analysis(codebase, [AnalysisType.DEPENDENCY])
        analysis = results.get('dependency_analysis')
    
    if not analysis:
        return {"nodes": [], "edges": [], "metadata": {}}
    
    nodes = []
    edges = []
    node_id_map = {}
    
    # Create nodes for each file/module
    for i, (filepath, deps) in enumerate(analysis.dependency_graph.items()):
        node_id = f"node_{i}"
        node_id_map[filepath] = node_id
        
        # Determine node type and color
        is_external = filepath in analysis.external_dependencies
        is_critical = filepath in analysis.critical_dependencies
        
        node_color = "#ff6b6b" if is_critical else "#4ecdc4" if is_external else "#45b7d1"
        node_size = 20 + (len(deps) * 2)  # Size based on number of dependencies
        
        nodes.append({
            "id": node_id,
            "label": filepath.split('/')[-1],  # Just filename
            "title": filepath,
            "color": node_color,
            "size": node_size,
            "group": "external" if is_external else "internal",
            "dependencies_count": len(deps),
            "is_critical": is_critical
        })
    
    # Create edges for dependencies
    edge_id = 0
    for filepath, deps in analysis.dependency_graph.items():
        source_id = node_id_map.get(filepath)
        if not source_id:
            continue
            
        for dep in deps:
            target_id = node_id_map.get(dep)
            if target_id:
                edges.append({
                    "id": f"edge_{edge_id}",
                    "from": source_id,
                    "to": target_id,
                    "arrows": "to",
                    "color": "#ff6b6b" if dep in analysis.critical_dependencies else "#999999"
                })
                edge_id += 1
    
    # Add circular dependency highlighting
    circular_edges = []
    for cycle in analysis.circular_dependencies:
        for i in range(len(cycle) - 1):
            source = node_id_map.get(cycle[i])
            target = node_id_map.get(cycle[i + 1])
            if source and target:
                circular_edges.append({
                    "from": source,
                    "to": target,
                    "color": "#ff0000",
                    "width": 3,
                    "dashes": True
                })
    
    return {
        "nodes": nodes,
        "edges": edges + circular_edges,
        "metadata": {
            "total_dependencies": analysis.total_dependencies,
            "circular_dependencies_count": len(analysis.circular_dependencies),
            "dependency_depth": analysis.dependency_depth,
            "critical_dependencies": analysis.critical_dependencies,
            "layout": "hierarchical"
        }
    }

def create_call_flow_diagram(codebase: Codebase, analysis: CallGraphAnalysis = None) -> Dict[str, Any]:
    """Create a call flow diagram visualization."""
    if analysis is None:
        results = perform_comprehensive_analysis(codebase, [AnalysisType.CALL_GRAPH])
        analysis = results.get('call_graph_analysis')
    
    if not analysis:
        return {"nodes": [], "edges": [], "metadata": {}}
    
    nodes = []
    edges = []
    
    # Create nodes for functions
    for i, (func_name, calls) in enumerate(analysis.call_graph.items()):
        is_entry_point = func_name in analysis.entry_points
        is_leaf = func_name in analysis.leaf_functions
        
        # Determine node characteristics
        node_color = "#ff6b6b" if is_entry_point else "#4ecdc4" if is_leaf else "#45b7d1"
        node_shape = "diamond" if is_entry_point else "square" if is_leaf else "circle"
        
        # Size based on connectivity
        connectivity = next((count for name, count in analysis.most_connected_functions if name == func_name), 0)
        node_size = 15 + min(connectivity * 2, 30)
        
        nodes.append({
            "id": func_name,
            "label": func_name,
            "color": node_color,
            "shape": node_shape,
            "size": node_size,
            "title": f"Function: {func_name}\\nCalls: {len(calls)}\\nConnectivity: {connectivity}",
            "group": "entry_point" if is_entry_point else "leaf" if is_leaf else "internal",
            "calls_count": len(calls),
            "connectivity": connectivity
        })
    
    # Create edges for function calls
    edge_id = 0
    for func_name, calls in analysis.call_graph.items():
        for called_func in calls:
            if called_func in analysis.call_graph:  # Only include functions we know about
                edges.append({
                    "id": f"call_{edge_id}",
                    "from": func_name,
                    "to": called_func,
                    "arrows": "to",
                    "color": "#666666"
                })
                edge_id += 1
    
    # Highlight call chains
    chain_edges = []
    for i, chain in enumerate(analysis.call_chains[:3]):  # Top 3 chains
        chain_color = ["#ff6b6b", "#4ecdc4", "#45b7d1"][i]
        for j in range(len(chain) - 1):
            chain_edges.append({
                "from": chain[j],
                "to": chain[j + 1],
                "color": chain_color,
                "width": 3,
                "title": f"Call chain {i + 1}"
            })
    
    return {
        "nodes": nodes,
        "edges": edges + chain_edges,
        "metadata": {
            "total_functions": len(analysis.call_graph),
            "total_call_relationships": analysis.total_call_relationships,
            "call_depth": analysis.call_depth,
            "entry_points": analysis.entry_points,
            "leaf_functions": analysis.leaf_functions,
            "call_chains_count": len(analysis.call_chains),
            "layout": "directed"
        }
    }

def create_quality_heatmap(codebase: Codebase, analysis: CodeQualityMetrics = None) -> Dict[str, Any]:
    """Create a code quality heatmap visualization."""
    if analysis is None:
        results = perform_comprehensive_analysis(codebase, [AnalysisType.CODE_QUALITY])
        analysis = results.get('code_quality_metrics')
    
    if not analysis:
        return {"data": [], "metadata": {}}
    
    # Create heatmap data for files
    heatmap_data = []
    
    for file in codebase.files:
        if not file.source:
            continue
            
        # Calculate quality metrics for this file
        lines = file.source.split('\n')
        loc = len([line for line in lines if line.strip()])
        
        # Simple quality score calculation
        quality_score = 100.0
        
        # Deduct for long files
        if loc > 500:
            quality_score -= min(30, (loc - 500) / 50)
        
        # Deduct for TODO/FIXME
        todo_count = file.source.lower().count('todo') + file.source.lower().count('fixme')
        quality_score -= min(20, todo_count * 5)
        
        # Deduct for lack of comments
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        comment_ratio = comment_lines / loc if loc > 0 else 0
        if comment_ratio < 0.1:
            quality_score -= 15
        
        quality_score = max(0, quality_score)
        
        heatmap_data.append({
            "file": file.filepath,
            "quality_score": quality_score,
            "lines_of_code": loc,
            "comment_ratio": comment_ratio * 100,
            "todo_count": todo_count,
            "functions_count": len(file.functions),
            "classes_count": len(file.classes)
        })
    
    # Sort by quality score
    heatmap_data.sort(key=lambda x: x['quality_score'])
    
    return {
        "data": heatmap_data,
        "metadata": {
            "total_files": len(heatmap_data),
            "average_quality": sum(item['quality_score'] for item in heatmap_data) / len(heatmap_data) if heatmap_data else 0,
            "documentation_coverage": analysis.documentation_coverage,
            "code_duplication_percentage": analysis.code_duplication_percentage,
            "technical_debt_ratio": analysis.technical_debt_ratio,
            "visualization_type": "heatmap"
        }
    }

def create_architectural_overview(codebase: Codebase, analysis: ArchitecturalInsights = None) -> Dict[str, Any]:
    """Create an architectural overview visualization."""
    if analysis is None:
        results = perform_comprehensive_analysis(codebase, [AnalysisType.ARCHITECTURAL])
        analysis = results.get('architectural_insights')
    
    if not analysis:
        return {"components": [], "relationships": [], "metadata": {}}
    
    components = []
    relationships = []
    
    # Create components from the component analysis
    for component_name, component_data in analysis.component_analysis.items():
        component_size = component_data.get('lines_of_code', 0)
        functions_count = component_data.get('functions', 0)
        classes_count = component_data.get('classes', 0)
        
        # Determine component type and color
        if 'test' in component_name.lower():
            component_type = "test"
            color = "#4ecdc4"
        elif 'util' in component_name.lower() or 'helper' in component_name.lower():
            component_type = "utility"
            color = "#45b7d1"
        elif 'model' in component_name.lower() or 'entity' in component_name.lower():
            component_type = "model"
            color = "#96ceb4"
        elif 'controller' in component_name.lower() or 'handler' in component_name.lower():
            component_type = "controller"
            color = "#feca57"
        elif 'service' in component_name.lower():
            component_type = "service"
            color = "#ff9ff3"
        else:
            component_type = "general"
            color = "#dda0dd"
        
        components.append({
            "id": component_name,
            "name": component_name,
            "type": component_type,
            "color": color,
            "size": max(20, min(100, component_size / 10)),
            "functions_count": functions_count,
            "classes_count": classes_count,
            "lines_of_code": component_size,
            "files": component_data.get('files', [])
        })
    
    # Create relationships based on coupling metrics
    for component1 in components:
        for component2 in components:
            if component1['id'] != component2['id']:
                # Simple heuristic for relationships
                shared_patterns = 0
                if component1['type'] == component2['type']:
                    shared_patterns += 1
                
                # Check for common naming patterns
                name1_parts = set(component1['name'].lower().split('/'))
                name2_parts = set(component2['name'].lower().split('/'))
                if name1_parts.intersection(name2_parts):
                    shared_patterns += 1
                
                if shared_patterns > 0:
                    relationships.append({
                        "from": component1['id'],
                        "to": component2['id'],
                        "strength": shared_patterns,
                        "type": "coupling"
                    })
    
    return {
        "components": components,
        "relationships": relationships,
        "metadata": {
            "architectural_patterns": analysis.architectural_patterns,
            "modularity_score": analysis.modularity_score,
            "total_components": len(components),
            "coupling_relationships": len(relationships),
            "visualization_type": "architectural_overview"
        }
    }

def create_security_risk_map(codebase: Codebase) -> Dict[str, Any]:
    """Create a security risk map visualization."""
    from advanced_analysis import analyze_security
    
    security_analysis = analyze_security(codebase)
    
    risk_map = []
    
    # Group vulnerabilities by file
    file_risks = defaultdict(list)
    for vuln in security_analysis.potential_vulnerabilities:
        file_risks[vuln['file']].append(vuln)
    
    for file_path, risks in file_risks.items():
        risk_level = "low"
        risk_score = 0
        
        for risk in risks:
            if risk['type'] in ['sql_injection_risk', 'code_injection_risk']:
                risk_score += 10
                risk_level = "high"
            elif risk['type'] == 'hardcoded_password':
                risk_score += 5
                if risk_level != "high":
                    risk_level = "medium"
        
        if risk_score == 0:
            risk_level = "low"
            risk_score = 1
        
        risk_map.append({
            "file": file_path,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "vulnerabilities": risks,
            "vulnerability_count": len(risks)
        })
    
    # Add files with no identified risks
    for file in codebase.files:
        if file.filepath not in file_risks:
            risk_map.append({
                "file": file.filepath,
                "risk_level": "low",
                "risk_score": 1,
                "vulnerabilities": [],
                "vulnerability_count": 0
            })
    
    return {
        "risk_map": risk_map,
        "metadata": {
            "total_files": len(risk_map),
            "high_risk_files": len([r for r in risk_map if r['risk_level'] == 'high']),
            "medium_risk_files": len([r for r in risk_map if r['risk_level'] == 'medium']),
            "low_risk_files": len([r for r in risk_map if r['risk_level'] == 'low']),
            "total_vulnerabilities": sum(r['vulnerability_count'] for r in risk_map),
            "visualization_type": "security_risk_map"
        }
    }

def create_performance_hotspot_map(codebase: Codebase) -> Dict[str, Any]:
    """Create a performance hotspot map visualization."""
    from advanced_analysis import analyze_performance
    
    performance_analysis = analyze_performance(codebase)
    
    hotspot_map = []
    
    # Group hotspots by function
    function_hotspots = defaultdict(list)
    for hotspot in performance_analysis.performance_hotspots:
        function_hotspots[hotspot['function']].append(hotspot)
    
    for func in codebase.functions:
        hotspots = function_hotspots.get(func.name, [])
        
        performance_score = 100
        hotspot_types = []
        
        for hotspot in hotspots:
            hotspot_types.append(hotspot['type'])
            if hotspot['type'] == 'nested_loops':
                performance_score -= 30
            elif hotspot['type'] == 'n_plus_one_query':
                performance_score -= 25
        
        performance_score = max(0, performance_score)
        
        # Calculate complexity-based performance impact
        if hasattr(func, 'code_block') and func.code_block:
            lines = len(func.code_block.source.split('\n'))
            if lines > 100:
                performance_score -= min(20, (lines - 100) / 10)
        
        hotspot_map.append({
            "function": func.name,
            "performance_score": performance_score,
            "hotspot_count": len(hotspots),
            "hotspot_types": hotspot_types,
            "file": getattr(func, 'filepath', 'unknown'),
            "severity": "high" if performance_score < 50 else "medium" if performance_score < 75 else "low"
        })
    
    return {
        "hotspot_map": hotspot_map,
        "metadata": {
            "total_functions": len(hotspot_map),
            "high_impact_functions": len([h for h in hotspot_map if h['severity'] == 'high']),
            "medium_impact_functions": len([h for h in hotspot_map if h['severity'] == 'medium']),
            "low_impact_functions": len([h for h in hotspot_map if h['severity'] == 'low']),
            "total_hotspots": sum(h['hotspot_count'] for h in hotspot_map),
            "visualization_type": "performance_hotspot_map"
        }
    }

def create_comprehensive_dashboard_data(codebase: Codebase) -> Dict[str, Any]:
    """Create comprehensive dashboard data with all visualizations."""
    # Perform all analyses
    comprehensive_results = perform_comprehensive_analysis(codebase)
    
    dashboard_data = {
        "dependency_graph": create_enhanced_dependency_graph(
            codebase, comprehensive_results.get('dependency_analysis')
        ),
        "call_flow_diagram": create_call_flow_diagram(
            codebase, comprehensive_results.get('call_graph_analysis')
        ),
        "quality_heatmap": create_quality_heatmap(
            codebase, comprehensive_results.get('code_quality_metrics')
        ),
        "architectural_overview": create_architectural_overview(
            codebase, comprehensive_results.get('architectural_insights')
        ),
        "security_risk_map": create_security_risk_map(codebase),
        "performance_hotspot_map": create_performance_hotspot_map(codebase),
        "metadata": {
            "generated_at": "2024-01-01T00:00:00Z",
            "total_files": len(codebase.files),
            "total_functions": len(codebase.functions),
            "total_classes": len(codebase.classes),
            "analysis_types": list(AnalysisType),
            "dashboard_version": "2.0.0"
        }
    }
    
    return dashboard_data

# Utility functions for visualization

def calculate_node_positions(nodes: List[Dict], layout_type: str = "force") -> List[Dict]:
    """Calculate positions for nodes based on layout type."""
    if layout_type == "hierarchical":
        # Simple hierarchical layout
        levels = {}
        for node in nodes:
            level = node.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        y_spacing = 100
        for level, level_nodes in levels.items():
            x_spacing = 800 / max(1, len(level_nodes) - 1) if len(level_nodes) > 1 else 0
            for i, node in enumerate(level_nodes):
                node['x'] = i * x_spacing
                node['y'] = level * y_spacing
    
    elif layout_type == "circular":
        # Circular layout
        import math
        center_x, center_y = 400, 300
        radius = 200
        
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / len(nodes)
            node['x'] = center_x + radius * math.cos(angle)
            node['y'] = center_y + radius * math.sin(angle)
    
    return nodes

def generate_color_palette(count: int) -> List[str]:
    """Generate a color palette for visualizations."""
    base_colors = [
        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
        "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43"
    ]
    
    if count <= len(base_colors):
        return base_colors[:count]
    
    # Generate additional colors
    colors = base_colors.copy()
    for i in range(count - len(base_colors)):
        # Generate colors based on HSL
        hue = (i * 137.508) % 360  # Golden angle approximation
        colors.append(f"hsl({hue}, 70%, 60%)")
    
    return colors

