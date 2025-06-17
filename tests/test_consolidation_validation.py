#!/usr/bin/env python3
"""
Consolidation Validation Tests

This module validates that the consolidated backend modules work correctly
and that all functions are properly accessible.
"""

import unittest
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

class TestConsolidationValidation(unittest.TestCase):
    """Test that consolidation was successful."""
    
    def test_analysis_module_imports(self):
        """Test that analysis module imports work correctly."""
        try:
            from analysis import (
                AnalysisType,
                DependencyAnalysis,
                CallGraphAnalysis,
                CodeQualityMetrics,
                perform_comprehensive_analysis,
                analyze_dependencies_comprehensive,
                calculate_cyclomatic_complexity
            )
            self.assertTrue(True, "Analysis module imports successful")
        except ImportError as e:
            self.fail(f"Analysis module import failed: {e}")
    
    def test_visualization_module_imports(self):
        """Test that visualization module imports work correctly."""
        try:
            from visualization import (
                VisualizationType,
                VisualizationConfig,
                create_call_graph,
                create_dependency_graph,
                create_comprehensive_dashboard_data,
                generate_all_visualizations
            )
            self.assertTrue(True, "Visualization module imports successful")
        except ImportError as e:
            self.fail(f"Visualization module import failed: {e}")
    
    def test_api_module_imports(self):
        """Test that API module imports work correctly."""
        try:
            from api import (
                app,
                AnalysisRequest,
                VisualizationRequest,
                ComprehensiveInsights,
                get_github_repo_description,
                analyze_functions_comprehensive
            )
            self.assertTrue(True, "API module imports successful")
        except ImportError as e:
            self.fail(f"API module import failed: {e}")
    
    def test_analysis_types_enum(self):
        """Test that AnalysisType enum is properly defined."""
        from analysis import AnalysisType
        
        expected_types = [
            'dependency', 'call_graph', 'code_quality', 
            'architectural', 'security', 'performance'
        ]
        
        actual_types = [at.value for at in AnalysisType]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, actual_types, 
                         f"Missing analysis type: {expected_type}")
    
    def test_visualization_types_enum(self):
        """Test that VisualizationType enum is properly defined."""
        from visualization import VisualizationType
        
        expected_types = [
            'call_graph', 'dependency_graph', 'class_hierarchy',
            'complexity_heatmap', 'enhanced_dependency_graph',
            'comprehensive_dashboard'
        ]
        
        actual_types = [vt.value for vt in VisualizationType]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, actual_types,
                         f"Missing visualization type: {expected_type}")
    
    def test_function_availability(self):
        """Test that key functions are available and callable."""
        from analysis import (
            calculate_cyclomatic_complexity,
            analyze_dependencies_comprehensive,
            perform_comprehensive_analysis
        )
        from visualization import (
            create_call_graph,
            create_dependency_graph,
            generate_all_visualizations
        )
        
        # Test that functions are callable
        self.assertTrue(callable(calculate_cyclomatic_complexity))
        self.assertTrue(callable(analyze_dependencies_comprehensive))
        self.assertTrue(callable(perform_comprehensive_analysis))
        self.assertTrue(callable(create_call_graph))
        self.assertTrue(callable(create_dependency_graph))
        self.assertTrue(callable(generate_all_visualizations))
    
    def test_data_classes_available(self):
        """Test that data classes are properly defined."""
        from analysis import (
            DependencyAnalysis,
            CallGraphAnalysis,
            CodeQualityMetrics,
            ArchitecturalInsights,
            SecurityAnalysis,
            PerformanceAnalysis
        )
        
        # Test that data classes can be instantiated
        dep_analysis = DependencyAnalysis()
        self.assertIsInstance(dep_analysis, DependencyAnalysis)
        
        call_analysis = CallGraphAnalysis()
        self.assertIsInstance(call_analysis, CallGraphAnalysis)
        
        quality_metrics = CodeQualityMetrics()
        self.assertIsInstance(quality_metrics, CodeQualityMetrics)

if __name__ == '__main__':
    unittest.main()
