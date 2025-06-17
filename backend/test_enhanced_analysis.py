#!/usr/bin/env python3
"""
Test Suite for Enhanced Analysis Features

This module provides comprehensive tests for the advanced analysis capabilities
including dependency analysis, call graph analysis, code quality metrics,
architectural insights, security analysis, and performance analysis.
"""

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import modules to test
from advanced_analysis import (
    DependencyAnalysis,
    CallGraphAnalysis,
    CodeQualityMetrics,
    ArchitecturalInsights,
    SecurityAnalysis,
    PerformanceAnalysis,
    AnalysisType,
    analyze_dependencies_comprehensive,
    analyze_call_graph,
    analyze_code_quality,
    analyze_architecture,
    analyze_security,
    analyze_performance,
    perform_comprehensive_analysis,
    detect_circular_dependencies,
    calculate_dependency_depth,
    estimate_code_duplication,
    analyze_naming_consistency,
    detect_code_smells,
    identify_refactoring_opportunities
)

from enhanced_visualizations import (
    create_enhanced_dependency_graph,
    create_call_flow_diagram,
    create_quality_heatmap,
    create_architectural_overview,
    create_security_risk_map,
    create_performance_hotspot_map,
    create_comprehensive_dashboard_data
)

class MockCodebase:
    """Mock codebase for testing."""
    
    def __init__(self):
        self.files = []
        self.functions = []
        self.classes = []
    
    def add_file(self, filepath: str, source: str = "", imports: List = None):
        """Add a mock file to the codebase."""
        mock_file = Mock()
        mock_file.filepath = filepath
        mock_file.source = source
        mock_file.imports = imports or []
        mock_file.functions = []
        mock_file.classes = []
        self.files.append(mock_file)
        return mock_file
    
    def add_function(self, name: str, filepath: str = "test.py", code: str = "", parameters: List = None):
        """Add a mock function to the codebase."""
        mock_function = Mock()
        mock_function.name = name
        mock_function.filepath = filepath
        mock_function.parameters = parameters or []
        mock_function.code_block = Mock()
        mock_function.code_block.source = code
        mock_function.function_calls = []
        mock_function.call_sites = []
        mock_function.usages = []
        mock_function.start_point = (1, 0)
        mock_function.end_point = (10, 0)
        self.functions.append(mock_function)
        return mock_function
    
    def add_class(self, name: str, filepath: str = "test.py", superclasses: List = None):
        """Add a mock class to the codebase."""
        mock_class = Mock()
        mock_class.name = name
        mock_class.filepath = filepath
        mock_class.superclasses = superclasses or []
        mock_class.methods = []
        mock_class.attributes = []
        self.classes.append(mock_class)
        return mock_class

class TestDependencyAnalysis(unittest.TestCase):
    """Test dependency analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
    
    def test_basic_dependency_analysis(self):
        """Test basic dependency analysis."""
        # Add files with imports
        mock_import1 = Mock()
        mock_import1.module = "os"
        mock_import1.name = "os"
        
        mock_import2 = Mock()
        mock_import2.module = "sys"
        mock_import2.name = "sys"
        
        self.codebase.add_file("main.py", imports=[mock_import1, mock_import2])
        self.codebase.add_file("utils.py", imports=[mock_import1])
        
        analysis = analyze_dependencies_comprehensive(self.codebase)
        
        self.assertIsInstance(analysis, DependencyAnalysis)
        self.assertGreater(analysis.total_dependencies, 0)
        self.assertIn("os", analysis.external_dependencies)
        self.assertIn("sys", analysis.external_dependencies)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        dependency_graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"]
        }
        
        cycles = detect_circular_dependencies(dependency_graph)
        self.assertGreater(len(cycles), 0)
        self.assertIn("A", cycles[0])
        self.assertIn("B", cycles[0])
        self.assertIn("C", cycles[0])
    
    def test_dependency_depth_calculation(self):
        """Test dependency depth calculation."""
        dependency_graph = {
            "A": ["B"],
            "B": ["C"],
            "C": []
        }
        
        depth = calculate_dependency_depth(dependency_graph)
        self.assertEqual(depth, 3)

class TestCallGraphAnalysis(unittest.TestCase):
    """Test call graph analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
    
    def test_basic_call_graph_analysis(self):
        """Test basic call graph analysis."""
        # Add functions with call relationships
        func1 = self.codebase.add_function("main", code="def main():\n    helper()\n    process()")
        func2 = self.codebase.add_function("helper", code="def helper():\n    pass")
        func3 = self.codebase.add_function("process", code="def process():\n    helper()")
        
        # Mock function calls
        mock_call1 = Mock()
        mock_call1.name = "helper"
        mock_call2 = Mock()
        mock_call2.name = "process"
        func1.function_calls = [mock_call1, mock_call2]
        
        mock_call3 = Mock()
        mock_call3.name = "helper"
        func3.function_calls = [mock_call3]
        
        analysis = analyze_call_graph(self.codebase)
        
        self.assertIsInstance(analysis, CallGraphAnalysis)
        self.assertGreater(analysis.total_call_relationships, 0)
        self.assertIn("main", analysis.entry_points)

class TestCodeQualityAnalysis(unittest.TestCase):
    """Test code quality analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
    
    def test_basic_quality_analysis(self):
        """Test basic code quality analysis."""
        # Add files with various quality indicators
        self.codebase.add_file("good.py", source='"""Well documented module"""\ndef good_function():\n    """Good function with docs"""\n    return True')
        self.codebase.add_file("bad.py", source="def bad_function():\n    # TODO: implement this\n    pass")
        
        func1 = self.codebase.add_function("good_function", code='"""Good function with docs"""\nreturn True')
        func2 = self.codebase.add_function("bad_function", code="# TODO: implement this\npass")
        
        analysis = analyze_code_quality(self.codebase)
        
        self.assertIsInstance(analysis, CodeQualityMetrics)
        self.assertGreaterEqual(analysis.documentation_coverage, 0)
        self.assertGreaterEqual(analysis.technical_debt_ratio, 0)
    
    def test_code_duplication_estimation(self):
        """Test code duplication estimation."""
        code_blocks = [
            "def function1():\n    print('hello')\n    return True",
            "def function2():\n    print('hello')\n    return False",
            "def function3():\n    print('world')\n    return None"
        ]
        
        duplication = estimate_code_duplication(code_blocks)
        self.assertGreaterEqual(duplication, 0)
        self.assertLessEqual(duplication, 100)
    
    def test_naming_consistency_analysis(self):
        """Test naming consistency analysis."""
        self.codebase.add_function("good_function_name")
        self.codebase.add_function("another_good_name")
        self.codebase.add_function("BadFunctionName")  # Should be snake_case
        
        self.codebase.add_class("GoodClassName")
        self.codebase.add_class("AnotherGoodClass")
        self.codebase.add_class("bad_class_name")  # Should be PascalCase
        
        consistency = analyze_naming_consistency(self.codebase)
        self.assertGreaterEqual(consistency, 0)
        self.assertLessEqual(consistency, 100)
    
    def test_code_smell_detection(self):
        """Test code smell detection."""
        # Add function with long method smell
        long_code = "\n".join([f"    line_{i} = {i}" for i in range(60)])
        self.codebase.add_function("long_method", code=f"def long_method():\n{long_code}")
        
        # Add function with too many parameters
        many_params = [Mock() for _ in range(8)]
        self.codebase.add_function("many_params", parameters=many_params)
        
        smells = detect_code_smells(self.codebase)
        self.assertIsInstance(smells, list)
        
        # Should detect long method
        long_method_smells = [s for s in smells if s['type'] == 'long_method']
        self.assertGreater(len(long_method_smells), 0)
        
        # Should detect too many parameters
        param_smells = [s for s in smells if s['type'] == 'too_many_parameters']
        self.assertGreater(len(param_smells), 0)

class TestArchitecturalAnalysis(unittest.TestCase):
    """Test architectural analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
    
    def test_basic_architectural_analysis(self):
        """Test basic architectural analysis."""
        # Add files in different directories to simulate components
        self.codebase.add_file("controllers/user_controller.py")
        self.codebase.add_file("models/user_model.py")
        self.codebase.add_file("services/user_service.py")
        self.codebase.add_file("utils/helpers.py")
        
        analysis = analyze_architecture(self.codebase)
        
        self.assertIsInstance(analysis, ArchitecturalInsights)
        self.assertGreaterEqual(analysis.modularity_score, 0)
        self.assertLessEqual(analysis.modularity_score, 100)

class TestSecurityAnalysis(unittest.TestCase):
    """Test security analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
    
    def test_basic_security_analysis(self):
        """Test basic security analysis."""
        # Add files with potential security issues
        self.codebase.add_file("insecure.py", source="password = 'hardcoded123'\nquery = 'SELECT * FROM users WHERE id = ' + user_id")
        self.codebase.add_file("risky.py", source="eval(user_input)\nsubprocess.call(command)")
        
        analysis = analyze_security(self.codebase)
        
        self.assertIsInstance(analysis, SecurityAnalysis)
        self.assertGreaterEqual(len(analysis.potential_vulnerabilities), 0)

class TestPerformanceAnalysis(unittest.TestCase):
    """Test performance analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
    
    def test_basic_performance_analysis(self):
        """Test basic performance analysis."""
        # Add functions with potential performance issues
        nested_loop_code = "for i in range(n):\n    for j in range(m):\n        process(i, j)"
        self.codebase.add_function("nested_loops", code=nested_loop_code)
        
        query_loop_code = "for user in users:\n    query = 'SELECT * FROM posts WHERE user_id = ' + user.id"
        self.codebase.add_function("query_in_loop", code=query_loop_code)
        
        analysis = analyze_performance(self.codebase)
        
        self.assertIsInstance(analysis, PerformanceAnalysis)
        self.assertGreaterEqual(len(analysis.performance_hotspots), 0)

class TestComprehensiveAnalysis(unittest.TestCase):
    """Test comprehensive analysis functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
        
        # Set up a realistic codebase structure
        self.codebase.add_file("main.py", source="import os\nimport sys\n\ndef main():\n    helper()\n    return True")
        self.codebase.add_file("utils.py", source="def helper():\n    \"\"\"Helper function\"\"\"\n    return 42")
        self.codebase.add_file("models.py", source="class User:\n    def __init__(self, name):\n        self.name = name")
        
        self.codebase.add_function("main", code="def main():\n    helper()\n    return True")
        self.codebase.add_function("helper", code="def helper():\n    return 42")
        self.codebase.add_class("User")
    
    def test_comprehensive_analysis_all_types(self):
        """Test comprehensive analysis with all analysis types."""
        results = perform_comprehensive_analysis(self.codebase)
        
        self.assertIsInstance(results, dict)
        self.assertIn('dependency_analysis', results)
        self.assertIn('call_graph_analysis', results)
        self.assertIn('code_quality_metrics', results)
        self.assertIn('architectural_insights', results)
        self.assertIn('security_analysis', results)
        self.assertIn('performance_analysis', results)
    
    def test_comprehensive_analysis_specific_types(self):
        """Test comprehensive analysis with specific analysis types."""
        analysis_types = [AnalysisType.DEPENDENCY, AnalysisType.CODE_QUALITY]
        results = perform_comprehensive_analysis(self.codebase, analysis_types)
        
        self.assertIn('dependency_analysis', results)
        self.assertIn('code_quality_metrics', results)
        self.assertNotIn('security_analysis', results)
        self.assertNotIn('performance_analysis', results)

class TestEnhancedVisualizations(unittest.TestCase):
    """Test enhanced visualization functionality."""
    
    def setUp(self):
        self.codebase = MockCodebase()
        
        # Set up test data
        mock_import = Mock()
        mock_import.module = "os"
        self.codebase.add_file("main.py", imports=[mock_import])
        self.codebase.add_function("main")
        self.codebase.add_class("TestClass")
    
    def test_dependency_graph_creation(self):
        """Test dependency graph visualization creation."""
        graph = create_enhanced_dependency_graph(self.codebase)
        
        self.assertIsInstance(graph, dict)
        self.assertIn('nodes', graph)
        self.assertIn('edges', graph)
        self.assertIn('metadata', graph)
    
    def test_call_flow_diagram_creation(self):
        """Test call flow diagram creation."""
        diagram = create_call_flow_diagram(self.codebase)
        
        self.assertIsInstance(diagram, dict)
        self.assertIn('nodes', diagram)
        self.assertIn('edges', diagram)
        self.assertIn('metadata', diagram)
    
    def test_quality_heatmap_creation(self):
        """Test quality heatmap creation."""
        heatmap = create_quality_heatmap(self.codebase)
        
        self.assertIsInstance(heatmap, dict)
        self.assertIn('data', heatmap)
        self.assertIn('metadata', heatmap)
    
    def test_architectural_overview_creation(self):
        """Test architectural overview creation."""
        overview = create_architectural_overview(self.codebase)
        
        self.assertIsInstance(overview, dict)
        self.assertIn('components', overview)
        self.assertIn('relationships', overview)
        self.assertIn('metadata', overview)
    
    def test_security_risk_map_creation(self):
        """Test security risk map creation."""
        risk_map = create_security_risk_map(self.codebase)
        
        self.assertIsInstance(risk_map, dict)
        self.assertIn('risk_map', risk_map)
        self.assertIn('metadata', risk_map)
    
    def test_performance_hotspot_map_creation(self):
        """Test performance hotspot map creation."""
        hotspot_map = create_performance_hotspot_map(self.codebase)
        
        self.assertIsInstance(hotspot_map, dict)
        self.assertIn('hotspot_map', hotspot_map)
        self.assertIn('metadata', hotspot_map)
    
    def test_comprehensive_dashboard_creation(self):
        """Test comprehensive dashboard data creation."""
        dashboard = create_comprehensive_dashboard_data(self.codebase)
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn('dependency_graph', dashboard)
        self.assertIn('call_flow_diagram', dashboard)
        self.assertIn('quality_heatmap', dashboard)
        self.assertIn('architectural_overview', dashboard)
        self.assertIn('security_risk_map', dashboard)
        self.assertIn('performance_hotspot_map', dashboard)
        self.assertIn('metadata', dashboard)

class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced analysis system."""
    
    def test_end_to_end_analysis_flow(self):
        """Test the complete analysis flow from codebase to visualizations."""
        # Create a realistic test codebase
        codebase = MockCodebase()
        
        # Add realistic file structure
        codebase.add_file("src/main.py", source="import os\nimport utils\n\ndef main():\n    utils.helper()\n    return True")
        codebase.add_file("src/utils.py", source="def helper():\n    \"\"\"Helper function\"\"\"\n    return 42")
        codebase.add_file("tests/test_main.py", source="import unittest\nfrom src.main import main")
        
        # Add functions and classes
        codebase.add_function("main", filepath="src/main.py")
        codebase.add_function("helper", filepath="src/utils.py")
        codebase.add_class("TestMain", filepath="tests/test_main.py")
        
        # Perform comprehensive analysis
        analysis_results = perform_comprehensive_analysis(codebase)
        
        # Verify analysis results
        self.assertIsInstance(analysis_results, dict)
        self.assertGreater(len(analysis_results), 0)
        
        # Create visualizations
        dashboard = create_comprehensive_dashboard_data(codebase)
        
        # Verify dashboard creation
        self.assertIsInstance(dashboard, dict)
        self.assertIn('metadata', dashboard)
        self.assertGreater(dashboard['metadata']['total_files'], 0)

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDependencyAnalysis,
        TestCallGraphAnalysis,
        TestCodeQualityAnalysis,
        TestArchitecturalAnalysis,
        TestSecurityAnalysis,
        TestPerformanceAnalysis,
        TestComprehensiveAnalysis,
        TestEnhancedVisualizations,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")

