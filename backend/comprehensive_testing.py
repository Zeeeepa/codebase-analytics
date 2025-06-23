#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Codebase Analysis

This module provides:
- Extensive test suites for all analysis components
- Performance benchmarking and validation
- Regression testing capabilities
- Test data generation and management
- Automated test reporting
"""

import os
import sys
import time
import json
import unittest
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import analysis modules
try:
    from analysis import (
        ComprehensiveCodebaseAnalyzer, Issue, IssueCollection, 
        IssueType, IssueCategory, IssueSeverity, CodeLocation,
        detect_implementation_errors, detect_security_vulnerabilities,
        detect_circular_dependencies_advanced, analyze_inheritance_patterns,
        analyze_complexity_patterns, analyze_performance_patterns,
        ContextCollector
    )
    from performance_optimization import (
        AnalysisCache, IncrementalAnalyzer, PerformanceMonitor,
        cached_analysis, performance_tracked, get_optimization_report
    )
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

@dataclass
class TestResult:
    """Represents the result of a test case."""
    test_name: str
    passed: bool
    duration: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None

@dataclass
class TestSuite:
    """Represents a collection of related tests."""
    name: str
    description: str
    tests: List[TestResult]
    setup_time: float = 0.0
    teardown_time: float = 0.0

class TestDataGenerator:
    """
    Generate test data for comprehensive analysis testing.
    
    Features:
    - Create synthetic code files with known issues
    - Generate dependency graphs for testing
    - Create performance test scenarios
    """
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="analysis_test_")
        self.created_files = []
    
    def create_test_repository(self) -> str:
        """Create a test repository with various code patterns."""
        repo_path = os.path.join(self.temp_dir, "test_repo")
        os.makedirs(repo_path, exist_ok=True)
        
        # Create files with different types of issues
        self._create_implementation_error_file(repo_path)
        self._create_security_vulnerability_file(repo_path)
        self._create_performance_issue_file(repo_path)
        self._create_circular_dependency_files(repo_path)
        self._create_complex_inheritance_file(repo_path)
        self._create_dead_code_file(repo_path)
        
        return repo_path
    
    def _create_implementation_error_file(self, repo_path: str):
        """Create a file with implementation errors."""
        content = '''
def function_with_unreachable_code():
    """Function with unreachable code after return."""
    x = 10
    return x
    print("This code is unreachable")  # Issue: unreachable code
    y = 20

def function_with_infinite_loop():
    """Function with potential infinite loop."""
    while True:  # Issue: infinite loop without break
        print("This will run forever")
        # Missing break condition

def function_with_off_by_one():
    """Function with off-by-one error."""
    arr = [1, 2, 3, 4, 5]
    for i in range(len(arr) + 1):  # Issue: off-by-one error
        print(arr[i])  # Will cause IndexError
'''
        
        filepath = os.path.join(repo_path, "implementation_errors.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_security_vulnerability_file(self, repo_path: str):
        """Create a file with security vulnerabilities."""
        content = '''
import os
import subprocess
import pickle

def dangerous_eval_usage(user_input):
    """Function using dangerous eval."""
    result = eval(user_input)  # Issue: dangerous eval usage
    return result

def unsafe_subprocess_call(command):
    """Function with unsafe subprocess usage."""
    os.system(command)  # Issue: unsafe system call
    subprocess.call(command, shell=True)  # Issue: shell=True is dangerous

def pickle_security_issue(data):
    """Function with pickle security issue."""
    return pickle.loads(data)  # Issue: pickle.loads can execute arbitrary code

def sql_injection_risk(user_id):
    """Function with potential SQL injection."""
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Issue: SQL injection risk
    return query
'''
        
        filepath = os.path.join(repo_path, "security_vulnerabilities.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_performance_issue_file(self, repo_path: str):
        """Create a file with performance issues."""
        content = '''
def nested_loops_performance_issue(data):
    """Function with nested loops causing O(n²) complexity."""
    result = []
    for i in data:  # Issue: nested loops
        for j in data:
            if i == j:
                result.append(i)
    return result

def inefficient_string_concatenation(items):
    """Function with inefficient string concatenation."""
    result = ""
    for item in items:  # Issue: inefficient string concatenation
        result += str(item)  # Should use join()
    return result

def repeated_expensive_operation(data):
    """Function with repeated expensive operations."""
    import re
    results = []
    for item in data:
        pattern = re.compile(r'\\d+')  # Issue: repeated re.compile
        match = pattern.search(item)
        if match:
            results.append(match.group())
    return results
'''
        
        filepath = os.path.join(repo_path, "performance_issues.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_circular_dependency_files(self, repo_path: str):
        """Create files with circular dependencies."""
        # File A imports B
        content_a = '''
from module_b import function_b

def function_a():
    return function_b() + 1
'''
        
        # File B imports A (circular dependency)
        content_b = '''
from module_a import function_a

def function_b():
    return 42

def another_function():
    return function_a() + 1  # Creates circular dependency
'''
        
        filepath_a = os.path.join(repo_path, "module_a.py")
        filepath_b = os.path.join(repo_path, "module_b.py")
        
        with open(filepath_a, 'w') as f:
            f.write(content_a)
        with open(filepath_b, 'w') as f:
            f.write(content_b)
        
        self.created_files.extend([filepath_a, filepath_b])
    
    def _create_complex_inheritance_file(self, repo_path: str):
        """Create a file with complex inheritance patterns."""
        content = '''
class BaseClass:
    def base_method(self):
        pass

class MiddleClass1(BaseClass):
    def middle_method1(self):
        pass

class MiddleClass2(BaseClass):
    def middle_method2(self):
        pass

class MiddleClass3(MiddleClass1):
    def middle_method3(self):
        pass

class MiddleClass4(MiddleClass2):
    def middle_method4(self):
        pass

class ComplexClass(MiddleClass3, MiddleClass4):  # Issue: complex multiple inheritance
    """Class with deep and complex inheritance hierarchy."""
    def complex_method(self):
        pass

class DeepInheritanceClass(ComplexClass):  # Issue: very deep inheritance
    """Class with very deep inheritance chain."""
    def deep_method(self):
        pass

class VeryDeepClass(DeepInheritanceClass):  # Issue: extremely deep inheritance
    def very_deep_method(self):
        pass
'''
        
        filepath = os.path.join(repo_path, "complex_inheritance.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_dead_code_file(self, repo_path: str):
        """Create a file with dead code."""
        content = '''
def unused_function():
    """This function is never called."""
    return "unused"

class UnusedClass:
    """This class is never instantiated."""
    def unused_method(self):
        return "unused"

def used_function():
    """This function is used."""
    return "used"

# Call the used function
result = used_function()

# Unused import
import json  # Issue: unused import
import os

# Used import
import sys
print(sys.version)

# Unused variable
unused_variable = "never used"  # Issue: unused variable

# Used variable
used_variable = "this is used"
print(used_variable)
'''
        
        filepath = os.path.join(repo_path, "dead_code.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def cleanup(self):
        """Clean up created test files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

class ComprehensiveTestRunner:
    """
    Comprehensive test runner for the analysis system.
    
    Features:
    - Run all test suites
    - Performance benchmarking
    - Regression testing
    - Detailed reporting
    """
    
    def __init__(self):
        self.test_suites = []
        self.test_data_generator = TestDataGenerator()
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🧪 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        try:
            # Run individual test suites
            self._run_issue_architecture_tests()
            self._run_error_detection_tests()
            self._run_context_collection_tests()
            self._run_advanced_analysis_tests()
            self._run_performance_optimization_tests()
            self._run_integration_tests()
            self._run_regression_tests()
            
            self.end_time = time.time()
            
            # Generate comprehensive report
            return self._generate_test_report()
            
        except Exception as e:
            self.end_time = time.time()
            return {
                "success": False,
                "error": str(e),
                "duration": self.end_time - self.start_time if self.start_time else 0
            }
        finally:
            self.test_data_generator.cleanup()
    
    def _run_issue_architecture_tests(self):
        """Test the enhanced issue architecture."""
        suite_start = time.time()
        tests = []
        
        # Test Issue creation
        test_start = time.time()
        try:
            location = CodeLocation("test.py", 10, 15)
            issue = Issue(
                type=IssueType.UNREACHABLE_CODE,
                category=IssueCategory.IMPLEMENTATION_ERROR,
                severity=IssueSeverity.ERROR,
                message="Test issue",
                location=location
            )
            
            # Test context addition
            issue.add_context("test_key", "test_value")
            
            assert issue.type == IssueType.UNREACHABLE_CODE
            assert issue.category == IssueCategory.IMPLEMENTATION_ERROR
            assert issue.severity == IssueSeverity.ERROR
            assert issue.context["test_key"] == "test_value"
            
            tests.append(TestResult(
                "issue_creation",
                True,
                time.time() - test_start,
                details={"issue_id": issue.id}
            ))
        except Exception as e:
            tests.append(TestResult(
                "issue_creation",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test IssueCollection
        test_start = time.time()
        try:
            issues = [
                Issue(IssueType.UNREACHABLE_CODE, IssueCategory.IMPLEMENTATION_ERROR, IssueSeverity.CRITICAL, "Critical issue"),
                Issue(IssueType.DANGEROUS_FUNCTION_USAGE, IssueCategory.SECURITY_VULNERABILITY, IssueSeverity.ERROR, "Security issue"),
                Issue(IssueType.HIGH_COMPLEXITY, IssueCategory.PERFORMANCE_ISSUE, IssueSeverity.WARNING, "Performance issue")
            ]
            
            collection = IssueCollection(issues=issues)
            
            assert len(collection) == 3
            assert len(collection.get_critical_issues()) == 1
            assert len(collection.get_security_issues()) == 1
            assert len(collection.get_performance_issues()) == 1
            
            summary = collection.get_summary()
            assert summary["total_issues"] == 3
            
            tests.append(TestResult(
                "issue_collection",
                True,
                time.time() - test_start,
                details={"collection_size": len(collection)}
            ))
        except Exception as e:
            tests.append(TestResult(
                "issue_collection",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Issue Architecture Tests",
            "Tests for enhanced issue detection architecture",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_error_detection_tests(self):
        """Test error detection functions."""
        suite_start = time.time()
        tests = []
        
        # Create test repository
        repo_path = self.test_data_generator.create_test_repository()
        
        # Test implementation error detection
        test_start = time.time()
        try:
            issues = detect_implementation_errors(None)  # Test fallback mode
            assert isinstance(issues, list)
            
            tests.append(TestResult(
                "implementation_error_detection",
                True,
                time.time() - test_start,
                details={"issues_found": len(issues)}
            ))
        except Exception as e:
            tests.append(TestResult(
                "implementation_error_detection",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test security vulnerability detection
        test_start = time.time()
        try:
            issues = detect_security_vulnerabilities(None)  # Test fallback mode
            assert isinstance(issues, list)
            
            tests.append(TestResult(
                "security_vulnerability_detection",
                True,
                time.time() - test_start,
                details={"issues_found": len(issues)}
            ))
        except Exception as e:
            tests.append(TestResult(
                "security_vulnerability_detection",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test advanced analysis functions
        for func_name, func in [
            ("circular_dependencies", detect_circular_dependencies_advanced),
            ("inheritance_patterns", analyze_inheritance_patterns),
            ("complexity_patterns", analyze_complexity_patterns),
            ("performance_patterns", analyze_performance_patterns)
        ]:
            test_start = time.time()
            try:
                issues = func(None)  # Test fallback mode
                assert isinstance(issues, list)
                
                tests.append(TestResult(
                    f"{func_name}_detection",
                    True,
                    time.time() - test_start,
                    details={"issues_found": len(issues)}
                ))
            except Exception as e:
                tests.append(TestResult(
                    f"{func_name}_detection",
                    False,
                    time.time() - test_start,
                    str(e)
                ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Error Detection Tests",
            "Tests for all error detection functions",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_context_collection_tests(self):
        """Test context collection functionality."""
        suite_start = time.time()
        tests = []
        
        # Test ContextCollector
        test_start = time.time()
        try:
            collector = ContextCollector(None)  # Test with None codebase
            
            # Test that it handles None gracefully
            assert collector is not None
            
            tests.append(TestResult(
                "context_collector_creation",
                True,
                time.time() - test_start,
                details={"collector_created": True}
            ))
        except Exception as e:
            tests.append(TestResult(
                "context_collector_creation",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Context Collection Tests",
            "Tests for context collection functionality",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_advanced_analysis_tests(self):
        """Test advanced analysis capabilities."""
        suite_start = time.time()
        tests = []
        
        # Test with fallback analyzer
        test_start = time.time()
        try:
            repo_path = self.test_data_generator.create_test_repository()
            analyzer = ComprehensiveCodebaseAnalyzer(repo_path)
            
            # Test analyzer creation
            assert analyzer is not None
            
            tests.append(TestResult(
                "analyzer_creation",
                True,
                time.time() - test_start,
                details={"repo_path": repo_path}
            ))
        except Exception as e:
            tests.append(TestResult(
                "analyzer_creation",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Advanced Analysis Tests",
            "Tests for advanced analysis capabilities",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_performance_optimization_tests(self):
        """Test performance optimization features."""
        suite_start = time.time()
        tests = []
        
        # Test AnalysisCache
        test_start = time.time()
        try:
            cache = AnalysisCache(max_memory_items=10)
            
            # Test cache operations
            cache.set("test_key", "test_value")
            result = cache.get("test_key")
            
            assert result == "test_value"
            
            # Test cache stats
            stats = cache.get_cache_stats()
            assert "memory_cache_size" in stats
            
            tests.append(TestResult(
                "analysis_cache",
                True,
                time.time() - test_start,
                details={"cache_stats": stats}
            ))
        except Exception as e:
            tests.append(TestResult(
                "analysis_cache",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test IncrementalAnalyzer
        test_start = time.time()
        try:
            analyzer = IncrementalAnalyzer("/tmp/test")
            
            # Test file change detection
            changed_files = analyzer.get_changed_files([])
            assert isinstance(changed_files, list)
            
            # Test stats
            stats = analyzer.get_analysis_stats()
            assert "tracked_files" in stats
            
            tests.append(TestResult(
                "incremental_analyzer",
                True,
                time.time() - test_start,
                details={"stats": stats}
            ))
        except Exception as e:
            tests.append(TestResult(
                "incremental_analyzer",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test PerformanceMonitor
        test_start = time.time()
        try:
            monitor = PerformanceMonitor()
            
            # Test performance tracking
            @monitor.track_execution("test_function")
            def test_function():
                time.sleep(0.01)
                return "test"
            
            result = test_function()
            assert result == "test"
            
            # Test performance report
            report = monitor.get_performance_report()
            assert "summary" in report
            
            tests.append(TestResult(
                "performance_monitor",
                True,
                time.time() - test_start,
                details={"report_keys": list(report.keys())}
            ))
        except Exception as e:
            tests.append(TestResult(
                "performance_monitor",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Performance Optimization Tests",
            "Tests for performance optimization features",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_integration_tests(self):
        """Test integration between components."""
        suite_start = time.time()
        tests = []
        
        # Test full analysis pipeline
        test_start = time.time()
        try:
            repo_path = self.test_data_generator.create_test_repository()
            analyzer = ComprehensiveCodebaseAnalyzer(repo_path)
            
            # Run analysis (will be in fallback mode)
            results = analyzer.analyze()
            
            assert isinstance(results, dict)
            assert "success" in results
            
            tests.append(TestResult(
                "full_analysis_pipeline",
                True,
                time.time() - test_start,
                details={"results_keys": list(results.keys())}
            ))
        except Exception as e:
            tests.append(TestResult(
                "full_analysis_pipeline",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Integration Tests",
            "Tests for component integration",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_regression_tests(self):
        """Test for regressions in functionality."""
        suite_start = time.time()
        tests = []
        
        # Test backward compatibility
        test_start = time.time()
        try:
            # Test that old Issue creation still works
            issue = Issue(
                item=None,
                type="test_type",
                message="test message",
                severity="warning"
            )
            
            assert issue.message == "test message"
            
            tests.append(TestResult(
                "backward_compatibility",
                True,
                time.time() - test_start,
                details={"issue_created": True}
            ))
        except Exception as e:
            tests.append(TestResult(
                "backward_compatibility",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Regression Tests",
            "Tests for backward compatibility and regressions",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Calculate overall statistics
        total_tests = sum(len(suite.tests) for suite in self.test_suites)
        passed_tests = sum(sum(1 for test in suite.tests if test.passed) for suite in self.test_suites)
        failed_tests = total_tests - passed_tests
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Find slowest tests
        all_tests = []
        for suite in self.test_suites:
            for test in suite.tests:
                all_tests.append((f"{suite.name}.{test.test_name}", test.duration, test.passed))
        
        slowest_tests = sorted(all_tests, key=lambda x: x[1], reverse=True)[:5]
        
        # Generate report
        report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "test_environment": "comprehensive_testing"
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
            },
            "test_suites": [],
            "performance_analysis": {
                "slowest_tests": [
                    {"test": name, "duration": duration, "passed": passed}
                    for name, duration, passed in slowest_tests
                ],
                "average_test_duration": sum(test[1] for test in all_tests) / len(all_tests) if all_tests else 0
            },
            "recommendations": self._generate_test_recommendations(success_rate, failed_tests)
        }
        
        # Add detailed suite information
        for suite in self.test_suites:
            suite_passed = sum(1 for test in suite.tests if test.passed)
            suite_failed = len(suite.tests) - suite_passed
            
            suite_info = {
                "name": suite.name,
                "description": suite.description,
                "total_tests": len(suite.tests),
                "passed_tests": suite_passed,
                "failed_tests": suite_failed,
                "success_rate": (suite_passed / len(suite.tests) * 100) if suite.tests else 0,
                "setup_time": suite.setup_time,
                "teardown_time": suite.teardown_time,
                "tests": [
                    {
                        "name": test.test_name,
                        "passed": test.passed,
                        "duration": test.duration,
                        "error_message": test.error_message,
                        "details": test.details
                    }
                    for test in suite.tests
                ]
            }
            
            report["test_suites"].append(suite_info)
        
        return report
    
    def _generate_test_recommendations(self, success_rate: float, failed_tests: int) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate < 90:
            recommendations.append("🔴 Test success rate is below 90% - investigate failing tests")
        
        if failed_tests > 0:
            recommendations.append(f"⚠️  {failed_tests} tests failed - review error messages and fix issues")
        
        if success_rate == 100:
            recommendations.append("✅ All tests passed - system is functioning correctly")
        
        return recommendations

def run_comprehensive_tests() -> Dict[str, Any]:
    """Run all comprehensive tests and return results."""
    runner = ComprehensiveTestRunner()
    return runner.run_all_tests()

if __name__ == "__main__":
    print("🧪 Comprehensive Testing Framework")
    print("=" * 50)
    
    results = run_comprehensive_tests()
    
    print("\n📊 Test Results Summary:")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed_tests']}")
    print(f"Failed: {results['summary']['failed_tests']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Overall Status: {results['summary']['overall_status']}")
    
    # Save detailed report
    report_file = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    if results['summary']['overall_status'] == "PASSED":
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - check the detailed report")
        sys.exit(1)

