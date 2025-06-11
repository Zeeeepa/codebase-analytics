#!/usr/bin/env python3
"""
Test runner for the codebase analytics project.
Runs all tests and reports results.
"""

import unittest
import sys
import os
import time
import argparse
from collections import defaultdict

# Add the parent directory to the path so we can import the backend module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import test modules
from tests.test_context_summary import TestContextSummary
from tests.test_upgrade_analysis import TestUpgradeAnalysis

def run_tests():
    """Run all tests and report results"""
    start_time = time.time()
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Create a custom test result class to track successes and failures
    class DetailedTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.successes = []
            self.test_timings = {}
            self.test_start_time = None
            
        def startTest(self, test):
            self.test_start_time = time.time()
            super().startTest(test)
            
        def addSuccess(self, test):
            super().addSuccess(test)
            self.successes.append(test)
            self.test_timings[test.id()] = time.time() - self.test_start_time
            
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_timings[test.id()] = time.time() - self.test_start_time
            
        def addError(self, test, err):
            super().addError(test, err)
            self.test_timings[test.id()] = time.time() - self.test_start_time
    
    # Run tests with detailed results
    result = unittest.TextTestRunner(
        resultclass=DetailedTestResult,
        verbosity=2
    ).run(test_suite)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print(f"TEST SUMMARY ({total_time:.2f}s)")
    print("="*80)
    
    # Group tests by module
    module_results = defaultdict(lambda: {"success": 0, "failure": 0, "error": 0, "time": 0})
    
    # Process successes
    for test in result.successes:
        module_name = test.id().split('.')[1]  # Get the module name from the test ID
        module_results[module_name]["success"] += 1
        module_results[module_name]["time"] += result.test_timings.get(test.id(), 0)
    
    # Process failures
    for test, _ in result.failures:
        module_name = test.id().split('.')[1]
        module_results[module_name]["failure"] += 1
        module_results[module_name]["time"] += result.test_timings.get(test.id(), 0)
    
    # Process errors
    for test, _ in result.errors:
        module_name = test.id().split('.')[1]
        module_results[module_name]["error"] += 1
        module_results[module_name]["time"] += result.test_timings.get(test.id(), 0)
    
    # Print module results
    print(f"{'Module':<30} {'Success':<10} {'Failure':<10} {'Error':<10} {'Time (s)':<10}")
    print("-"*80)
    
    for module, counts in sorted(module_results.items()):
        print(f"{module:<30} {counts['success']:<10} {counts['failure']:<10} {counts['error']:<10} {counts['time']:.2f}")
    
    # Print overall results
    print("-"*80)
    total_tests = len(result.successes) + len(result.failures) + len(result.errors)
    print(f"{'TOTAL':<30} {len(result.successes):<10} {len(result.failures):<10} {len(result.errors):<10} {total_time:.2f}")
    print("="*80)
    
    # Return success if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    # Add the current directory to the path so we can import modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests and exit with appropriate status code
    success = run_tests()
    sys.exit(0 if success else 1)
