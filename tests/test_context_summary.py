"""
Tests for context summary functions in api.py
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path so we can import the api module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from backend.api import (
    get_codebase_summary,
    get_file_summary,
    get_class_summary,
    get_function_summary,
    get_symbol_summary,
    get_context_summary,
    get_context_summary_dict
)

# Import test helpers
from tests.helpers import (
    create_mock_codebase,
    create_mock_file,
    create_mock_class,
    create_mock_function,
    create_mock_symbol
)

class TestContextSummaryFunctions(unittest.TestCase):
    """Test suite for context summary functions"""

    def setUp(self):
        """Set up mock objects for testing"""
        self.mock_codebase = create_mock_codebase()
        self.mock_file = create_mock_file()
        self.mock_class = create_mock_class()
        self.mock_function = create_mock_function()
        self.mock_symbol = create_mock_symbol()

    def test_get_codebase_summary(self):
        """Test get_codebase_summary function"""
        summary = get_codebase_summary(self.mock_codebase)
        
        # Check that the summary contains expected information
        self.assertIn("Contains 100 nodes", summary)
        self.assertIn("10 files", summary)
        self.assertIn("20 imports", summary)
        self.assertIn("5 external_modules", summary)
        self.assertIn("50 symbols", summary)
        self.assertIn("15 classes", summary)
        self.assertIn("25 functions", summary)
        self.assertIn("10 global_vars", summary)
        self.assertIn("5 interfaces", summary)
        self.assertIn("Contains 30 edges", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_file_summary(self):
        """Test get_file_summary function"""
        summary = get_file_summary(self.mock_file)
        
        # Check that the summary contains expected information
        self.assertIn("test_file.py", summary)
        self.assertIn("5 imports", summary)
        self.assertIn("20 symbol references", summary)
        self.assertIn("3 classes", summary)
        self.assertIn("10 functions", summary)
        self.assertIn("7 global variables", summary)
        self.assertIn("0 interfaces", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_class_summary(self):
        """Test get_class_summary function"""
        summary = get_class_summary(self.mock_class)
        
        # Check that the summary contains expected information
        self.assertIn("TestClass", summary)
        self.assertIn("parent classes: ['BaseClass', 'MixinClass']", summary)
        self.assertIn("5 methods", summary)
        self.assertIn("3 attributes", summary)
        self.assertIn("2 decorators", summary)
        self.assertIn("4 dependencies", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_function_summary(self):
        """Test get_function_summary function"""
        summary = get_function_summary(self.mock_function)
        
        # Check that the summary contains expected information
        self.assertIn("test_function", summary)
        self.assertIn("2 return statements", summary)
        self.assertIn("3 parameters", summary)
        self.assertIn("4 function calls", summary)
        self.assertIn("2 call sites", summary)
        self.assertIn("1 decorators", summary)
        self.assertIn("5 dependencies", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_symbol_summary(self):
        """Test get_symbol_summary function"""
        summary = get_symbol_summary(self.mock_symbol)
        
        # Check that the summary contains expected information
        self.assertIn("test_symbol", summary)
        self.assertIn("15 usages", summary)  # Total of all mock usages
        self.assertIn("15 imports", summary)  # All usages are treated as imports in our mock
        self.assertIn("1 functions", summary)  # From imported symbols
        self.assertIn("1 classes", summary)  # From imported symbols
        self.assertIn("1 global variables", summary)  # From imported symbols
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_context_summary_codebase(self):
        """Test get_context_summary with a Codebase object"""
        summary = get_context_summary(self.mock_codebase)
        self.assertIn("Contains 100 nodes", summary)
        self.assertIn("10 files", summary)
        self.assertIn("20 imports", summary)

    def test_get_context_summary_file(self):
        """Test get_context_summary with a SourceFile object"""
        summary = get_context_summary(self.mock_file)
        self.assertIn("test_file.py", summary)
        self.assertIn("5 imports", summary)
        self.assertIn("20 symbol references", summary)

    def test_get_context_summary_class(self):
        """Test get_context_summary with a Class object"""
        summary = get_context_summary(self.mock_class)
        self.assertIn("TestClass", summary)
        self.assertIn("parent classes: ['BaseClass', 'MixinClass']", summary)
        self.assertIn("5 methods", summary)

    def test_get_context_summary_function(self):
        """Test get_context_summary with a Function object"""
        summary = get_context_summary(self.mock_function)
        self.assertIn("test_function", summary)
        self.assertIn("2 return statements", summary)
        self.assertIn("3 parameters", summary)

    def test_get_context_summary_symbol(self):
        """Test get_context_summary with a Symbol object"""
        summary = get_context_summary(self.mock_symbol)
        self.assertIn("test_symbol", summary)
        self.assertIn("15 usages", summary)
        self.assertIn("15 imports", summary)

    def test_get_context_summary_unsupported(self):
        """Test get_context_summary with an unsupported object type"""
        summary = get_context_summary("not a valid context object")
        self.assertIn("Unsupported context type", summary)
        self.assertIsInstance(summary, str)

    def test_get_context_summary_dict_codebase(self):
        """Test get_context_summary_dict with a Codebase object"""
        summary_dict = get_context_summary_dict(self.mock_codebase)
        self.assertEqual(summary_dict["type"], "Codebase")
        self.assertEqual(summary_dict["nodes"], 100)
        self.assertEqual(summary_dict["files"], 10)
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_file(self):
        """Test get_context_summary_dict with a SourceFile object"""
        summary_dict = get_context_summary_dict(self.mock_file)
        self.assertEqual(summary_dict["type"], "SourceFile")
        self.assertEqual(summary_dict["name"], "test_file.py")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_class(self):
        """Test get_context_summary_dict with a Class object"""
        summary_dict = get_context_summary_dict(self.mock_class)
        self.assertEqual(summary_dict["type"], "Class")
        self.assertEqual(summary_dict["name"], "TestClass")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_function(self):
        """Test get_context_summary_dict with a Function object"""
        summary_dict = get_context_summary_dict(self.mock_function)
        self.assertEqual(summary_dict["type"], "Function")
        self.assertEqual(summary_dict["name"], "test_function")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_symbol(self):
        """Test get_context_summary_dict with a Symbol object"""
        summary_dict = get_context_summary_dict(self.mock_symbol)
        self.assertEqual(summary_dict["name"], "test_symbol")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_unsupported(self):
        """Test get_context_summary_dict with an unsupported object type"""
        summary_dict = get_context_summary_dict("not a valid context object")
        self.assertIn("error", summary_dict)
        self.assertIsInstance(summary_dict, dict)

