"""
Tests for context summary functions
"""

import unittest
import sys
import os
from typing import List

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from backend.context_summary import (
    get_codebase_summary,
    get_file_summary,
    get_class_summary,
    get_function_summary,
    get_symbol_summary,
    get_context_summary,
    get_context_summary_dict
)

# Import mock graph-sitter components for testing
from tests.mock_graph_sitter import (
    Context,
    Codebase,
    SourceFile,
    Class,
    Function,
    Symbol,
    Import,
    ExternalModule,
    EdgeType,
    SymbolType
)


class TestContextSummaryFunctions(unittest.TestCase):
    """Test suite for context summary functions"""

    def setUp(self):
        """Set up real objects for testing"""
        # Create a context
        self.ctx = Context()
        
        # Create a codebase
        self.codebase = Codebase(self.ctx)
        
        # Create a file
        self.file = SourceFile("test_file.py", self.ctx)
        self.codebase.add_file(self.file)
        
        # Create a class
        self.cls = Class("TestClass", self.file, self.ctx)
        self.cls.parent_class_names = ["BaseClass", "MixinClass"]
        self.file.add_class(self.cls)
        
        # Create a function
        self.func = Function("test_function", self.file, self.ctx)
        self.file.add_function(self.func)
        
        # Create a symbol
        self.symbol = Symbol("test_symbol", self.file, self.ctx)
        self.symbol.symbol_type = SymbolType.GlobalVar
        self.file.add_symbol(self.symbol)
        
        # Add some edges to the context
        self.ctx.add_edge(self.symbol, self.func, EdgeType.SYMBOL_USAGE)
        self.ctx.add_edge(self.func, self.cls, EdgeType.SYMBOL_USAGE)
        
        # Create an import
        self.import_obj = Import("os", self.file, self.ctx)
        self.import_obj.imported_symbol = ExternalModule("os", self.ctx)
        self.file.add_import(self.import_obj)
        
        # Add the import to the symbol's usages
        self.symbol.symbol_usages.append(self.import_obj)
        
        # Add some attributes to the objects
        self.cls.methods = [Function(f"method_{i}", self.file, self.ctx) for i in range(3)]
        self.cls.attributes = [Symbol(f"attr_{i}", self.file, self.ctx) for i in range(2)]
        self.cls.decorators = [Symbol(f"decorator_{i}", self.file, self.ctx) for i in range(1)]
        self.cls.dependencies = [Symbol(f"dependency_{i}", self.file, self.ctx) for i in range(2)]
        
        self.func.return_statements = ["return True", "return False"]
        self.func.parameters = ["param1", "param2"]
        self.func.function_calls = [Function(f"called_func_{i}", self.file, self.ctx) for i in range(3)]
        self.func.call_sites = ["site1", "site2"]
        self.func.decorators = [Symbol("func_decorator", self.file, self.ctx)]
        self.func.dependencies = [Symbol(f"func_dependency_{i}", self.file, self.ctx) for i in range(2)]

    def test_get_codebase_summary(self):
        """Test get_codebase_summary function"""
        summary = get_codebase_summary(self.codebase)
        
        # Check that the summary contains expected information
        self.assertIn("Contains", summary)
        self.assertIn("files", summary)
        self.assertIn("imports", summary)
        self.assertIn("symbols", summary)
        self.assertIn("edges", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_file_summary(self):
        """Test get_file_summary function"""
        summary = get_file_summary(self.file)
        
        # Check that the summary contains expected information
        self.assertIn("test_file.py", summary)
        self.assertIn("imports", summary)
        self.assertIn("symbol references", summary)
        self.assertIn("classes", summary)
        self.assertIn("functions", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_class_summary(self):
        """Test get_class_summary function"""
        summary = get_class_summary(self.cls)
        
        # Check that the summary contains expected information
        self.assertIn("TestClass", summary)
        self.assertIn("parent classes", summary)
        self.assertIn("BaseClass", summary)
        self.assertIn("MixinClass", summary)
        self.assertIn("methods", summary)
        self.assertIn("attributes", summary)
        self.assertIn("decorators", summary)
        self.assertIn("dependencies", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_function_summary(self):
        """Test get_function_summary function"""
        summary = get_function_summary(self.func)
        
        # Check that the summary contains expected information
        self.assertIn("test_function", summary)
        self.assertIn("return statements", summary)
        self.assertIn("parameters", summary)
        self.assertIn("function calls", summary)
        self.assertIn("call sites", summary)
        self.assertIn("decorators", summary)
        self.assertIn("dependencies", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_symbol_summary(self):
        """Test get_symbol_summary function"""
        summary = get_symbol_summary(self.symbol)
        
        # Check that the summary contains expected information
        self.assertIn("test_symbol", summary)
        self.assertIn("usages", summary)
        self.assertIn("imports", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_context_summary_codebase(self):
        """Test get_context_summary with a Codebase object"""
        summary = get_context_summary(self.codebase)
        self.assertIn("Contains", summary)
        self.assertIn("files", summary)
        self.assertIn("imports", summary)

    def test_get_context_summary_file(self):
        """Test get_context_summary with a SourceFile object"""
        summary = get_context_summary(self.file)
        self.assertIn("test_file.py", summary)
        self.assertIn("imports", summary)
        self.assertIn("symbol references", summary)

    def test_get_context_summary_class(self):
        """Test get_context_summary with a Class object"""
        summary = get_context_summary(self.cls)
        self.assertIn("TestClass", summary)
        self.assertIn("parent classes", summary)
        self.assertIn("methods", summary)

    def test_get_context_summary_function(self):
        """Test get_context_summary with a Function object"""
        summary = get_context_summary(self.func)
        self.assertIn("test_function", summary)
        self.assertIn("return statements", summary)
        self.assertIn("parameters", summary)

    def test_get_context_summary_symbol(self):
        """Test get_context_summary with a Symbol object"""
        summary = get_context_summary(self.symbol)
        self.assertIn("test_symbol", summary)
        self.assertIn("usages", summary)

    def test_get_context_summary_unsupported(self):
        """Test get_context_summary with an unsupported object type"""
        summary = get_context_summary("not a valid context object")
        self.assertIn("Unsupported context type", summary)
        self.assertIsInstance(summary, str)

    def test_get_context_summary_dict_codebase(self):
        """Test get_context_summary_dict with a Codebase object"""
        summary_dict = get_context_summary_dict(self.codebase)
        self.assertEqual(summary_dict["type"], "Codebase")
        self.assertIsInstance(summary_dict["nodes"], int)
        self.assertIsInstance(summary_dict["files"], int)
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_file(self):
        """Test get_context_summary_dict with a SourceFile object"""
        summary_dict = get_context_summary_dict(self.file)
        self.assertEqual(summary_dict["type"], "SourceFile")
        self.assertEqual(summary_dict["name"], "test_file.py")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_class(self):
        """Test get_context_summary_dict with a Class object"""
        summary_dict = get_context_summary_dict(self.cls)
        self.assertEqual(summary_dict["type"], "Class")
        self.assertEqual(summary_dict["name"], "TestClass")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_function(self):
        """Test get_context_summary_dict with a Function object"""
        summary_dict = get_context_summary_dict(self.func)
        self.assertEqual(summary_dict["type"], "Function")
        self.assertEqual(summary_dict["name"], "test_function")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_symbol(self):
        """Test get_context_summary_dict with a Symbol object"""
        summary_dict = get_context_summary_dict(self.symbol)
        self.assertEqual(summary_dict["name"], "test_symbol")
        self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_unsupported(self):
        """Test get_context_summary_dict with an unsupported object type"""
        summary_dict = get_context_summary_dict("not a valid context object")
        self.assertIn("error", summary_dict)
        self.assertIsInstance(summary_dict, dict)


if __name__ == "__main__":
    unittest.main()
