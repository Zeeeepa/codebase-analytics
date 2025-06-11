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

class TestContextSummaryFunctions(unittest.TestCase):
    """Test suite for context summary functions"""

    def setUp(self):
        """Set up mock objects for testing"""
        # Mock Codebase
        self.mock_codebase = MagicMock()
        self.mock_codebase.ctx.get_nodes.return_value = list(range(100))
        self.mock_codebase.ctx.edges = [(1, 2, MagicMock(type="edge_type")) for _ in range(30)]
        self.mock_codebase.files = list(range(10))
        self.mock_codebase.imports = list(range(20))
        self.mock_codebase.external_modules = list(range(5))
        self.mock_codebase.symbols = list(range(50))
        self.mock_codebase.classes = list(range(15))
        self.mock_codebase.functions = list(range(25))
        self.mock_codebase.global_vars = list(range(10))
        self.mock_codebase.interfaces = list(range(5))
        
        # Set up edge types
        from graph_sitter.enums import EdgeType
        for i, edge in enumerate(self.mock_codebase.ctx.edges):
            if i < 10:
                edge[2].type = EdgeType.SYMBOL_USAGE
            elif i < 20:
                edge[2].type = EdgeType.IMPORT_SYMBOL_RESOLUTION
            else:
                edge[2].type = EdgeType.EXPORT
        
        # Mock SourceFile
        self.mock_file = MagicMock()
        self.mock_file.name = "test_file.py"
        self.mock_file.imports = list(range(5))
        self.mock_file.symbols = list(range(20))
        self.mock_file.classes = list(range(3))
        self.mock_file.functions = list(range(10))
        self.mock_file.global_vars = list(range(7))
        self.mock_file.interfaces = list(range(0))
        
        # Mock Class
        self.mock_class = MagicMock()
        self.mock_class.name = "TestClass"
        self.mock_class.parent_class_names = ["BaseClass", "MixinClass"]
        self.mock_class.methods = list(range(5))
        self.mock_class.attributes = list(range(3))
        self.mock_class.decorators = list(range(2))
        self.mock_class.dependencies = list(range(4))
        self.mock_class.symbol_usages = []
        
        # Mock Function
        self.mock_function = MagicMock()
        self.mock_function.name = "test_function"
        self.mock_function.return_statements = list(range(2))
        self.mock_function.parameters = list(range(3))
        self.mock_function.function_calls = list(range(4))
        self.mock_function.call_sites = list(range(2))
        self.mock_function.decorators = list(range(1))
        self.mock_function.dependencies = list(range(5))
        self.mock_function.symbol_usages = []
        
        # Mock Symbol
        self.mock_symbol = MagicMock()
        self.mock_symbol.name = "test_symbol"
        
        # Create mock usages
        from graph_sitter.enums import SymbolType
        from graph_sitter.core.symbol import Symbol
        from graph_sitter.core.import_resolution import Import
        from graph_sitter.core.external_module import ExternalModule
        from graph_sitter.core.file import SourceFile
        
        # Create mock usages for symbols
        mock_usages = []
        
        # Add function symbols
        for i in range(3):
            mock_func = MagicMock(spec=Symbol)
            mock_func.symbol_type = SymbolType.Function
            mock_usages.append(mock_func)
        
        # Add class symbols
        for i in range(2):
            mock_class = MagicMock(spec=Symbol)
            mock_class.symbol_type = SymbolType.Class
            mock_usages.append(mock_class)
        
        # Add global var symbols
        for i in range(4):
            mock_var = MagicMock(spec=Symbol)
            mock_var.symbol_type = SymbolType.GlobalVar
            mock_usages.append(mock_var)
        
        # Add interface symbols
        for i in range(1):
            mock_interface = MagicMock(spec=Symbol)
            mock_interface.symbol_type = SymbolType.Interface
            mock_usages.append(mock_interface)
        
        # Add imports with imported symbols
        mock_imports = []
        for i in range(5):
            mock_import = MagicMock(spec=Import)
            
            if i == 0:
                # Function import
                mock_imported = MagicMock(spec=Symbol)
                mock_imported.symbol_type = SymbolType.Function
                mock_import.imported_symbol = mock_imported
            elif i == 1:
                # Class import
                mock_imported = MagicMock(spec=Symbol)
                mock_imported.symbol_type = SymbolType.Class
                mock_import.imported_symbol = mock_imported
            elif i == 2:
                # Global var import
                mock_imported = MagicMock(spec=Symbol)
                mock_imported.symbol_type = SymbolType.GlobalVar
                mock_import.imported_symbol = mock_imported
            elif i == 3:
                # External module import
                mock_imported = MagicMock(spec=ExternalModule)
                mock_import.imported_symbol = mock_imported
            else:
                # File import
                mock_imported = MagicMock(spec=SourceFile)
                mock_imported.imports = []
                mock_imported.name = "imported_file.py"
                mock_import.imported_symbol = mock_imported
            
            mock_imports.append(mock_import)
        
        # Combine all usages
        mock_usages.extend(mock_imports)
        self.mock_symbol.symbol_usages = mock_usages
        self.mock_class.symbol_usages = mock_usages[:5]  # Just use a subset for class
        self.mock_function.symbol_usages = mock_usages[5:]  # Use another subset for function

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
        self.assertIn("3 functions", summary)
        self.assertIn("2 classes", summary)
        self.assertIn("4 global variables", summary)
        self.assertIn("1 interfaces", summary)
        self.assertIn("5 imports", summary)
        
        # Check that it's a string
        self.assertIsInstance(summary, str)

    def test_get_context_summary_codebase(self):
        """Test get_context_summary with a Codebase object"""
        with patch('backend.api.get_codebase_summary') as mock_get_codebase_summary:
            mock_get_codebase_summary.return_value = "Codebase summary"
            summary = get_context_summary(self.mock_codebase)
            self.assertEqual(summary, "Codebase summary")
            mock_get_codebase_summary.assert_called_once_with(self.mock_codebase)

    def test_get_context_summary_file(self):
        """Test get_context_summary with a SourceFile object"""
        with patch('backend.api.get_file_summary') as mock_get_file_summary:
            mock_get_file_summary.return_value = "File summary"
            summary = get_context_summary(self.mock_file)
            self.assertEqual(summary, "File summary")
            mock_get_file_summary.assert_called_once_with(self.mock_file)

    def test_get_context_summary_class(self):
        """Test get_context_summary with a Class object"""
        with patch('backend.api.get_class_summary') as mock_get_class_summary:
            mock_get_class_summary.return_value = "Class summary"
            summary = get_context_summary(self.mock_class)
            self.assertEqual(summary, "Class summary")
            mock_get_class_summary.assert_called_once_with(self.mock_class)

    def test_get_context_summary_function(self):
        """Test get_context_summary with a Function object"""
        with patch('backend.api.get_function_summary') as mock_get_function_summary:
            mock_get_function_summary.return_value = "Function summary"
            summary = get_context_summary(self.mock_function)
            self.assertEqual(summary, "Function summary")
            mock_get_function_summary.assert_called_once_with(self.mock_function)

    def test_get_context_summary_symbol(self):
        """Test get_context_summary with a Symbol object"""
        with patch('backend.api.get_symbol_summary') as mock_get_symbol_summary:
            mock_get_symbol_summary.return_value = "Symbol summary"
            summary = get_context_summary(self.mock_symbol)
            self.assertEqual(summary, "Symbol summary")
            mock_get_symbol_summary.assert_called_once_with(self.mock_symbol)

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
        with patch('backend.api._get_symbol_summary_dict') as mock_get_symbol_summary_dict:
            mock_get_symbol_summary_dict.return_value = {"name": "TestClass", "usages": {}}
            summary_dict = get_context_summary_dict(self.mock_class)
            self.assertEqual(summary_dict["type"], "Class")
            self.assertEqual(summary_dict["name"], "TestClass")
            self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_function(self):
        """Test get_context_summary_dict with a Function object"""
        with patch('backend.api._get_symbol_summary_dict') as mock_get_symbol_summary_dict:
            mock_get_symbol_summary_dict.return_value = {"name": "test_function", "usages": {}}
            summary_dict = get_context_summary_dict(self.mock_function)
            self.assertEqual(summary_dict["type"], "Function")
            self.assertEqual(summary_dict["name"], "test_function")
            self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_symbol(self):
        """Test get_context_summary_dict with a Symbol object"""
        with patch('backend.api._get_symbol_summary_dict') as mock_get_symbol_summary_dict:
            mock_get_symbol_summary_dict.return_value = {"name": "test_symbol", "usages": {}}
            summary_dict = get_context_summary_dict(self.mock_symbol)
            self.assertEqual(summary_dict["name"], "test_symbol")
            self.assertIsInstance(summary_dict, dict)

    def test_get_context_summary_dict_unsupported(self):
        """Test get_context_summary_dict with an unsupported object type"""
        summary_dict = get_context_summary_dict("not a valid context object")
        self.assertIn("error", summary_dict)
        self.assertIsInstance(summary_dict, dict)


if __name__ == "__main__":
    unittest.main()

