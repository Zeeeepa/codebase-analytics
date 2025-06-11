"""
Helper functions and mock objects for tests
"""

from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path so we can import the api module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from graph_sitter.enums import EdgeType, SymbolType
from graph_sitter.core.symbol import Symbol
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.class_definition import Class
from graph_sitter.core.function import Function
from graph_sitter.core.codebase import Codebase

def create_mock_codebase():
    """Create a mock Codebase object"""
    mock_codebase = MagicMock(spec=Codebase)
    mock_codebase.ctx.get_nodes.return_value = list(range(100))
    mock_codebase.ctx.edges = [(1, 2, MagicMock(type="edge_type")) for _ in range(30)]
    mock_codebase.files = list(range(10))
    mock_codebase.imports = list(range(20))
    mock_codebase.external_modules = list(range(5))
    mock_codebase.symbols = list(range(50))
    mock_codebase.classes = list(range(15))
    mock_codebase.functions = list(range(25))
    mock_codebase.global_vars = list(range(10))
    mock_codebase.interfaces = list(range(5))
    
    # Set up edge types
    for i, edge in enumerate(mock_codebase.ctx.edges):
        if i < 10:
            edge[2].type = EdgeType.SYMBOL_USAGE
        elif i < 20:
            edge[2].type = EdgeType.IMPORT_SYMBOL_RESOLUTION
        else:
            edge[2].type = EdgeType.EXPORT
    
    return mock_codebase

def create_mock_file():
    """Create a mock SourceFile object"""
    mock_file = MagicMock(spec=SourceFile)
    mock_file.name = "test_file.py"
    mock_file.imports = list(range(5))
    mock_file.symbols = list(range(20))
    mock_file.classes = list(range(3))
    mock_file.functions = list(range(10))
    mock_file.global_vars = list(range(7))
    mock_file.interfaces = list(range(0))
    
    return mock_file

def create_mock_class():
    """Create a mock Class object"""
    mock_class = MagicMock(spec=Class)
    mock_class.name = "TestClass"
    mock_class.parent_class_names = ["BaseClass", "MixinClass"]
    mock_class.methods = list(range(5))
    mock_class.attributes = list(range(3))
    mock_class.decorators = list(range(2))
    mock_class.dependencies = list(range(4))
    mock_class.symbol_usages = []
    
    return mock_class

def create_mock_function():
    """Create a mock Function object"""
    mock_function = MagicMock(spec=Function)
    mock_function.name = "test_function"
    mock_function.return_statements = list(range(2))
    mock_function.parameters = list(range(3))
    mock_function.function_calls = list(range(4))
    mock_function.call_sites = list(range(2))
    mock_function.decorators = list(range(1))
    mock_function.dependencies = list(range(5))
    mock_function.symbol_usages = []
    
    return mock_function

def create_mock_symbol():
    """Create a mock Symbol object"""
    mock_symbol = MagicMock(spec=Symbol)
    mock_symbol.name = "test_symbol"
    
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
    mock_symbol.symbol_usages = mock_usages
    
    return mock_symbol

