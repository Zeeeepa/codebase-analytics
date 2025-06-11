"""
Mock implementations of graph-sitter classes for testing
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum


class EdgeType(Enum):
    """Mock EdgeType enum"""
    SYMBOL_USAGE = "SYMBOL_USAGE"
    IMPORT_SYMBOL_RESOLUTION = "IMPORT_SYMBOL_RESOLUTION"
    EXPORT = "EXPORT"
    
    @property
    def type(self):
        """Return the value of the enum"""
        return self.value


class SymbolType(Enum):
    """Mock SymbolType enum"""
    Function = "Function"
    Class = "Class"
    GlobalVar = "GlobalVar"
    Interface = "Interface"


class Context:
    """Mock Context class"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def get_nodes(self):
        """Return all nodes"""
        return self.nodes
    
    def add_node(self, node):
        """Add a node to the context"""
        self.nodes.append(node)
        return len(self.nodes) - 1  # Return node ID
    
    def add_edge(self, source, target, edge_type):
        """Add an edge to the context"""
        edge = (source, target, edge_type)
        self.edges.append(edge)
        return len(self.edges) - 1  # Return edge ID


class Codebase:
    """Mock Codebase class"""
    
    def __init__(self, ctx):
        self.ctx = ctx
        self.files = []
        self.imports = []
        self.external_modules = []
        self.symbols = []
        self.classes = []
        self.functions = []
        self.global_vars = []
        self.interfaces = []
    
    def add_file(self, file):
        """Add a file to the codebase"""
        self.files.append(file)


class SourceFile:
    """Mock SourceFile class"""
    
    def __init__(self, name, ctx):
        self.name = name
        self.ctx = ctx
        self.imports = []
        self.symbols = []
        self.classes = []
        self.functions = []
        self.global_vars = []
        self.interfaces = []
    
    def add_import(self, import_obj):
        """Add an import to the file"""
        self.imports.append(import_obj)
    
    def add_symbol(self, symbol):
        """Add a symbol to the file"""
        self.symbols.append(symbol)
    
    def add_class(self, cls):
        """Add a class to the file"""
        self.classes.append(cls)
        self.symbols.append(cls)
    
    def add_function(self, func):
        """Add a function to the file"""
        self.functions.append(func)
        self.symbols.append(func)
    
    def add_global_var(self, var):
        """Add a global variable to the file"""
        self.global_vars.append(var)
        self.symbols.append(var)
    
    def add_interface(self, interface):
        """Add an interface to the file"""
        self.interfaces.append(interface)
        self.symbols.append(interface)


class Class:
    """Mock Class class"""
    
    def __init__(self, name, file, ctx):
        self.name = name
        self.file = file
        self.ctx = ctx
        self.parent_class_names = []
        self.methods = []
        self.attributes = []
        self.decorators = []
        self.dependencies = []
        self.symbol_usages = []
        self.symbol_type = SymbolType.Class


class Function:
    """Mock Function class"""
    
    def __init__(self, name, file, ctx):
        self.name = name
        self.file = file
        self.ctx = ctx
        self.return_statements = []
        self.parameters = []
        self.function_calls = []
        self.call_sites = []
        self.decorators = []
        self.dependencies = []
        self.symbol_usages = []
        self.symbol_type = SymbolType.Function


class Symbol:
    """Mock Symbol class"""
    
    def __init__(self, name, file, ctx):
        self.name = name
        self.file = file
        self.ctx = ctx
        self.symbol_usages = []
        self.symbol_type = SymbolType.GlobalVar


class Import:
    """Mock Import class"""
    
    def __init__(self, name, file, ctx):
        self.name = name
        self.file = file
        self.ctx = ctx
        self.imported_symbol = None


class ExternalModule:
    """Mock ExternalModule class"""
    
    def __init__(self, name, ctx):
        self.name = name
        self.external_name = name
        self.ctx = ctx

