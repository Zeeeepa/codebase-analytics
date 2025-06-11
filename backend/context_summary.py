#!/usr/bin/env python3
"""
Context summary functions for codebase analytics.
These functions provide summaries of different code elements.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union

# Import graph-sitter components
from graph_sitter.core.class_definition import Class
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.function import Function
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType


def get_codebase_summary(codebase: 'Codebase') -> str:
    """
    Generate a comprehensive summary of a codebase.
    
    Args:
        codebase: The Codebase object to summarize
        
    Returns:
        A formatted string with node and edge statistics
    """
    node_summary = f"""Contains {len(codebase.ctx.get_nodes())} nodes
- {len(list(codebase.files))} files
- {len(list(codebase.imports))} imports
- {len(list(codebase.external_modules))} external_modules
- {len(list(codebase.symbols))} symbols
\t- {len(list(codebase.classes))} classes
\t- {len(list(codebase.functions))} functions
\t- {len(list(codebase.global_vars))} global_vars
\t- {len(list(codebase.interfaces))} interfaces
"""
    edge_summary = f"""Contains {len(codebase.ctx.edges)} edges
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.SYMBOL_USAGE])} symbol -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION])} import -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.EXPORT])} export -> exported symbol
    """

    return f"{node_summary}\n{edge_summary}"


def get_file_summary(file: 'SourceFile') -> str:
    """
    Generate a summary of a source file.
    
    Args:
        file: The SourceFile object to summarize
        
    Returns:
        A formatted string with file statistics
    """
    return f"""==== [ {file.name} ] ====
- {len(file.imports)} imports
- {len(file.symbols)} symbol references
\t- {len(file.classes)} classes
\t- {len(file.functions)} functions
\t- {len(file.global_vars)} global variables
\t- {len(file.interfaces)} interfaces
    """


def get_class_summary(cls: 'Class') -> str:
    """
    Generate a summary of a class.
    
    Args:
        cls: The Class object to summarize
        
    Returns:
        A formatted string with class statistics
    """
    return f"""==== [ {cls.name} ] ====
- parent classes: {cls.parent_class_names}
- {len(cls.methods)} methods
- {len(cls.attributes)} attributes
- {len(cls.decorators)} decorators
- {len(cls.dependencies)} dependencies
    """


def get_function_summary(func: 'Function') -> str:
    """
    Generate a summary of a function.
    
    Args:
        func: The Function object to summarize
        
    Returns:
        A formatted string with function statistics
    """
    return f"""==== [ {func.name} ] ====
- {len(func.return_statements)} return statements
- {len(func.parameters)} parameters
- {len(func.function_calls)} function calls
- {len(func.call_sites)} call sites
- {len(func.decorators)} decorators
- {len(func.dependencies)} dependencies
    """


def get_symbol_summary(symbol: 'Symbol') -> str:
    """
    Generate a summary of a symbol.
    
    Args:
        symbol: The Symbol object to summarize
        
    Returns:
        A formatted string with symbol usage statistics
    """
    # Count usages by type
    function_count = 0
    class_count = 0
    global_var_count = 0
    interface_count = 0
    import_count = 0
    
    # Count imported symbols by type
    imported_functions = 0
    imported_classes = 0
    imported_global_vars = 0
    imported_interfaces = 0
    imported_external_modules = 0
    imported_files = 0
    
    # Process all symbol usages
    for usage in symbol.symbol_usages:
        # Check if it's an import
        usage_type = type(usage).__name__
        if usage_type == 'Import':
            import_count += 1
            imported_symbol = usage.imported_symbol
            
            # Check the type of the imported symbol
            if hasattr(imported_symbol, 'symbol_type'):
                if imported_symbol.symbol_type == SymbolType.Function:
                    imported_functions += 1
                elif imported_symbol.symbol_type == SymbolType.Class:
                    imported_classes += 1
                elif imported_symbol.symbol_type == SymbolType.GlobalVar:
                    imported_global_vars += 1
                elif imported_symbol.symbol_type == SymbolType.Interface:
                    imported_interfaces += 1
            # Check if it's an external module
            elif type(imported_symbol).__name__ == 'ExternalModule':
                imported_external_modules += 1
            # Check if it's a file
            elif type(imported_symbol).__name__ == 'SourceFile':
                imported_files += 1
        # Check if it's a symbol
        elif usage_type == 'Symbol':
            if usage.symbol_type == SymbolType.Function:
                function_count += 1
            elif usage.symbol_type == SymbolType.Class:
                class_count += 1
            elif usage.symbol_type == SymbolType.GlobalVar:
                global_var_count += 1
            elif usage.symbol_type == SymbolType.Interface:
                interface_count += 1
    
    # Format the summary
    summary = f"""==== [ `{symbol.name}` ({type(symbol).__name__}) Usage Summary ] ====
- {len(symbol.symbol_usages)} usages
\t- {function_count} functions
\t- {class_count} classes
\t- {global_var_count} global variables
\t- {interface_count} interfaces
\t- {import_count} imports
\t\t- {imported_functions} functions
\t\t- {imported_classes} classes
\t\t- {imported_global_vars} global variables
\t\t- {imported_interfaces} interfaces
\t\t- {imported_external_modules} external modules
\t\t- {imported_files} files
    """
    
    return summary


def get_context_summary(context: Union['Codebase', 'SourceFile', 'Class', 'Function', 'Symbol']) -> str:
    """
    Generate a summary for any context object.
    
    Args:
        context: The context object to summarize (Codebase, SourceFile, Class, Function, or Symbol)
        
    Returns:
        A formatted string with context-specific summary information
    """
    # Check the type of the context object by name
    context_type = type(context).__name__
    
    if context_type == 'Codebase':
        return get_codebase_summary(context)
    elif context_type == 'SourceFile':
        return get_file_summary(context)
    elif context_type == 'Class':
        return get_class_summary(context)
    elif context_type == 'Function':
        return get_function_summary(context)
    elif context_type == 'Symbol':
        return get_symbol_summary(context)
    else:
        return f"Unsupported context type: {context_type}"


def get_context_summary_dict(context: Union['Codebase', 'SourceFile', 'Class', 'Function', 'Symbol']) -> Dict[str, Any]:
    """
    Generate a dictionary summary for any context object.
    
    Args:
        context: The context object to summarize (Codebase, SourceFile, Class, Function, or Symbol)
        
    Returns:
        A dictionary with context-specific summary information
    """
    # Check the type of the context object by name
    context_type = type(context).__name__
    
    if context_type == 'Codebase':
        return {
            "type": "Codebase",
            "nodes": len(context.ctx.get_nodes()),
            "edges": len(context.ctx.edges),
            "files": len(list(context.files)),
            "imports": len(list(context.imports)),
            "external_modules": len(list(context.external_modules)),
            "symbols": {
                "total": len(list(context.symbols)),
                "classes": len(list(context.classes)),
                "functions": len(list(context.functions)),
                "global_vars": len(list(context.global_vars)),
                "interfaces": len(list(context.interfaces))
            },
            "edge_types": {
                "symbol_usage": len([x for x in context.ctx.edges if x[2].type == EdgeType.SYMBOL_USAGE]),
                "import_resolution": len([x for x in context.ctx.edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION]),
                "export": len([x for x in context.ctx.edges if x[2].type == EdgeType.EXPORT])
            }
        }
    elif context_type == 'SourceFile':
        return {
            "type": "SourceFile",
            "name": context.name,
            "imports": len(context.imports),
            "symbols": {
                "total": len(context.symbols),
                "classes": len(context.classes),
                "functions": len(context.functions),
                "global_vars": len(context.global_vars),
                "interfaces": len(context.interfaces)
            },
            "importers": len(context.imports)
        }
    elif context_type == 'Class':
        return {
            "type": "Class",
            "name": context.name,
            "parent_classes": context.parent_class_names,
            "methods": len(context.methods),
            "attributes": len(context.attributes),
            "decorators": len(context.decorators),
            "dependencies": len(context.dependencies),
            "usages": _get_symbol_summary_dict(context)
        }
    elif context_type == 'Function':
        return {
            "type": "Function",
            "name": context.name,
            "return_statements": len(context.return_statements),
            "parameters": len(context.parameters),
            "function_calls": len(context.function_calls),
            "call_sites": len(context.call_sites),
            "decorators": len(context.decorators),
            "dependencies": len(context.dependencies),
            "usages": _get_symbol_summary_dict(context)
        }
    elif context_type == 'Symbol':
        return _get_symbol_summary_dict(context)
    else:
        return {"error": f"Unsupported context type: {context_type}"}


def _get_symbol_summary_dict(symbol: 'Symbol') -> Dict[str, Any]:
    """
    Generate a dictionary summary of a symbol's usage.
    
    Args:
        symbol: The Symbol object to summarize
        
    Returns:
        A dictionary with symbol usage information
    """
    # Count usages by type
    function_count = 0
    class_count = 0
    global_var_count = 0
    interface_count = 0
    import_count = 0
    
    # Count imported symbols by type
    imported_functions = 0
    imported_classes = 0
    imported_global_vars = 0
    imported_interfaces = 0
    imported_external_modules = 0
    imported_files = 0
    
    # Process all symbol usages
    for usage in symbol.symbol_usages:
        # Check if it's an import
        usage_type = type(usage).__name__
        if usage_type == 'Import':
            import_count += 1
            imported_symbol = usage.imported_symbol
            
            # Check the type of the imported symbol
            if hasattr(imported_symbol, 'symbol_type'):
                if imported_symbol.symbol_type == SymbolType.Function:
                    imported_functions += 1
                elif imported_symbol.symbol_type == SymbolType.Class:
                    imported_classes += 1
                elif imported_symbol.symbol_type == SymbolType.GlobalVar:
                    imported_global_vars += 1
                elif imported_symbol.symbol_type == SymbolType.Interface:
                    imported_interfaces += 1
            # Check if it's an external module
            elif type(imported_symbol).__name__ == 'ExternalModule':
                imported_external_modules += 1
            # Check if it's a file
            elif type(imported_symbol).__name__ == 'SourceFile':
                imported_files += 1
        # Check if it's a symbol
        elif usage_type == 'Symbol':
            if usage.symbol_type == SymbolType.Function:
                function_count += 1
            elif usage.symbol_type == SymbolType.Class:
                class_count += 1
            elif usage.symbol_type == SymbolType.GlobalVar:
                global_var_count += 1
            elif usage.symbol_type == SymbolType.Interface:
                interface_count += 1
    
    return {
        "type": type(symbol).__name__,
        "name": symbol.name,
        "total_usages": len(symbol.symbol_usages),
        "usage_types": {
            "functions": function_count,
            "classes": class_count,
            "global_vars": global_var_count,
            "interfaces": interface_count
        },
        "imports": {
            "total": import_count,
            "functions": imported_functions,
            "classes": imported_classes,
            "global_vars": imported_global_vars,
            "interfaces": imported_interfaces,
            "external_modules": imported_external_modules,
            "files": imported_files
        }
    }
