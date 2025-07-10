#!/usr/bin/env python3
"""
Simple test to see what's available in the graph-sitter codebase.
"""

from graph_sitter.core.codebase import Codebase
from graph_sitter.configs.models.codebase import CodebaseConfig

def simple_test():
    """Test what's available in the codebase."""
    print("🔍 Testing graph-sitter codebase...")
    
    try:
        # Initialize codebase with basic configuration
        config = CodebaseConfig(
            allow_external=True,
        )
        
        # Analyze current directory
        codebase = Codebase(".", config=config)
        
        print(f"📊 Codebase initialized successfully!")
        print(f"📁 Available attributes: {[attr for attr in dir(codebase) if not attr.startswith('_')]}")
        
        # Test basic properties
        if hasattr(codebase, 'files'):
            files = codebase.files
            print(f"📄 Files: {len(list(files))} files found")
            
        if hasattr(codebase, 'functions'):
            functions = codebase.functions
            print(f"🔧 Functions: {len(list(functions))} functions found")
            
        if hasattr(codebase, 'classes'):
            classes = codebase.classes
            print(f"📦 Classes: {len(list(classes))} classes found")
            
        if hasattr(codebase, 'imports'):
            imports = codebase.imports
            print(f"🔗 Imports: {len(list(imports))} imports found")
            
        # Try to get a sample function
        if hasattr(codebase, 'functions'):
            functions_list = list(codebase.functions)
            if functions_list:
                sample_func = functions_list[0]
                print(f"📝 Sample function: {sample_func.name}")
                print(f"   Available attributes: {[attr for attr in dir(sample_func) if not attr.startswith('_')]}")
                
        # Try to get a sample file
        if hasattr(codebase, 'files'):
            files_list = list(codebase.files)
            if files_list:
                sample_file = files_list[0]
                print(f"📄 Sample file: {sample_file.path}")
                print(f"   Available attributes: {[attr for attr in dir(sample_file) if not attr.startswith('_')]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()

