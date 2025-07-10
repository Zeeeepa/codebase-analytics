#!/usr/bin/env python3

import sys
import os
sys.path.append('backend')

# Simple analysis using graph_sitter directly
try:
    from graph_sitter.codebase.codebase_analysis import Codebase
    
    print("🚀 Starting analysis of allwefantasy/auto-coder.web...")
    
    # Initialize codebase
    codebase = Codebase.from_repo("allwefantasy/auto-coder.web")
    
    # Basic analysis
    files = list(codebase.files)
    functions = list(codebase.functions)
    classes = list(codebase.classes)
    symbols = list(codebase.symbols)
    imports = list(codebase.imports)
    
    print("\n" + "="*60)
    print("📊 AUTO-CODER.WEB REPOSITORY ANALYSIS")
    print("="*60)
    
    print(f"📁 Total Files: {len(files)}")
    print(f"🔧 Total Functions: {len(functions)}")
    print(f"🏗️ Total Classes: {len(classes)}")
    print(f"🔗 Total Symbols: {len(symbols)}")
    print(f"📦 Total Imports: {len(imports)}")
    
    # Language distribution
    language_stats = {}
    for file in files:
        if hasattr(file, 'filepath'):
            ext = file.filepath.split('.')[-1] if '.' in file.filepath else 'no_extension'
            language_stats[ext] = language_stats.get(ext, 0) + 1
    
    print(f"\n🌐 Language Distribution:")
    for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {lang}: {count} files")
    
    # Largest files
    file_sizes = []
    for file in files:
        if hasattr(file, 'source') and hasattr(file, 'filepath'):
            lines = len(file.source.splitlines())
            file_sizes.append((file.filepath, lines))
    
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"\n📄 Largest Files:")
    for filepath, lines in file_sizes[:10]:
        print(f"   {filepath}: {lines} lines")
    
    # Function analysis
    if functions:
        print(f"\n🔧 Function Analysis:")
        
        # Functions with most usages
        func_usages = []
        for func in functions:
            if hasattr(func, 'usages') and hasattr(func, 'name'):
                usage_count = len(func.usages)
                func_usages.append((func.name, usage_count))
        
        func_usages.sort(key=lambda x: x[1], reverse=True)
        print(f"   Most Called Functions:")
        for name, count in func_usages[:5]:
            print(f"     {name}: {count} usages")
        
        # Functions with most calls
        func_calls = []
        for func in functions:
            if hasattr(func, 'function_calls') and hasattr(func, 'name'):
                call_count = len(func.function_calls)
                func_calls.append((func.name, call_count))
        
        func_calls.sort(key=lambda x: x[1], reverse=True)
        print(f"   Functions Making Most Calls:")
        for name, count in func_calls[:5]:
            print(f"     {name}: {count} calls")
    
    # Class analysis
    if classes:
        print(f"\n🏗️ Class Analysis:")
        
        # Classes with most methods
        class_methods = []
        for cls in classes:
            if hasattr(cls, 'methods') and hasattr(cls, 'name'):
                method_count = len(list(cls.methods))
                class_methods.append((cls.name, method_count))
        
        class_methods.sort(key=lambda x: x[1], reverse=True)
        print(f"   Classes with Most Methods:")
        for name, count in class_methods[:5]:
            print(f"     {name}: {count} methods")
        
        # Inheritance analysis
        inheritance_chains = []
        for cls in classes:
            if hasattr(cls, 'superclasses') and hasattr(cls, 'name'):
                chain_depth = len(list(cls.superclasses))
                if chain_depth > 0:
                    inheritance_chains.append((cls.name, chain_depth))
        
        inheritance_chains.sort(key=lambda x: x[1], reverse=True)
        if inheritance_chains:
            print(f"   Classes with Inheritance:")
            for name, depth in inheritance_chains[:5]:
                print(f"     {name}: {depth} levels")
    
    # Import analysis
    if imports:
        print(f"\n📦 Import Analysis:")
        
        import_sources = {}
        external_imports = []
        
        for imp in imports:
            if hasattr(imp, 'source'):
                import_sources[imp.source] = import_sources.get(imp.source, 0) + 1
                
                # Check if external (simple heuristic)
                if not imp.source.startswith('.') and '/' not in imp.source:
                    external_imports.append(imp.source)
        
        print(f"   Most Imported Modules:")
        sorted_imports = sorted(import_sources.items(), key=lambda x: x[1], reverse=True)
        for source, count in sorted_imports[:10]:
            print(f"     {source}: {count} times")
        
        print(f"   External Dependencies:")
        unique_external = list(set(external_imports))[:10]
        for dep in unique_external:
            print(f"     {dep}")
    
    # Dead code analysis
    dead_functions = []
    for func in functions:
        if hasattr(func, 'usages') and hasattr(func, 'name'):
            if len(func.usages) == 0:
                # Simple entry point check
                if not any(pattern in func.name.lower() for pattern in ['main', 'run', 'start', 'handler', '__']):
                    dead_functions.append(func.name)
    
    if dead_functions:
        print(f"\n💀 Potential Dead Code:")
        print(f"   Unused Functions: {len(dead_functions)}")
        for name in dead_functions[:10]:
            print(f"     {name}")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
except Exception as e:
    print(f"❌ Analysis failed: {e}")
    import traceback
    traceback.print_exc()
