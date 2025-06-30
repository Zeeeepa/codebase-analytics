#!/usr/bin/env python3
"""
Enhanced consolidation script that creates a fully functional unified API
by intelligently merging the three codebase analytics files.
"""

import ast
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass

@dataclass
class ConsolidationConfig:
    """Configuration for the consolidation process."""
    input_files: List[str]
    output_file: str
    preserve_fastapi_endpoints: bool = True
    merge_duplicate_classes: bool = True
    add_comprehensive_cli: bool = True
    generate_unified_main: bool = True

class EnhancedConsolidator:
    """Enhanced consolidator that creates a fully functional unified API."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
        self.file_contents = {}
        self.fastapi_endpoints = []
        self.class_definitions = {}
        self.function_definitions = {}
        self.imports = set()
        
    def load_files(self):
        """Load and parse all input files."""
        print("📂 Loading source files...")
        
        for file_path in self.config.input_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.file_contents[file_path] = content
                    print(f"  ✅ Loaded {file_path}")
            else:
                print(f"  ❌ File not found: {file_path}")
                
    def extract_fastapi_endpoints(self):
        """Extract FastAPI endpoint definitions."""
        print("🔍 Extracting FastAPI endpoints...")
        
        for file_path, content in self.file_contents.items():
            # Find FastAPI app decorators and their functions
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('@fastapi_app.') or line.startswith('@app.'):
                    # Found an endpoint decorator
                    endpoint_lines = [line]
                    i += 1
                    
                    # Get the function definition
                    while i < len(lines) and not lines[i].strip().startswith('def '):
                        if lines[i].strip():  # Skip empty lines
                            endpoint_lines.append(lines[i])
                        i += 1
                    
                    if i < len(lines):
                        # Get the function definition and body
                        func_start = i
                        indent_level = len(lines[i]) - len(lines[i].lstrip())
                        endpoint_lines.append(lines[i])  # def line
                        i += 1
                        
                        # Get function body
                        while i < len(lines):
                            current_line = lines[i]
                            if current_line.strip() == '':
                                endpoint_lines.append(current_line)
                            elif len(current_line) - len(current_line.lstrip()) > indent_level:
                                endpoint_lines.append(current_line)
                            else:
                                break
                            i += 1
                        
                        self.fastapi_endpoints.append('\n'.join(endpoint_lines))
                        print(f"  📍 Found endpoint: {lines[func_start].strip()}")
                else:
                    i += 1
                    
    def extract_imports(self):
        """Extract and deduplicate imports."""
        print("📦 Processing imports...")
        
        for file_path, content in self.file_contents.items():
            lines = content.split('\n')
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    # Skip relative imports and problematic ones
                    if not any(skip in stripped for skip in ['from .', 'from codegen_on_oss']):
                        self.imports.add(stripped)
                        
        print(f"  📋 Processed {len(self.imports)} unique imports")
        
    def extract_key_classes(self):
        """Extract key class definitions with priority handling."""
        print("🏗️ Extracting class definitions...")
        
        # Priority order: analyzer.py > comprehensive_analysis.py > api.py
        file_priority = {
            'analyzer.py': 3,
            'comprehensive_analysis.py': 2,
            'api.py': 1
        }
        
        for file_path, content in self.file_contents.items():
            priority = max([v for k, v in file_priority.items() if k in file_path], default=0)
            
            # Parse AST to extract classes
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        
                        # Skip if we already have a higher priority version
                        if class_name in self.class_definitions:
                            if self.class_definitions[class_name]['priority'] >= priority:
                                continue
                                
                        # Extract class source
                        class_source = ast.unparse(node)
                        self.class_definitions[class_name] = {
                            'source': class_source,
                            'file': file_path,
                            'priority': priority
                        }
                        print(f"  🏛️ Extracted class: {class_name} from {file_path}")
                        
            except SyntaxError as e:
                print(f"  ❌ Syntax error in {file_path}: {e}")
                
    def extract_key_functions(self):
        """Extract key function definitions."""
        print("⚙️ Extracting function definitions...")
        
        # Functions to prioritize from each file
        key_functions = {
            'analyzer.py': ['analyze_codebase', 'create_issue', 'get_codebase_summary'],
            'comprehensive_analysis.py': ['main', 'analyze'],
            'api.py': ['get_monthly_commits', 'calculate_cyclomatic_complexity', 'analyze_repo']
        }
        
        for file_path, content in self.file_contents.items():
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        
                        # Check if this is a key function for this file
                        file_key = next((k for k in key_functions.keys() if k in file_path), None)
                        if file_key and func_name in key_functions[file_key]:
                            func_source = ast.unparse(node)
                            self.function_definitions[func_name] = {
                                'source': func_source,
                                'file': file_path
                            }
                            print(f"  🔧 Extracted function: {func_name} from {file_path}")
                            
            except SyntaxError as e:
                print(f"  ❌ Syntax error in {file_path}: {e}")
                
    def generate_unified_file(self):
        """Generate the unified API file."""
        print("🚀 Generating unified API file...")
        
        output_lines = []
        
        # File header
        output_lines.extend([
            '#!/usr/bin/env python3',
            '"""',
            'Unified Codebase Analytics API',
            '',
            'This file consolidates functionality from:',
            '- comprehensive_analysis.py: Deep codebase analysis with Codegen SDK',
            '- api.py: FastAPI web service with metrics calculation',
            '- analyzer.py: Advanced issue management and transaction support',
            '',
            'Features:',
            '- Comprehensive codebase analysis',
            '- REST API endpoints for web access',
            '- Advanced issue detection and management',
            '- Multiple output formats (JSON, HTML, console)',
            '- CLI and web interfaces',
            '',
            'Generated by enhanced_consolidation.py',
            '"""',
            ''
        ])
        
        # Add imports
        output_lines.append('# =' * 40)
        output_lines.append('# IMPORTS')
        output_lines.append('# =' * 40)
        output_lines.append('')
        
        # Sort imports: standard library first, then third-party, then local
        stdlib_imports = []
        thirdparty_imports = []
        local_imports = []
        
        for imp in sorted(self.imports):
            if any(stdlib in imp for stdlib in ['import os', 'import sys', 'import time', 'import json', 'import logging', 'import argparse', 'import math', 'import re', 'import subprocess', 'import tempfile', 'import hashlib', 'import socket', 'import traceback']):
                stdlib_imports.append(imp)
            elif imp.startswith('from codegen'):
                local_imports.append(imp)
            else:
                thirdparty_imports.append(imp)
                
        for imp_list in [stdlib_imports, thirdparty_imports, local_imports]:
            if imp_list:
                output_lines.extend(imp_list)
                output_lines.append('')
                
        # Add essential imports that might be missing
        essential_imports = [
            'import uvicorn',
            'from fastapi import FastAPI, HTTPException',
            'from fastapi.middleware.cors import CORSMiddleware',
            'from pydantic import BaseModel',
            'from typing import Dict, List, Optional, Any',
            'from datetime import datetime',
            'import modal'
        ]
        
        for imp in essential_imports:
            if imp not in self.imports:
                output_lines.append(imp)
        output_lines.append('')
        
        # Add FastAPI app initialization
        output_lines.extend([
            '# =' * 40,
            '# FASTAPI APP INITIALIZATION',
            '# =' * 40,
            '',
            '# Modal image configuration',
            'image = (',
            '    modal.Image.debian_slim()',
            '    .apt_install("git")',
            '    .pip_install(',
            '        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic",',
            '        "networkx", "datetime"',
            '    )',
            ')',
            '',
            'app = modal.App(name="unified-analytics-app", image=image)',
            'fastapi_app = FastAPI(title="Unified Codebase Analytics API", version="1.0.0")',
            '',
            'fastapi_app.add_middleware(',
            '    CORSMiddleware,',
            '    allow_origins=["*"],',
            '    allow_credentials=True,',
            '    allow_methods=["*"],',
            '    allow_headers=["*"],',
            ')',
            ''
        ])
        
        # Add class definitions
        output_lines.append('# =' * 40)
        output_lines.append('# CONSOLIDATED CLASSES')
        output_lines.append('# =' * 40)
        output_lines.append('')
        
        for class_name, class_info in self.class_definitions.items():
            output_lines.append(f"# From {class_info['file']}")
            output_lines.append(class_info['source'])
            output_lines.append('')
            
        # Add function definitions
        output_lines.append('# =' * 40)
        output_lines.append('# CORE FUNCTIONS')
        output_lines.append('# =' * 40)
        output_lines.append('')
        
        for func_name, func_info in self.function_definitions.items():
            output_lines.append(f"# From {func_info['file']}")
            output_lines.append(func_info['source'])
            output_lines.append('')
            
        # Add FastAPI endpoints
        if self.fastapi_endpoints:
            output_lines.append('# =' * 40)
            output_lines.append('# API ENDPOINTS')
            output_lines.append('# =' * 40)
            output_lines.append('')
            
            for endpoint in self.fastapi_endpoints:
                output_lines.append(endpoint)
                output_lines.append('')
                
        # Add Modal deployment
        output_lines.extend([
            '# =' * 40,
            '# MODAL DEPLOYMENT',
            '# =' * 40,
            '',
            '@app.function(image=image)',
            '@modal.asgi_app()',
            'def fastapi_modal_app():',
            '    return fastapi_app',
            ''
        ])
        
        # Add main execution block
        output_lines.extend([
            '# =' * 40,
            '# MAIN EXECUTION',
            '# =' * 40,
            '',
            'if __name__ == "__main__":',
            '    import argparse',
            '    import socket',
            '    ',
            '    parser = argparse.ArgumentParser(',
            '        description="Unified Codebase Analytics API"',
            '    )',
            '    parser.add_argument(',
            '        "--mode", ',
            '        choices=["api", "cli", "analyze"],',
            '        default="api",',
            '        help="Run mode: api (web server), cli (command line), analyze (direct analysis)"',
            '    )',
            '    parser.add_argument(',
            '        "--repo",',
            '        help="Repository URL or path for analysis"',
            '    )',
            '    parser.add_argument(',
            '        "--port",',
            '        type=int,',
            '        default=8000,',
            '        help="Port for API server"',
            '    )',
            '    args = parser.parse_args()',
            '    ',
            '    if args.mode == "api":',
            '        def find_available_port(start_port=8000, max_port=8100):',
            '            """Find an available port starting from start_port"""',
            '            for port in range(start_port, max_port):',
            '                try:',
            '                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:',
            '                        s.bind(("0.0.0.0", port))',
            '                        return port',
            '                except OSError:',
            '                    continue',
            '            raise RuntimeError(f"No available ports found between {start_port} and {max_port}")',
            '        ',
            '        port = find_available_port(args.port)',
            '        print(f"🚀 Starting Unified Codebase Analytics API on http://localhost:{port}")',
            '        print(f"📚 API documentation available at http://localhost:{port}/docs")',
            '        print(f"🔍 Interactive API explorer at http://localhost:{port}/redoc")',
            '        ',
            '        uvicorn.run(fastapi_app, host="0.0.0.0", port=port)',
            '        ',
            '    elif args.mode == "cli":',
            '        if not args.repo:',
            '            print("❌ --repo is required for CLI mode")',
            '            sys.exit(1)',
            '        ',
            '        print(f"🔍 Analyzing repository: {args.repo}")',
            '        # Add CLI analysis logic here',
            '        try:',
            '            result = analyze_codebase(repo_url=args.repo)',
            '            print("✅ Analysis complete!")',
            '            print(f"📊 Results: {result}")',
            '        except Exception as e:',
            '            print(f"❌ Analysis failed: {e}")',
            '            sys.exit(1)',
            '        ',
            '    elif args.mode == "analyze":',
            '        if not args.repo:',
            '            print("❌ --repo is required for analyze mode")',
            '            sys.exit(1)',
            '        ',
            '        print(f"🔬 Direct analysis of: {args.repo}")',
            '        # Add direct analysis logic here',
            '        try:',
            '            # Use the comprehensive analyzer',
            '            analyzer = ComprehensiveAnalyzer(args.repo)',
            '            result = analyzer.analyze()',
            '            print("✅ Comprehensive analysis complete!")',
            '        except Exception as e:',
            '            print(f"❌ Analysis failed: {e}")',
            '            sys.exit(1)'
        ])
        
        # Write the output file
        with open(self.config.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            
        print(f"✅ Enhanced unified file generated: {self.config.output_file}")
        
    def consolidate(self):
        """Main consolidation process."""
        print("🔄 Starting enhanced consolidation process...")
        
        self.load_files()
        self.extract_imports()
        self.extract_key_classes()
        self.extract_key_functions()
        self.extract_fastapi_endpoints()
        self.generate_unified_file()
        
        print("✅ Enhanced consolidation complete!")
        print(f"📊 Summary:")
        print(f"  📁 Files processed: {len(self.file_contents)}")
        print(f"  📦 Imports: {len(self.imports)}")
        print(f"  🏛️ Classes: {len(self.class_definitions)}")
        print(f"  ⚙️ Functions: {len(self.function_definitions)}")
        print(f"  📍 API Endpoints: {len(self.fastapi_endpoints)}")


def main():
    """Main entry point."""
    config = ConsolidationConfig(
        input_files=[
            "backend/comprehensive_analysis.py",
            "backend/api.py",
            "backend/analyzer.py"
        ],
        output_file="backend/unified_api_enhanced.py"
    )
    
    consolidator = EnhancedConsolidator(config)
    consolidator.consolidate()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

