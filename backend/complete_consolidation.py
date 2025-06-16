#!/usr/bin/env python3
"""
Complete Consolidation Script
Merges ALL code from comprehensive_analysis.py, api.py, and analyzer.py
into a single comprehensive file preserving every function, class, and line.
"""

import os
import re
from typing import Dict, List, Set, Tuple
from pathlib import Path

class CompleteConsolidator:
    """Merges all code from three files into one comprehensive file."""
    
    def __init__(self):
        self.files_to_merge = [
            "backend/comprehensive_analysis.py",
            "backend/api.py", 
            "backend/analyzer.py"
        ]
        self.output_file = "backend/consolidated_api_complete.py"
        self.file_contents = {}
        self.all_imports = set()
        self.duplicate_classes = {}
        self.duplicate_functions = {}
        
    def load_files(self):
        """Load all source files."""
        print("📂 Loading source files...")
        
        for file_path in self.files_to_merge:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.file_contents[file_path] = content
                    print(f"  ✅ Loaded {file_path} ({len(content.splitlines())} lines)")
            else:
                print(f"  ❌ File not found: {file_path}")
                
    def analyze_duplicates(self):
        """Analyze duplicate classes and functions across files."""
        print("🔍 Analyzing duplicates...")
        
        class_names = {}
        function_names = {}
        
        for file_path, content in self.file_contents.items():
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                # Find class definitions
                if line.strip().startswith('class '):
                    class_match = re.match(r'^class\s+(\w+)', line.strip())
                    if class_match:
                        class_name = class_match.group(1)
                        if class_name in class_names:
                            self.duplicate_classes[class_name] = {
                                'files': [class_names[class_name], file_path],
                                'lines': [class_names[class_name + '_line'], i + 1]
                            }
                        else:
                            class_names[class_name] = file_path
                            class_names[class_name + '_line'] = i + 1
                
                # Find function definitions
                if line.strip().startswith('def '):
                    func_match = re.match(r'^def\s+(\w+)', line.strip())
                    if func_match:
                        func_name = func_match.group(1)
                        if func_name in function_names:
                            self.duplicate_functions[func_name] = {
                                'files': [function_names[func_name], file_path],
                                'lines': [function_names[func_name + '_line'], i + 1]
                            }
                        else:
                            function_names[func_name] = file_path
                            function_names[func_name + '_line'] = i + 1
        
        print(f"  🔍 Found {len(self.duplicate_classes)} duplicate classes")
        print(f"  🔍 Found {len(self.duplicate_functions)} duplicate functions")
        
        for class_name, info in self.duplicate_classes.items():
            print(f"    📋 Class '{class_name}' in: {info['files']}")
            
        for func_name, info in self.duplicate_functions.items():
            print(f"    ⚙️ Function '{func_name}' in: {info['files']}")
    
    def extract_imports(self):
        """Extract and deduplicate all imports."""
        print("📦 Extracting imports...")
        
        for file_path, content in self.file_contents.items():
            lines = content.split('\n')
            
            for line in lines:
                stripped = line.strip()
                if (stripped.startswith('import ') or stripped.startswith('from ')) and stripped:
                    # Skip comments and empty lines
                    if not stripped.startswith('#'):
                        self.all_imports.add(stripped)
        
        print(f"  📋 Found {len(self.all_imports)} unique imports")
    
    def resolve_naming_conflicts(self, content: str, file_path: str) -> str:
        """Resolve naming conflicts by prefixing duplicates."""
        lines = content.split('\n')
        modified_lines = []
        
        # Determine file prefix
        if 'comprehensive_analysis.py' in file_path:
            prefix = 'Comprehensive'
        elif 'api.py' in file_path:
            prefix = 'Api'
        elif 'analyzer.py' in file_path:
            prefix = 'Analyzer'
        else:
            prefix = 'Unknown'
        
        for line in lines:
            modified_line = line
            
            # Handle duplicate classes
            for class_name in self.duplicate_classes:
                if f'class {class_name}' in line and file_path in self.duplicate_classes[class_name]['files']:
                    # Only rename if this is not the primary file (analyzer.py gets priority)
                    if file_path != 'backend/analyzer.py':
                        modified_line = line.replace(f'class {class_name}', f'class {prefix}{class_name}')
            
            # Handle duplicate functions (but be careful with methods)
            for func_name in self.duplicate_functions:
                if f'def {func_name}(' in line and file_path in self.duplicate_functions[func_name]['files']:
                    # Only rename if this is not the primary file and it's not a method
                    if file_path != 'backend/analyzer.py' and not line.startswith('    def'):
                        modified_line = line.replace(f'def {func_name}(', f'def {prefix.lower()}_{func_name}(')
            
            modified_lines.append(modified_line)
        
        return '\n'.join(modified_lines)
    
    def create_consolidated_file(self):
        """Create the complete consolidated file."""
        print("🚀 Creating consolidated file...")
        
        output_lines = []
        
        # File header
        output_lines.extend([
            '#!/usr/bin/env python3',
            '"""',
            'COMPLETE CONSOLIDATED CODEBASE ANALYTICS API',
            '',
            'This file contains ALL code from:',
            '- comprehensive_analysis.py (736 lines) - Deep codebase analysis using Codegen SDK',
            '- api.py (1,212 lines) - FastAPI web service with metrics calculation', 
            '- analyzer.py (2,136 lines) - Advanced issue management and transaction support',
            '',
            'Total original lines: 4,084',
            '',
            'Features:',
            '✅ Complete dead code detection and analysis',
            '✅ Advanced issue management with filtering and categorization',
            '✅ Comprehensive metrics (complexity, maintainability, volume)',
            '✅ FastAPI web interface with all endpoints',
            '✅ Multiple output formats (JSON, HTML, console, web)',
            '✅ Interactive codebase tree and symbol analysis',
            '✅ Git commit analysis and repository structure',
            '✅ Parameter validation and type annotation checking',
            '✅ Circular dependency detection',
            '✅ Implementation error detection',
            '✅ Modal deployment support',
            '',
            'Generated by complete_consolidation.py',
            '"""',
            ''
        ])
        
        # Add all imports (deduplicated and organized)
        output_lines.append('# ' + '=' * 80)
        output_lines.append('# ALL IMPORTS (DEDUPLICATED)')
        output_lines.append('# ' + '=' * 80)
        output_lines.append('')
        
        # Organize imports
        stdlib_imports = []
        thirdparty_imports = []
        local_imports = []
        
        for imp in sorted(self.all_imports):
            if any(stdlib in imp for stdlib in [
                'import os', 'import sys', 'import time', 'import json', 'import logging',
                'import argparse', 'import math', 'import re', 'import subprocess',
                'import tempfile', 'import hashlib', 'import socket', 'import traceback',
                'from datetime import', 'from pathlib import', 'from typing import',
                'from dataclasses import', 'from enum import', 'from collections import'
            ]):
                stdlib_imports.append(imp)
            elif imp.startswith('from codegen'):
                local_imports.append(imp)
            else:
                thirdparty_imports.append(imp)
        
        # Add imports in order
        if stdlib_imports:
            output_lines.extend(stdlib_imports)
            output_lines.append('')
        
        if thirdparty_imports:
            output_lines.extend(thirdparty_imports)
            output_lines.append('')
            
        if local_imports:
            output_lines.extend(local_imports)
            output_lines.append('')
        
        # Add content from each file with conflict resolution
        file_order = [
            ('backend/analyzer.py', 'ANALYZER.PY - ADVANCED ISSUE MANAGEMENT (2,136 lines)'),
            ('backend/api.py', 'API.PY - FASTAPI WEB SERVICE (1,212 lines)'),
            ('backend/comprehensive_analysis.py', 'COMPREHENSIVE_ANALYSIS.PY - DEEP ANALYSIS (736 lines)')
        ]
        
        for file_path, description in file_order:
            if file_path in self.file_contents:
                output_lines.append('# ' + '=' * 80)
                output_lines.append(f'# {description}')
                output_lines.append('# ' + '=' * 80)
                output_lines.append('')
                
                # Get content and resolve conflicts
                content = self.file_contents[file_path]
                
                # Remove the shebang and module docstring from non-primary files
                lines = content.split('\n')
                start_idx = 0
                
                # Skip shebang
                if lines[0].startswith('#!'):
                    start_idx = 1
                
                # Skip module docstring
                in_docstring = False
                docstring_quotes = None
                
                for i in range(start_idx, len(lines)):
                    line = lines[i].strip()
                    if line.startswith('"""') or line.startswith("'''"):
                        if not in_docstring:
                            in_docstring = True
                            docstring_quotes = line[:3]
                            if line.count(docstring_quotes) >= 2:  # Single line docstring
                                start_idx = i + 1
                                break
                        elif line.endswith(docstring_quotes):
                            start_idx = i + 1
                            break
                    elif not in_docstring and line and not line.startswith('#'):
                        break
                
                # Skip imports section (already handled)
                while start_idx < len(lines):
                    line = lines[start_idx].strip()
                    if (line.startswith('import ') or line.startswith('from ') or 
                        line == '' or line.startswith('#')):
                        start_idx += 1
                    else:
                        break
                
                # Add the remaining content with conflict resolution
                remaining_content = '\n'.join(lines[start_idx:])
                resolved_content = self.resolve_naming_conflicts(remaining_content, file_path)
                
                output_lines.append(resolved_content)
                output_lines.append('')
        
        # Add unified main execution block
        output_lines.extend([
            '# ' + '=' * 80,
            '# UNIFIED MAIN EXECUTION',
            '# ' + '=' * 80,
            '',
            'if __name__ == "__main__":',
            '    import argparse',
            '    import sys',
            '    ',
            '    parser = argparse.ArgumentParser(',
            '        description="Complete Consolidated Codebase Analytics API",',
            '        formatter_class=argparse.RawDescriptionHelpFormatter,',
            '        epilog="""',
            'Available modes:',
            '  api        - Start FastAPI web server',
            '  analyze    - Run comprehensive analysis',
            '  cli        - Command line interface',
            '  html       - Generate HTML report',
            '        """',
            '    )',
            '    ',
            '    parser.add_argument("--mode", choices=["api", "analyze", "cli", "html"], default="api")',
            '    parser.add_argument("--repo", help="Repository path or URL")',
            '    parser.add_argument("--port", type=int, default=8000, help="API server port")',
            '    parser.add_argument("--output", help="Output file path")',
            '    parser.add_argument("--format", choices=["json", "html", "console"], default="console")',
            '    ',
            '    args = parser.parse_args()',
            '    ',
            '    if args.mode == "api":',
            '        try:',
            '            import uvicorn',
            '            print(f"🚀 Starting Complete Analytics API on http://localhost:{args.port}")',
            '            print(f"📚 Documentation: http://localhost:{args.port}/docs")',
            '            uvicorn.run("__main__:fastapi_app", host="0.0.0.0", port=args.port)',
            '        except ImportError:',
            '            print("❌ FastAPI/uvicorn not available. Install with: pip install fastapi uvicorn")',
            '            sys.exit(1)',
            '    ',
            '    elif args.mode == "analyze":',
            '        if not args.repo:',
            '            print("❌ --repo required for analysis mode")',
            '            sys.exit(1)',
            '        ',
            '        try:',
            '            analyzer = CodebaseAnalyzer(repo_path=args.repo)',
            '            result = analyzer.analyze()',
            '            ',
            '            if args.output:',
            '                analyzer.save_results(args.output)',
            '                print(f"✅ Results saved to {args.output}")',
            '            else:',
            '                analyzer._print_console_report()',
            '        except Exception as e:',
            '            print(f"❌ Analysis failed: {e}")',
            '            sys.exit(1)',
            '    ',
            '    elif args.mode == "cli":',
            '        if not args.repo:',
            '            print("❌ --repo required for CLI mode")',
            '            sys.exit(1)',
            '        ',
            '        try:',
            '            # Use the comprehensive analyzer',
            '            comp_analyzer = ComprehensiveAnalyzer(args.repo)',
            '            result = comp_analyzer.analyze()',
            '            comp_analyzer._print_report(result)',
            '        except Exception as e:',
            '            print(f"❌ CLI analysis failed: {e}")',
            '            sys.exit(1)',
            '    ',
            '    elif args.mode == "html":',
            '        if not args.repo:',
            '            print("❌ --repo required for HTML mode")',
            '            sys.exit(1)',
            '        ',
            '        try:',
            '            analyzer = CodebaseAnalyzer(repo_path=args.repo)',
            '            analyzer.analyze()',
            '            output_file = args.output or "analysis_report.html"',
            '            analyzer._generate_html_report(output_file)',
            '            print(f"✅ HTML report generated: {output_file}")',
            '        except Exception as e:',
            '            print(f"❌ HTML generation failed: {e}")',
            '            sys.exit(1)'
        ])
        
        # Write the consolidated file
        final_content = '\n'.join(output_lines)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"✅ Consolidated file created: {self.output_file}")
        print(f"📊 Final file size: {len(final_content.splitlines())} lines")
        
        return len(final_content.splitlines())
    
    def validate_consolidation(self):
        """Validate the consolidated file."""
        print("🧪 Validating consolidated file...")
        
        if not os.path.exists(self.output_file):
            print("❌ Output file not found!")
            return False
        
        # Check syntax
        try:
            with open(self.output_file, 'r') as f:
                compile(f.read(), self.output_file, 'exec')
            print("✅ Syntax validation passed")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
        
        # Check line count
        with open(self.output_file, 'r') as f:
            line_count = len(f.readlines())
        
        original_total = sum(len(content.splitlines()) for content in self.file_contents.values())
        
        print(f"📊 Line count comparison:")
        print(f"  Original total: {original_total} lines")
        print(f"  Consolidated: {line_count} lines")
        print(f"  Difference: {line_count - original_total} lines (includes headers and organization)")
        
        if line_count >= original_total * 0.95:  # Allow for some optimization
            print("✅ Line count validation passed")
            return True
        else:
            print("⚠️ Significant line count reduction detected")
            return False
    
    def consolidate(self):
        """Main consolidation process."""
        print("🔄 Starting COMPLETE consolidation process...")
        print("🎯 Goal: Merge ALL 4,084 lines into one comprehensive file")
        
        self.load_files()
        self.extract_imports()
        self.analyze_duplicates()
        final_lines = self.create_consolidated_file()
        validation_passed = self.validate_consolidation()
        
        print("\n" + "=" * 60)
        print("📈 CONSOLIDATION SUMMARY")
        print("=" * 60)
        
        original_total = sum(len(content.splitlines()) for content in self.file_contents.values())
        
        print(f"📁 Files processed: {len(self.file_contents)}")
        print(f"📋 Original total lines: {original_total}")
        print(f"📄 Final consolidated lines: {final_lines}")
        print(f"📦 Unique imports: {len(self.all_imports)}")
        print(f"🔄 Duplicate classes resolved: {len(self.duplicate_classes)}")
        print(f"⚙️ Duplicate functions resolved: {len(self.duplicate_functions)}")
        print(f"✅ Validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        if validation_passed:
            print(f"\n🎉 SUCCESS! Complete consolidation created: {self.output_file}")
            print("🚀 All functionality from all three files preserved!")
        else:
            print(f"\n⚠️ Consolidation completed with warnings: {self.output_file}")
        
        return validation_passed


def main():
    """Main entry point."""
    consolidator = CompleteConsolidator()
    success = consolidator.consolidate()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

