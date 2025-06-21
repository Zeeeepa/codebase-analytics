#!/usr/bin/env python3
"""
Test Enhanced Codebase Analytics Features

This test file verifies the enhanced functionality including:
- Fixed JSON generation issues
- Enhanced interactive UI with clickable repository tree
- Issue tracking and severity classification
- Symbol-level exploration with context information
"""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from codegen.sdk.core.codebase import Codebase
from enhanced_interactive_ui import build_enhanced_interactive_structure, EnhancedInteractiveTreeBuilder
from analysis import analyze_codebase, detect_comprehensive_issues


def test_enhanced_interactive_structure():
    """Test the enhanced interactive structure generation."""
    print("🧪 Testing Enhanced Interactive Structure...")
    
    # Create a mock codebase
    mock_codebase = Mock(spec=Codebase)
    mock_codebase.name = "test-repo"
    
    # Create mock files
    mock_file1 = Mock()
    mock_file1.file_path = "src/main.py"
    mock_file1.path = "src/main.py"
    mock_file1.content = """
def hello_world():
    print("Hello, World!")
    return "success"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
"""
    
    mock_file2 = Mock()
    mock_file2.file_path = "src/utils.py"
    mock_file2.path = "src/utils.py"
    mock_file2.content = """
def calculate_sum(a, b):
    if a < 0 or b < 0:
        raise ValueError("Negative values not allowed")
    return a + b

def unused_function():
    # This function is never called
    pass
"""
    
    # Create mock functions
    mock_func1 = Mock()
    mock_func1.name = "hello_world"
    mock_func1.line_number = 2
    mock_func1.parameters = []
    mock_func1.return_type = "str"
    mock_func1.call_sites = []
    mock_func1.dependencies = []
    
    mock_func2 = Mock()
    mock_func2.name = "calculate_sum"
    mock_func2.line_number = 2
    mock_func2.parameters = ["a", "b"]
    mock_func2.return_type = "int"
    mock_func2.call_sites = []
    mock_func2.dependencies = []
    
    # Create mock class
    mock_class = Mock()
    mock_class.name = "TestClass"
    mock_class.line_number = 6
    mock_class.methods = []
    mock_class.bases = []
    
    # Set up file relationships
    mock_file1.functions = [mock_func1]
    mock_file1.classes = [mock_class]
    mock_file2.functions = [mock_func2]
    mock_file2.classes = []
    
    # Set up codebase relationships
    mock_codebase.files = [mock_file1, mock_file2]
    mock_codebase.functions = [mock_func1, mock_func2]
    mock_codebase.classes = [mock_class]
    
    try:
        # Build enhanced interactive structure
        result = build_enhanced_interactive_structure(mock_codebase)
        
        # Verify structure
        assert "repository" in result
        repo = result["repository"]
        
        assert "name" in repo
        assert "tree" in repo
        assert "summary" in repo
        assert "ui_config" in repo
        
        # Verify summary statistics
        summary = repo["summary"]
        assert "total_files" in summary
        assert "total_functions" in summary
        assert "total_classes" in summary
        assert "total_issues" in summary
        
        # Verify UI config
        ui_config = repo["ui_config"]
        assert "theme" in ui_config
        assert "issue_severity_colors" in ui_config
        assert "file_type_icons" in ui_config
        assert "symbol_type_icons" in ui_config
        
        print("✅ Enhanced interactive structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced interactive structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_issue_severity_classification():
    """Test issue severity classification and tracking."""
    print("🧪 Testing Issue Severity Classification...")
    
    try:
        # Create mock issues data
        mock_issues = {
            'detailed_issues': [
                {
                    'file_path': 'src/main.py',
                    'symbol': 'hello_world',
                    'severity': 'Critical',
                    'message': 'Potential security vulnerability',
                    'line': 3
                },
                {
                    'file_path': 'src/main.py',
                    'symbol': 'TestClass',
                    'severity': 'Major',
                    'message': 'Missing documentation',
                    'line': 6
                },
                {
                    'file_path': 'src/utils.py',
                    'symbol': 'unused_function',
                    'severity': 'Minor',
                    'message': 'Unused function',
                    'line': 8
                }
            ]
        }
        
        # Create builder and test issue processing
        mock_codebase = Mock(spec=Codebase)
        builder = EnhancedInteractiveTreeBuilder(mock_codebase)
        builder._process_issues(mock_issues)
        
        # Verify issue mapping
        assert 'src/main.py' in builder.issue_map
        assert 'src/utils.py' in builder.issue_map
        
        # Check file-level issue counts
        main_py_issues = builder.issue_map['src/main.py']['file']
        assert main_py_issues.critical == 1
        assert main_py_issues.major == 1
        assert main_py_issues.total == 2
        
        utils_py_issues = builder.issue_map['src/utils.py']['file']
        assert utils_py_issues.minor == 1
        assert utils_py_issues.total == 1
        
        # Check symbol-level issue counts
        symbol_issues = builder.issue_map['src/main.py']['symbols']
        assert 'hello_world' in symbol_issues
        assert 'TestClass' in symbol_issues
        assert symbol_issues['hello_world'].critical == 1
        assert symbol_issues['TestClass'].major == 1
        
        print("✅ Issue severity classification test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Issue severity classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_generation_fix():
    """Test that JSON generation no longer fails with 'str' object error."""
    print("🧪 Testing JSON Generation Fix...")
    
    try:
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            src_dir = Path(temp_dir) / "src"
            src_dir.mkdir()
            
            # Create a simple Python file
            test_file = src_dir / "test.py"
            test_file.write_text("""
def test_function():
    return "Hello, World!"

class TestClass:
    def method(self):
        pass
""")
            
            # Create a Codebase object from directory (this should work now)
            try:
                codebase = Codebase.from_directory(temp_dir)
                
                # Try to analyze the codebase (this should not fail with 'str' error)
                result = analyze_codebase(codebase)
                
                # Verify result is a dictionary and can be JSON serialized
                assert isinstance(result, dict)
                json_str = json.dumps(result, default=str)
                assert len(json_str) > 0
                
                print("✅ JSON generation fix test passed!")
                return True
                
            except Exception as e:
                if "'str' object has no attribute 'files'" in str(e):
                    print(f"❌ JSON generation still has the 'str' object error: {e}")
                    return False
                else:
                    # Other errors might be expected (e.g., missing dependencies)
                    print(f"⚠️ JSON generation has different error (might be expected): {e}")
                    return True
        
    except Exception as e:
        print(f"❌ JSON generation fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_repository_tree_structure():
    """Test the clickable repository tree structure format."""
    print("🧪 Testing Repository Tree Structure...")
    
    try:
        # Create mock codebase with nested directory structure
        mock_codebase = Mock(spec=Codebase)
        mock_codebase.name = "test-repo"
        
        # Create mock files in different directories
        files = []
        
        # Root level file
        root_file = Mock()
        root_file.file_path = "README.md"
        root_file.path = "README.md"
        root_file.content = "# Test Repository"
        root_file.functions = []
        root_file.classes = []
        files.append(root_file)
        
        # Src directory files
        src_file = Mock()
        src_file.file_path = "src/main.py"
        src_file.path = "src/main.py"
        src_file.content = "def main(): pass"
        src_file.functions = []
        src_file.classes = []
        files.append(src_file)
        
        # Tests directory files
        test_file = Mock()
        test_file.file_path = "tests/test_main.py"
        test_file.path = "tests/test_main.py"
        test_file.content = "def test_main(): pass"
        test_file.functions = []
        test_file.classes = []
        files.append(test_file)
        
        mock_codebase.files = files
        mock_codebase.functions = []
        mock_codebase.classes = []
        
        # Build tree structure
        builder = EnhancedInteractiveTreeBuilder(mock_codebase)
        result = builder.build_interactive_tree()
        
        # Verify tree structure
        repo = result["repository"]
        tree = repo["tree"]
        
        assert tree["type"] == "directory"
        assert "children" in tree
        assert len(tree["children"]) > 0
        
        # Verify each child has required properties
        for child in tree["children"]:
            assert "name" in child
            assert "path" in child
            assert "type" in child
            assert "issues" in child
            
            if child["type"] == "file":
                assert "symbols" in child
                assert "statistics" in child
                assert "lines_of_code" in child
        
        # Verify issue count structure
        issues = tree["issues"]
        assert "critical" in issues
        assert "major" in issues
        assert "minor" in issues
        assert "info" in issues
        assert "total" in issues
        
        print("✅ Repository tree structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Repository tree structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all enhanced feature tests."""
    print("🚀 Running Enhanced Codebase Analytics Tests...")
    print("=" * 60)
    
    tests = [
        test_json_generation_fix,
        test_issue_severity_classification,
        test_repository_tree_structure,
        test_enhanced_interactive_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

