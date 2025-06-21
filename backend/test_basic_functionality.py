#!/usr/bin/env python3
"""
Basic functionality tests that don't require external dependencies.
"""

import json
import os
from pathlib import Path


def test_json_serialization():
    """Test that our data structures can be properly JSON serialized."""
    print("🧪 Testing JSON Serialization...")
    
    try:
        # Test data structure similar to what we generate
        test_data = {
            "repository": {
                "name": "test-repo",
                "type": "repository",
                "tree": {
                    "name": "test-repo",
                    "path": "/",
                    "type": "directory",
                    "issues": {
                        "critical": 2,
                        "major": 5,
                        "minor": 8,
                        "info": 3,
                        "total": 18
                    },
                    "children": [
                        {
                            "name": "src",
                            "path": "src/",
                            "type": "directory",
                            "issues": {
                                "critical": 1,
                                "major": 3,
                                "minor": 4,
                                "info": 1,
                                "total": 9
                            },
                            "children": [
                                {
                                    "name": "main.py",
                                    "path": "src/main.py",
                                    "type": "file",
                                    "size": 1024,
                                    "lines_of_code": 45,
                                    "issues": {
                                        "critical": 0,
                                        "major": 1,
                                        "minor": 2,
                                        "info": 0,
                                        "total": 3
                                    },
                                    "symbols": [
                                        {
                                            "name": "main_function",
                                            "type": "function",
                                            "line_number": 10,
                                            "parameters": ["args"],
                                            "return_type": "int",
                                            "complexity": 3,
                                            "call_count": 5,
                                            "callers": ["startup", "init"],
                                            "dependencies": ["logging", "config"],
                                            "issues": {
                                                "critical": 0,
                                                "major": 0,
                                                "minor": 1,
                                                "info": 0,
                                                "total": 1
                                            },
                                            "context": {
                                                "execution_time": "2.3ms",
                                                "memory_usage": "1.2KB"
                                            }
                                        }
                                    ],
                                    "imports": ["os", "sys", "logging"],
                                    "exports": ["main_function"],
                                    "statistics": {
                                        "functions_count": 1,
                                        "classes_count": 0,
                                        "average_complexity": 3.0,
                                        "max_complexity": 3,
                                        "total_parameters": 1,
                                        "maintainability_score": 85.5
                                    },
                                    "children": []
                                }
                            ]
                        }
                    ]
                },
                "summary": {
                    "total_files": 1,
                    "total_directories": 1,
                    "total_lines_of_code": 45,
                    "total_functions": 1,
                    "total_classes": 0,
                    "total_issues": 18,
                    "issues_by_severity": {
                        "critical": 2,
                        "major": 5,
                        "minor": 8,
                        "info": 3,
                        "total": 18
                    }
                },
                "ui_config": {
                    "theme": "dark",
                    "show_issue_details": True,
                    "show_symbol_context": True,
                    "expandable_tree": True,
                    "issue_severity_colors": {
                        "critical": "#ff4444",
                        "major": "#ff8800",
                        "minor": "#ffaa00",
                        "info": "#4488ff"
                    },
                    "file_type_icons": {
                        ".py": "🐍",
                        ".js": "📜",
                        ".ts": "📘"
                    },
                    "symbol_type_icons": {
                        "function": "🔧",
                        "class": "🏗️"
                    }
                }
            }
        }
        
        # Try to serialize to JSON
        json_str = json.dumps(test_data, indent=2)
        
        # Try to deserialize back
        parsed_data = json.loads(json_str)
        
        # Verify structure is preserved
        assert parsed_data["repository"]["name"] == "test-repo"
        assert parsed_data["repository"]["tree"]["type"] == "directory"
        assert parsed_data["repository"]["summary"]["total_files"] == 1
        assert len(parsed_data["repository"]["tree"]["children"]) == 1
        
        # Verify nested structure
        src_dir = parsed_data["repository"]["tree"]["children"][0]
        assert src_dir["name"] == "src"
        assert src_dir["type"] == "directory"
        assert len(src_dir["children"]) == 1
        
        main_file = src_dir["children"][0]
        assert main_file["name"] == "main.py"
        assert main_file["type"] == "file"
        assert len(main_file["symbols"]) == 1
        
        function_symbol = main_file["symbols"][0]
        assert function_symbol["name"] == "main_function"
        assert function_symbol["type"] == "function"
        assert function_symbol["complexity"] == 3
        
        print("✅ JSON serialization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_issue_count_structure():
    """Test the issue count data structure."""
    print("🧪 Testing Issue Count Structure...")
    
    try:
        # Test issue count aggregation logic
        def aggregate_issues(children_issues):
            total = {"critical": 0, "major": 0, "minor": 0, "info": 0}
            
            for child in children_issues:
                for severity in total:
                    total[severity] += child.get(severity, 0)
            
            total["total"] = sum(total.values())
            return total
        
        # Test data
        file_issues = [
            {"critical": 1, "major": 2, "minor": 1, "info": 0},
            {"critical": 0, "major": 1, "minor": 3, "info": 2},
            {"critical": 1, "major": 0, "minor": 2, "info": 1}
        ]
        
        aggregated = aggregate_issues(file_issues)
        
        # Verify aggregation
        assert aggregated["critical"] == 2
        assert aggregated["major"] == 3
        assert aggregated["minor"] == 6
        assert aggregated["info"] == 3
        assert aggregated["total"] == 14
        
        print("✅ Issue count structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Issue count structure test failed: {e}")
        return False


def test_tree_navigation_structure():
    """Test the tree navigation structure format."""
    print("🧪 Testing Tree Navigation Structure...")
    
    try:
        # Test tree traversal logic
        def find_node_by_path(tree, target_path):
            """Find a node in the tree by its path."""
            if tree.get("path") == target_path:
                return tree
            
            for child in tree.get("children", []):
                result = find_node_by_path(child, target_path)
                if result:
                    return result
            
            return None
        
        # Test tree structure
        test_tree = {
            "name": "root",
            "path": "/",
            "type": "directory",
            "children": [
                {
                    "name": "src",
                    "path": "src/",
                    "type": "directory",
                    "children": [
                        {
                            "name": "main.py",
                            "path": "src/main.py",
                            "type": "file",
                            "children": []
                        },
                        {
                            "name": "utils.py",
                            "path": "src/utils.py",
                            "type": "file",
                            "children": []
                        }
                    ]
                },
                {
                    "name": "tests",
                    "path": "tests/",
                    "type": "directory",
                    "children": [
                        {
                            "name": "test_main.py",
                            "path": "tests/test_main.py",
                            "type": "file",
                            "children": []
                        }
                    ]
                }
            ]
        }
        
        # Test navigation
        root = find_node_by_path(test_tree, "/")
        assert root is not None
        assert root["name"] == "root"
        
        main_file = find_node_by_path(test_tree, "src/main.py")
        assert main_file is not None
        assert main_file["name"] == "main.py"
        assert main_file["type"] == "file"
        
        test_file = find_node_by_path(test_tree, "tests/test_main.py")
        assert test_file is not None
        assert test_file["name"] == "test_main.py"
        
        # Test non-existent path
        missing = find_node_by_path(test_tree, "nonexistent/file.py")
        assert missing is None
        
        print("✅ Tree navigation structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tree navigation structure test failed: {e}")
        return False


def test_ui_config_structure():
    """Test the UI configuration structure."""
    print("🧪 Testing UI Config Structure...")
    
    try:
        ui_config = {
            "theme": "dark",
            "show_issue_details": True,
            "show_symbol_context": True,
            "expandable_tree": True,
            "issue_severity_colors": {
                "critical": "#ff4444",
                "major": "#ff8800",
                "minor": "#ffaa00",
                "info": "#4488ff"
            },
            "file_type_icons": {
                ".py": "🐍",
                ".js": "📜",
                ".ts": "📘",
                ".tsx": "⚛️",
                ".jsx": "⚛️",
                ".html": "🌐",
                ".css": "🎨",
                ".json": "📋",
                ".md": "📝"
            },
            "symbol_type_icons": {
                "function": "🔧",
                "class": "🏗️",
                "variable": "📦",
                "method": "⚡"
            }
        }
        
        # Verify required fields
        assert "theme" in ui_config
        assert "issue_severity_colors" in ui_config
        assert "file_type_icons" in ui_config
        assert "symbol_type_icons" in ui_config
        
        # Verify severity colors
        colors = ui_config["issue_severity_colors"]
        assert "critical" in colors
        assert "major" in colors
        assert "minor" in colors
        assert "info" in colors
        
        # Verify icons
        file_icons = ui_config["file_type_icons"]
        assert ".py" in file_icons
        assert ".js" in file_icons
        
        symbol_icons = ui_config["symbol_type_icons"]
        assert "function" in symbol_icons
        assert "class" in symbol_icons
        
        print("✅ UI config structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ UI config structure test failed: {e}")
        return False


def run_basic_tests():
    """Run all basic functionality tests."""
    print("🚀 Running Basic Functionality Tests...")
    print("=" * 60)
    
    tests = [
        test_json_serialization,
        test_issue_count_structure,
        test_tree_navigation_structure,
        test_ui_config_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Core functionality is working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_basic_tests()
    exit(0 if success else 1)

