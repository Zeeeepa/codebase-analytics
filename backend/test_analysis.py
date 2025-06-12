import requests
import json
from typing import Dict, Any
from pathlib import Path
import os

def print_json(data: Dict[str, Any]) -> None:
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def test_repo_analysis(repo_url: str) -> None:
    """Test repository analysis with a given repo URL."""
    print(f"\n🔍 Testing analysis for {repo_url}")
    
    try:
        # Test the main analysis endpoint
        response = requests.post(
            "http://localhost:8000/analyze",
            json={"repo_url": repo_url}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Test repository structure
            repo_structure = data['repo_structure']
            print("\n📁 Repository Structure:")
            print_structure(repo_structure)
            
            # Test symbol context
            if repo_structure['symbols']:
                symbol = repo_structure['symbols'][0]
                print(f"\n🔍 Testing symbol context for {symbol['name']}")
                response = requests.get(
                    f"http://localhost:8000/symbol/{symbol['id']}/context"
                )
                
                if response.status_code == 200:
                    symbol_data = response.json()
                    print("\n📊 Symbol Context:")
                    print_json(symbol_data)
                else:
                    print(f"❌ Error getting symbol context: {response.status_code}")
                    print(response.text)
            
            # Test metrics
            print("\n📊 Repository Metrics:")
            print(f"Files: {data['num_files']}")
            print(f"Functions: {data['num_functions']}")
            print(f"Classes: {data['num_classes']}")
            
            print("\n📝 Line Metrics:")
            line_metrics = data['line_metrics']['total']
            print(f"LOC: {line_metrics['loc']}")
            print(f"LLOC: {line_metrics['lloc']}")
            print(f"SLOC: {line_metrics['sloc']}")
            print(f"Comments: {line_metrics['comments']}")
            print(f"Comment Density: {line_metrics['comment_density']:.2f}%")
            
            print("\n🔄 Complexity Metrics:")
            print(f"Average Cyclomatic Complexity: {data['cyclomatic_complexity']['average']:.2f}")
            print(f"Average Maintainability Index: {data['maintainability_index']['average']}")
            
            # Test monthly commits
            print("\n📅 Monthly Commits:")
            for month, count in data['monthly_commits'].items():
                print(f"{month}: {count} commits")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error analyzing repository: {str(e)}")

def print_structure(node: Dict[str, Any], level: int = 0) -> None:
    """Print repository structure in a tree format."""
    indent = "  " * level
    
    # Print node name and type
    print(f"{indent}{'📁' if node['type'] == 'directory' else '📄'} {node['name']}")
    
    # Print stats if available
    if 'stats' in node:
        stats = node['stats']
        print(f"{indent}  📊 {stats['files']} files, {stats['directories']} dirs, {stats['symbols']} symbols")
    
    # Print issues if available
    if 'issues' in node and any(node['issues'].values()):
        issues = node['issues']
        if issues['critical'] > 0:
            print(f"{indent}  ⚠️ {issues['critical']} critical issues")
        if issues['major'] > 0:
            print(f"{indent}  ⚡ {issues['major']} major issues")
        if issues['minor'] > 0:
            print(f"{indent}  ℹ️ {issues['minor']} minor issues")
    
    # Print symbols if available
    if 'symbols' in node and node['symbols']:
        print(f"{indent}  🔧 Symbols:")
        for symbol in node['symbols']:
            print(f"{indent}    - {symbol['name']} ({symbol['type']}) @ L{symbol['start_line']}-{symbol['end_line']}")
            if 'issues' in symbol and symbol['issues']:
                for issue in symbol['issues']:
                    print(f"{indent}      ⚠️ {issue['message']}")
    
    # Recursively print children
    if 'children' in node and node['children']:
        for child in sorted(node['children'].values(), key=lambda x: (x['type'] != 'directory', x['name'])):
            print_structure(child, level + 1)

def main():
    """Test repository analysis with various repositories."""
    test_repos = [
        "Zeeeepa/codebase-analytics",  # Our own repository
        "microsoft/vscode",            # Large, well-tested codebase
        "pallets/flask",              # Medium-sized Python codebase
        "facebook/react"              # Large JavaScript codebase
    ]
    
    for repo in test_repos:
        test_repo_analysis(repo)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()

