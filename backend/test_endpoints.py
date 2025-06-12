import requests
import json
from typing import Dict, Any

def print_json(data: Dict[str, Any]) -> None:
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def test_analyze_repo():
    """Test the analyze_repo endpoint with various repositories."""
    base_url = "http://localhost:8000"
    
    test_repos = [
        "microsoft/vscode",  # Large, well-tested codebase
        "pallets/flask",     # Medium-sized Python codebase
        "facebook/react",    # Large JavaScript codebase
    ]
    
    for repo in test_repos:
        print(f"\n🔍 Testing analysis for {repo}")
        try:
            response = requests.post(
                f"{base_url}/analyze_repo",
                json={"repo_url": repo}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Print basic metrics
                print("\n📊 Basic Metrics:")
                print(f"Files: {data['num_files']}")
                print(f"Functions: {data['num_functions']}")
                print(f"Classes: {data['num_classes']}")
                
                # Print line metrics
                print("\n📝 Line Metrics:")
                line_metrics = data['line_metrics']['total']
                print(f"LOC: {line_metrics['loc']}")
                print(f"LLOC: {line_metrics['lloc']}")
                print(f"SLOC: {line_metrics['sloc']}")
                print(f"Comments: {line_metrics['comments']}")
                print(f"Comment Density: {line_metrics['comment_density']:.2f}%")
                
                # Print complexity metrics
                print("\n🔄 Complexity Metrics:")
                print(f"Average Cyclomatic Complexity: {data['cyclomatic_complexity']['average']:.2f}")
                print(f"Average Maintainability Index: {data['maintainability_index']['average']}")
                
                # Print extended analysis
                ext = data['extended_analysis']
                
                print("\n🧪 Test Analysis:")
                test_analysis = ext['test_analysis']
                print(f"Test Functions: {test_analysis['total_test_functions']}")
                print(f"Test Classes: {test_analysis['total_test_classes']}")
                print(f"Tests per File: {test_analysis['tests_per_file']:.2f}")
                
                print("\n⚙️ Function Analysis:")
                func_analysis = ext['function_analysis']
                print(f"Most Called Function: {func_analysis['most_called_function']['name']} ({func_analysis['most_called_function']['call_count']} calls)")
                print(f"Function with Most Calls: {func_analysis['function_with_most_calls']['name']} ({func_analysis['function_with_most_calls']['calls_count']} calls)")
                print(f"Recursive Functions: {', '.join(func_analysis['recursive_functions'])}")
                
                print("\n🔍 Issue Analysis:")
                total_critical = sum(len(issues['critical']) for issues in ext['file_issues'].values())
                total_major = sum(len(issues['major']) for issues in ext['file_issues'].values())
                total_minor = sum(len(issues['minor']) for issues in ext['file_issues'].values())
                print(f"Critical Issues: {total_critical}")
                print(f"Major Issues: {total_major}")
                print(f"Minor Issues: {total_minor}")
                
                # Print repository structure summary
                print("\n📁 Repository Structure:")
                def count_files(node):
                    if isinstance(node, dict) and 'children' in node:
                        return sum(count_files(child) for child in node['children'].values())
                    return 1
                
                structure = ext['repo_structure']
                total_files = count_files(structure)
                print(f"Total Files in Structure: {total_files}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error testing {repo}: {str(e)}")
            
if __name__ == "__main__":
    test_analyze_repo()

