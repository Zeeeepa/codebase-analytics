import requests
import json
from typing import Dict, Any

def print_json(data: Dict[str, Any]) -> None:
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def test_repo_analysis(repo_url: str) -> None:
    """Test repository analysis with a given repo URL."""
    print(f"\n🔍 Testing analysis for {repo_url}")
    
    try:
        # Simulate the analysis that would be done by the API
        print("\n📊 Basic Repository Info:")
        print(f"URL: {repo_url}")
        
        # Test getting repository description
        api_url = f"https://api.github.com/repos/{repo_url}"
        response = requests.get(api_url)
        if response.status_code == 200:
            repo_data = response.json()
            print(f"Description: {repo_data.get('description', 'No description available')}")
            print(f"Stars: {repo_data.get('stargazers_count', 0)}")
            print(f"Forks: {repo_data.get('forks_count', 0)}")
            print(f"Open Issues: {repo_data.get('open_issues_count', 0)}")
        else:
            print(f"❌ Error fetching repository info: {response.status_code}")
        
        # Test getting commit history
        commits_url = f"https://api.github.com/repos/{repo_url}/commits"
        response = requests.get(commits_url)
        if response.status_code == 200:
            commits = response.json()
            print(f"\n📝 Recent Commits: {len(commits)}")
            for commit in commits[:3]:  # Show last 3 commits
                print(f"- {commit['commit']['message'][:50]}...")
        else:
            print(f"❌ Error fetching commit history: {response.status_code}")
        
        # Test getting repository contents
        contents_url = f"https://api.github.com/repos/{repo_url}/contents"
        response = requests.get(contents_url)
        if response.status_code == 200:
            contents = response.json()
            print(f"\n📁 Repository Contents:")
            for item in contents:
                print(f"- {item['type']}: {item['name']}")
        else:
            print(f"❌ Error fetching repository contents: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error analyzing repository: {str(e)}")

def main():
    """Test repository analysis with various repositories."""
    test_repos = [
        "microsoft/vscode",
        "pallets/flask",
        "facebook/react"
    ]
    
    for repo in test_repos:
        test_repo_analysis(repo)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()

