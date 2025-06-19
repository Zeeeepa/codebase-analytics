#!/usr/bin/env python3
"""
CLI Interface for Codebase Analytics

Usage:
    python cli.py analyze <owner>/<repo>
    python cli.py analyze https://github.com/<owner>/<repo>
"""

import sys
import argparse
import requests
import json
from urllib.parse import urlparse

def parse_repo_url(repo_input):
    """Parse repository input to extract owner and repo name."""
    if repo_input.startswith('https://github.com/'):
        # Parse GitHub URL
        parsed = urlparse(repo_input)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
        else:
            raise ValueError("Invalid GitHub URL format")
    elif '/' in repo_input:
        # Parse owner/repo format
        parts = repo_input.split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            raise ValueError("Invalid owner/repo format")
    else:
        raise ValueError("Invalid repository format. Use 'owner/repo' or GitHub URL")

def analyze_repository(owner, repo, api_url="http://localhost:8000"):
    """Analyze a repository using the API."""
    url = f"{api_url}/analyze/{owner}/{repo}"
    
    print(f"🔍 Analyzing repository: {owner}/{repo}")
    print(f"📡 API URL: {url}")
    print("⏳ This may take a moment...")
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        # Display results
        print("\n" + "="*70)
        print("🎉 CODEBASE ANALYTICS RESULTS")
        print("="*70)
        
        # Repository info
        repo_info = data['repository']
        print(f"📊 Repository: {repo_info['name']} ({repo_info['owner']})")
        print(f"📁 Files: {repo_info['total_files']} | ⚡ Functions: {repo_info['total_functions']} | 🏗️ Classes: {repo_info['total_classes']}")
        
        # Analysis results
        analysis = data['analysis']
        
        # Issues
        if 'comprehensive_issues' in analysis:
            issues = analysis['comprehensive_issues']
            print(f"\n🚨 ISSUES DETECTED:")
            print(f"  • Total Issues: {issues.get('total_issues', 0)}")
            
            if 'issues_by_severity' in issues:
                severity_counts = issues['issues_by_severity']
                print(f"  • High: {severity_counts.get('High', 0)}")
                print(f"  • Medium: {severity_counts.get('Medium', 0)}")
                print(f"  • Low: {severity_counts.get('Low', 0)}")
        
        # Entry points
        if 'most_important_entry_points' in analysis:
            entry_points = analysis['most_important_entry_points']
            print(f"\n🎯 ENTRY POINTS:")
            print(f"  • Top Functions by Heat: {len(entry_points.get('top_10_by_heat', []))}")
            print(f"  • Main Functions: {len(entry_points.get('main_functions', []))}")
            print(f"  • API Endpoints: {len(entry_points.get('api_endpoints', []))}")
            print(f"  • High Usage Functions: {len(entry_points.get('high_usage_functions', []))}")
        
        # Halstead metrics
        if 'halstead_metrics' in analysis:
            halstead = analysis['halstead_metrics']
            print(f"\n📊 HALSTEAD METRICS:")
            print(f"  • Functions Analyzed: {halstead.get('total_functions', 0)}")
            if 'summary' in halstead:
                summary = halstead['summary']
                print(f"  • Average Difficulty: {summary.get('avg_difficulty', 0):.2f}")
                print(f"  • Average Effort: {summary.get('avg_effort', 0):.2f}")
        
        # Code quality
        if 'code_quality' in analysis:
            quality = analysis['code_quality']
            print(f"\n🏆 CODE QUALITY:")
            print(f"  • Maintainability Index: {quality.get('maintainability_index', 0):.2f}")
            print(f"  • Comment Density: {quality.get('comment_density', 0):.2f}%")
            print(f"  • Technical Debt Ratio: {quality.get('technical_debt_ratio', 0):.2f}%")
        
        print(f"\n✅ Analysis complete!")
        print(f"🌐 View full results at: {api_url}/analyze/{owner}/{repo}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to API: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing API response: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Codebase Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py analyze Zeeeepa/codebase-analytics
  python cli.py analyze https://github.com/Zeeeepa/codebase-analytics
  python cli.py analyze --api-url http://localhost:8000 owner/repo
        """
    )
    
    parser.add_argument('command', choices=['analyze'], help='Command to execute')
    parser.add_argument('repository', help='Repository to analyze (owner/repo or GitHub URL)')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API server URL (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    try:
        owner, repo = parse_repo_url(args.repository)
        
        if args.command == 'analyze':
            success = analyze_repository(owner, repo, args.api_url)
            sys.exit(0 if success else 1)
            
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nUsage examples:")
        print("  python cli.py analyze owner/repo")
        print("  python cli.py analyze https://github.com/owner/repo")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()

