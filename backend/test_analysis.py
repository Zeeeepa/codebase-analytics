#!/usr/bin/env python3
"""
Test script to run analysis on the codebase-analytics repository.
"""

import requests
import time
import json

def test_cli_endpoint():
    """Test the CLI endpoint for codebase analysis."""
    
    # Start the API server (you would need to run this separately)
    # For now, let's assume the server is running on localhost:8000
    
    base_url = "http://localhost:8000"
    
    # Test the CLI endpoint
    repo_name = "codebase-analytics"
    branch = "expand1"
    
    print(f"Starting analysis for {repo_name} on branch {branch}")
    
    # Make request to CLI endpoint
    response = requests.get(f"{base_url}/endpoint/{repo_name}/", params={"branch": branch})
    
    if response.status_code == 200:
        result = response.json()
        analysis_id = result["analysis_id"]
        print(f"Analysis started with ID: {analysis_id}")
        print(f"Status URL: {base_url}{result['check_status_url']}")
        print(f"Report URL: {base_url}{result['report_url']}")
        
        # Poll for completion
        while True:
            status_response = requests.get(f"{base_url}/analysis/{analysis_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']} - {status_data['message']}")
                
                if status_data["status"] in ["completed", "failed", "completed_with_errors"]:
                    break
                    
            time.sleep(5)  # Wait 5 seconds before checking again
            
        # Get final results
        if status_data["status"] == "completed":
            print("Analysis completed successfully!")
            if "results" in status_data:
                print("Results summary:")
                results = status_data["results"]
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key not in ["detailed_analysis", "files"]:  # Skip large data
                            print(f"  {key}: {value}")
        else:
            print(f"Analysis failed or completed with errors: {status_data['message']}")
            
    else:
        print(f"Failed to start analysis: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_cli_endpoint()
