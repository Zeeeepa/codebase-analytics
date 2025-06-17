#!/usr/bin/env python3
"""
Test script to test the function analysis endpoint.
"""

import requests
import json
import time
import subprocess
import sys
from threading import Thread

def start_server():
    """Start the API server in the background."""
    try:
        subprocess.run([sys.executable, "api.py"], cwd=".", check=True)
    except subprocess.CalledProcessError as e:
        print(f"Server failed to start: {e}")

def test_function_analysis():
    """Test the function analysis with a simple repository."""
    
    # Wait a bit for server to start
    time.sleep(3)
    
    # Test data
    test_data = {
        "repo_url": "https://github.com/Zeeeepa/codebase-analytics"
    }
    
    print("🧪 Testing function analysis endpoint...")
    print(f"📡 Sending request to: http://localhost:8000/analyze")
    print(f"📦 Data: {test_data}")
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze", 
            json=test_data, 
            timeout=60
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if function_analysis exists
            if "function_analysis" in result:
                func_analysis = result["function_analysis"]
                print("✅ Function analysis found in response!")
                
                print(f"📈 Total functions: {func_analysis.get('total_functions', 'N/A')}")
                print(f"🎯 Most called function: {func_analysis.get('most_called_function', 'N/A')}")
                print(f"⚡ Most calling function: {func_analysis.get('most_calling_function', 'N/A')}")
                print(f"💀 Dead functions count: {func_analysis.get('dead_functions_count', 'N/A')}")
                print(f"📦 Sample imports count: {len(func_analysis.get('sample_imports', []))}")
                
                # Print some debug info if available
                if func_analysis.get('most_called_function'):
                    mcf = func_analysis['most_called_function']
                    print(f"   Most called: {mcf.get('name', 'N/A')} (called {mcf.get('call_count', 0)} times)")
                
                if func_analysis.get('most_calling_function'):
                    mcf = func_analysis['most_calling_function']
                    print(f"   Most calling: {mcf.get('name', 'N/A')} (makes {mcf.get('calls_made', 0)} calls)")
                
                return True
            else:
                print("❌ Function analysis not found in response")
                print(f"Available keys: {list(result.keys())}")
                return False
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting function analysis test...\n")
    
    # Start server in background
    print("🔧 Starting API server...")
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Test the function analysis
    success = test_function_analysis()
    
    if success:
        print("\n🎉 Function analysis test PASSED!")
    else:
        print("\n❌ Function analysis test FAILED!")
    
    print("\n🛑 Test completed. Server will continue running...")
