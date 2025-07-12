#!/usr/bin/env python3
"""
Test script for the API endpoint
"""

import sys
import json
import requests
import subprocess
import time
from datetime import datetime

def start_api_server():
    """Start the API server in background"""
    print("🚀 Starting API server...")
    
    # Start the server
    process = subprocess.Popen([
        "python", "-m", "uvicorn", "backend.api:fastapi_app", 
        "--host", "0.0.0.0", "--port", "8000"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server started successfully!")
            return process
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return None

def test_api_endpoint():
    """Test the analysis API endpoint"""
    print("\n🔍 Testing API endpoint...")
    print("=" * 60)
    
    # Test data
    test_request = {
        "repo_url": "https://github.com/Zeeeepa/codebase-analytics"
    }
    
    try:
        # Make API request
        print(f"📊 Analyzing repository: {test_request['repo_url']}")
        response = requests.post(
            "http://localhost:8000/analyze",
            json=test_request,
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API request successful!")
            print("=" * 60)
            
            # Display results
            if result.get("success", False):
                analysis = result.get("analysis_results", {})
                facts = analysis.get("repository_facts", {})
                
                print(f"📁 Total Files: {facts.get('total_files', 0)}")
                print(f"💻 Code Files: {facts.get('code_files', 0)}")
                print(f"🔧 Total Functions: {facts.get('total_functions', 0)}")
                print(f"🌐 Languages: {facts.get('languages', {})}")
                
                important_files = analysis.get("most_important_files", [])
                print(f"\n🎯 Most Important Files: {len(important_files)}")
                
                entry_points = analysis.get("entry_points", [])
                print(f"🚀 Entry Points: {len(entry_points)}")
                
                errors = analysis.get("actual_errors", [])
                print(f"🚨 Actual Errors: {len(errors)}")
                
                processing_time = result.get("processing_time", 0)
                print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
                
                # Save results
                output_file = f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {output_file}")
                
                return True
            else:
                print(f"❌ Analysis failed: {result.get('error_message', 'Unknown error')}")
                return False
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check passed: {result}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 API Test Suite")
    print("=" * 60)
    
    # Start API server
    server_process = start_api_server()
    
    if not server_process:
        print("❌ Failed to start API server")
        sys.exit(1)
    
    try:
        # Test health endpoint
        health_success = test_health_endpoint()
        
        # Test analysis endpoint
        if health_success:
            api_success = test_api_endpoint()
        else:
            api_success = False
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS:")
        print(f"   Health Endpoint: {'✅ PASS' if health_success else '❌ FAIL'}")
        print(f"   Analysis Endpoint: {'✅ PASS' if api_success else '❌ FAIL'}")
        
        if health_success and api_success:
            print("\n🎉 All API tests passed! Ready for production.")
            sys.exit(0)
        else:
            print("\n⚠️ Some API tests failed. Check the output above.")
            sys.exit(1)
            
    finally:
        # Clean up server process
        if server_process:
            print("\n🛑 Stopping API server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Server stopped.")

