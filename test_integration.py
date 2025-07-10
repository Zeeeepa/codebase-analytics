#!/usr/bin/env python3
"""
Integration test script to verify frontend-backend communication
"""
import requests
import json
import time
import subprocess
import os
import signal
import sys

def test_api_endpoint():
    """Test the API endpoint directly"""
    print("🧪 Testing API endpoint...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
        # Test analyze endpoint with a small repo
        test_data = {"repo_url": "octocat/Hello-World"}
        response = requests.post(
            "http://localhost:8000/analyze_repo",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API analysis successful")
            print(f"   Repository: {data['repo_url']}")
            print(f"   Description: {data['description']}")
            print(f"   Files: {data['num_files']}")
            print(f"   LOC: {data['line_metrics']['total']['loc']}")
            print(f"   Maintainability: {data['maintainability_index']['average']:.1f}")
            return True
        else:
            print(f"❌ API analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_frontend_build():
    """Test that frontend builds successfully"""
    print("\n🏗️  Testing frontend build...")
    
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd="frontend",
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Frontend build successful")
            return True
        else:
            print("❌ Frontend build failed")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Frontend build test failed: {e}")
        return False

def main():
    print("🚀 Starting Integration Tests for Codebase Analytics")
    print("=" * 50)
    
    # Check if backend is running
    try:
        requests.get("http://localhost:8000/health", timeout=2)
        print("✅ Backend server is running")
    except:
        print("❌ Backend server is not running")
        print("   Please start the backend with: cd backend && python3 test_api.py")
        return False
    
    # Run tests
    api_test = test_api_endpoint()
    build_test = test_frontend_build()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   API Integration: {'✅ PASS' if api_test else '❌ FAIL'}")
    print(f"   Frontend Build:  {'✅ PASS' if build_test else '❌ FAIL'}")
    
    if api_test and build_test:
        print("\n🎉 All tests passed! The integration is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Start the backend: cd backend && python3 test_api.py")
        print("   2. Start the frontend: cd frontend && npm run dev")
        print("   3. Open http://localhost:3000 in your browser")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
