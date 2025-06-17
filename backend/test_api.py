#!/usr/bin/env python3
"""
Test script to check if the API works without the codegen SDK.
This will help us understand what's failing.
"""

import sys
import traceback

def test_imports():
    """Test if we can import all the required modules."""
    print("Testing imports...")
    
    try:
        from fastapi import FastAPI, HTTPException
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel
        print("✅ Pydantic imported successfully")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ Requests imported successfully")
    except ImportError as e:
        print(f"❌ Requests import failed: {e}")
        return False
    
    try:
        import networkx
        print("✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"❌ NetworkX import failed: {e}")
        return False
    
    try:
        import modal
        print("✅ Modal imported successfully")
    except ImportError as e:
        print(f"❌ Modal import failed: {e}")
        return False
    
    try:
        from codegen import Codebase
        print("✅ Codegen SDK imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Codegen SDK import failed: {e}")
        print("This is expected if codegen SDK is not installed")
        return False

def test_api_without_codegen():
    """Test if we can start the API without actually using codegen."""
    print("\nTesting API startup...")
    
    try:
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/health")
        def health_check():
            return {"status": "ok"}
        
        print("✅ FastAPI app created successfully")
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing codebase-analytics API dependencies\n")
    
    imports_ok = test_imports()
    api_ok = test_api_without_codegen()
    
    print(f"\n📊 Test Results:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"API: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if not imports_ok:
        print("\n💡 To fix import issues, run:")
        print("pip install fastapi uvicorn requests networkx pydantic modal")
        print("pip install codegen  # This might take a while")
    
    if imports_ok and api_ok:
        print("\n🎉 All tests passed! The API should work.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
