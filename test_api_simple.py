#!/usr/bin/env python3
"""
Simple test for the API functionality without starting a server
"""

import sys
import json
from datetime import datetime

# Add backend to path
sys.path.append('./backend')

try:
    from backend.api import analyze_repository
    from backend.models import CodebaseAnalysisRequest
    print("✅ Successfully imported API components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_api_function():
    """Test the API function directly"""
    print("🔍 Testing API function directly...")
    print("=" * 60)
    
    # Create test request
    request = CodebaseAnalysisRequest(
        repo_url="https://github.com/Zeeeepa/codebase-analytics"
    )
    
    try:
        # Call the API function
        print(f"📊 Analyzing repository: {request.repo_url}")
        result = await analyze_repository(request)
        
        print("✅ API function call successful!")
        print("=" * 60)
        
        # Display results
        if result.success:
            analysis = result.analysis_results
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
            
            print(f"⏱️ Processing Time: {result.processing_time:.2f} seconds")
            
            # Save results summary
            summary = {
                "success": result.success,
                "repository_facts": facts,
                "important_files_count": len(important_files),
                "entry_points_count": len(entry_points),
                "errors_count": len(errors),
                "processing_time": result.processing_time,
                "features_analyzed": result.features_analyzed
            }
            
            output_file = f"api_function_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results summary saved to: {output_file}")
            
            return True
        else:
            print(f"❌ Analysis failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ API function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    
    print("🚀 API Function Test")
    print("=" * 60)
    
    # Run the async test
    success = asyncio.run(test_api_function())
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   API Function: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print("\n🎉 API function test passed! Ready for deployment.")
        sys.exit(0)
    else:
        print("\n⚠️ API function test failed. Check the output above.")
        sys.exit(1)

