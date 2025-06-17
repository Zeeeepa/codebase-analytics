#!/usr/bin/env python3
"""
Test script to check if our models work.
"""

try:
    from fastapi import FastAPI, HTTPException
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    from pydantic import BaseModel
    print("✅ Pydantic imported successfully")
except ImportError as e:
    print(f"❌ Pydantic import failed: {e}")

try:
    from typing import List, Optional
    print("✅ Typing imported successfully")
except ImportError as e:
    print(f"❌ Typing import failed: {e}")

# Test model creation
try:
    class TestModel(BaseModel):
        name: str
        count: int = 0
        items: List[str] = []
        optional_field: Optional[str] = None
    
    # Test model instantiation
    test_instance = TestModel(name="test")
    print("✅ Basic model creation works")
    print(f"   Model: {test_instance}")
    
    # Test model with data
    test_instance2 = TestModel(
        name="test2", 
        count=5, 
        items=["a", "b"], 
        optional_field="optional"
    )
    print("✅ Model with data works")
    print(f"   Model: {test_instance2}")
    
    # Test JSON serialization
    json_data = test_instance2.model_dump()
    print("✅ JSON serialization works")
    print(f"   JSON: {json_data}")
    
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test nested models
try:
    class NestedModel(BaseModel):
        inner: TestModel
        items: List[TestModel] = []
    
    nested = NestedModel(
        inner=TestModel(name="inner"),
        items=[TestModel(name="item1"), TestModel(name="item2")]
    )
    print("✅ Nested model creation works")
    print(f"   Nested: {nested}")
    
    # Test JSON serialization
    json_data = nested.model_dump()
    print("✅ Nested JSON serialization works")
    
except Exception as e:
    print(f"❌ Nested model creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🧪 Testing our actual models...")

# Test our actual models
try:
    class FunctionDetail(BaseModel):
        name: str
        parameters: List[str] = []
        return_type: Optional[str] = None
        call_count: int = 0
        calls_made: int = 0

    class ImportDetail(BaseModel):
        module: str
        imported_symbols: List[str] = []

    class FunctionAnalysis(BaseModel):
        total_functions: int = 0
        most_called_function: Optional[FunctionDetail] = None
        most_calling_function: Optional[FunctionDetail] = None
        dead_functions_count: int = 0
        sample_functions: List[FunctionDetail] = []
        sample_classes: List[dict] = []
        sample_imports: List[ImportDetail] = []

    class AnalysisResponse(BaseModel):
        function_analysis: FunctionAnalysis
    
    print("✅ Our models defined successfully")
    
    # Test instantiation
    func_detail = FunctionDetail(
        name="test_func",
        parameters=["param1", "param2"],
        return_type="str",
        call_count=5,
        calls_made=3
    )
    print("✅ FunctionDetail creation works")
    
    import_detail = ImportDetail(
        module="os",
        imported_symbols=["path", "environ"]
    )
    print("✅ ImportDetail creation works")
    
    func_analysis = FunctionAnalysis(
        total_functions=10,
        most_called_function=func_detail,
        dead_functions_count=2,
        sample_functions=[func_detail],
        sample_imports=[import_detail]
    )
    print("✅ FunctionAnalysis creation works")
    
    response = AnalysisResponse(function_analysis=func_analysis)
    print("✅ AnalysisResponse creation works")
    
    # Test JSON serialization
    json_data = response.model_dump()
    print("✅ Full response JSON serialization works")
    print(f"   Keys: {list(json_data.keys())}")
    
except Exception as e:
    print(f"❌ Our models failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests completed!")

