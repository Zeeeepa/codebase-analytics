#!/bin/bash

echo "🔍 Debugging Codebase Analytics API..."
echo "=================================="

# Check if backend is running
echo "1. Checking if backend is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running!"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
else
    echo "❌ Backend is NOT running!"
    echo "   Please start it with: cd backend && python api.py"
    exit 1
fi

echo ""
echo "2. Testing analyze_repo endpoint..."

# Test the analyze endpoint
echo "📡 Testing POST /analyze_repo..."
response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST http://localhost:8000/analyze_repo \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/Zeeeepa/open_codegen"}')

http_code=$(echo "$response" | tail -n1 | cut -d: -f2)
body=$(echo "$response" | head -n -1)

echo "HTTP Status: $http_code"
echo "Response Body:"
echo "$body" | jq . 2>/dev/null || echo "$body"

echo ""
echo "3. Checking CORS headers..."
curl -s -I -X OPTIONS http://localhost:8000/analyze_repo \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type"

echo ""
echo "🔍 Debug complete!"

