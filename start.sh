#!/bin/bash

# Unified Codebase Analytics System Startup Script
# This script starts both the backend FastAPI server and the frontend Next.js application

echo "🚀 Starting Codebase Analytics System..."

# Check if required dependencies are installed
echo "🔍 Checking dependencies..."

# Check for Python
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm and try again."
    exit 1
fi

# Function to check if a port is in use
is_port_in_use() {
    if command -v lsof &> /dev/null; then
        lsof -i:"$1" &> /dev/null
        return $?
    elif command -v netstat &> /dev/null; then
        netstat -tuln | grep ":$1 " &> /dev/null
        return $?
    else
        # If neither lsof nor netstat is available, assume port is free
        return 1
    fi
}

# Find an available port
find_available_port() {
    local start_port=$1
    local max_port=$2
    local port=$start_port
    
    while [ $port -le $max_port ]; do
        if ! is_port_in_use $port; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    
    echo "No available ports found between $start_port and $max_port" >&2
    return 1
}

# Kill any processes using our ports
echo "🧹 Cleaning up existing processes..."
if is_port_in_use 8000; then
    echo "   Port 8000 is in use. Attempting to free it..."
    if command -v lsof &> /dev/null; then
        kill $(lsof -t -i:8000) 2>/dev/null
    fi
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Install backend dependencies if needed
echo "📦 Installing backend dependencies..."
cd backend
pip install -q fastapi uvicorn pydantic typing-extensions &> /dev/null
cd ..

# Install frontend dependencies if needed
echo "📦 Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "   Installing npm packages (this may take a moment)..."
    npm install --silent --legacy-peer-deps &> /dev/null
fi
cd ..

# Find available ports
BACKEND_PORT=$(find_available_port 8000 8100)
if [ $? -ne 0 ]; then
    echo "❌ $BACKEND_PORT"
    exit 1
fi

# Start the backend server using the unified entry point
echo "🔧 Starting backend server on port $BACKEND_PORT..."
cd backend
python main.py --port $BACKEND_PORT > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "   Waiting for backend to initialize..."
sleep 5

# Check if backend started successfully
if ! curl -s http://localhost:$BACKEND_PORT/health > /dev/null; then
    echo "❌ Failed to start backend server. Check logs/backend.log for details."
    cat logs/backend.log
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Backend server started successfully on port $BACKEND_PORT"

# Start the frontend application
echo "🎨 Starting frontend application..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "   Waiting for frontend to initialize..."
sleep 10

# Determine which port the frontend is using by checking the log
FRONTEND_PORT=$(grep -o "http://localhost:[0-9]\+" logs/frontend.log | head -1 | cut -d':' -f3)
if [ -z "$FRONTEND_PORT" ]; then
    echo "❌ Failed to determine frontend port. Check logs/frontend.log for details."
    cat logs/frontend.log
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 1
fi

# Check if frontend started successfully
if ! curl -s http://localhost:$FRONTEND_PORT > /dev/null; then
    echo "❌ Failed to start frontend application. Check logs/frontend.log for details."
    cat logs/frontend.log
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "✅ Codebase Analytics System is running!"
echo "   Backend API:  http://localhost:$BACKEND_PORT"
echo "   API Docs:     http://localhost:$BACKEND_PORT/docs"
echo "   Simple API:   http://localhost:$BACKEND_PORT/simple"
echo "   Full API:     http://localhost:$BACKEND_PORT/full"
echo "   Frontend UI:  http://localhost:$FRONTEND_PORT"
echo ""
echo "📊 Open http://localhost:$FRONTEND_PORT in your browser to access the dashboard"
echo "💡 Press Ctrl+C to stop all services"
echo ""

# Function to handle script termination
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "👋 Codebase Analytics System stopped"
    exit 0
}

# Register the cleanup function for script termination
trap cleanup SIGINT SIGTERM

# Keep the script running to maintain the processes
while true; do
    sleep 1
done

