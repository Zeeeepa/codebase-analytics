#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to kill processes on script exit
cleanup() {
    print_warning "Cleaning up processes..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        print_status "Backend process terminated"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        print_status "Frontend process terminated"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

print_step "🚀 Starting Codebase Analytics"
echo "=================================================================="

# Check prerequisites
print_step "Checking prerequisites..."
if ! command_exists git; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

if ! command_exists python3; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi

if ! command_exists node; then
    print_error "Node.js is not installed. Please install Node.js first."
    exit 1
fi

if ! command_exists npm; then
    print_error "npm is not installed. Please install npm first."
    exit 1
fi

print_success "All prerequisites are available!"

# Navigate to home and clean up
print_step "Setting up project directory..."
cd ~

# Remove existing directory if it exists
if [ -d "ab" ]; then
    print_warning "Removing existing 'ab' directory..."
    rm -rf ab
fi

# Clone the project
print_step "Cloning codebase-analytics project..."
git clone https://github.com/Zeeeepa/codebase-analytics.git ab
if [ $? -ne 0 ]; then
    print_error "Failed to clone repository"
    exit 1
fi

cd ab
print_success "Repository cloned successfully!"

# Checkout treeview branch
print_step "Switching to treeview branch..."
git checkout treeview
if [ $? -ne 0 ]; then
    print_error "Failed to checkout treeview branch"
    exit 1
fi

print_success "Switched to treeview branch: $(git branch --show-current)"

# Setup backend (no virtual environment)
print_step "Setting up backend..."
cd backend

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
elif [ -f "../requirements.txt" ]; then
    pip3 install -r ../requirements.txt
else
    print_warning "No requirements.txt found. Installing common dependencies..."
    pip3 install fastapi uvicorn codegen gitpython requests pydantic networkx matplotlib
fi

print_success "Backend dependencies installed!"

# Setup frontend
print_step "Setting up frontend..."
cd ../frontend

print_status "Installing Node.js dependencies..."
npm install --silent
if [ $? -ne 0 ]; then
    print_error "Failed to install frontend dependencies"
    exit 1
fi

print_success "Frontend dependencies installed!"

# Start backend server
print_step "Starting backend server..."
cd ../backend

print_status "Starting backend on port 8001..."
if [ -f "api.py" ]; then
    python3 api.py &
    BACKEND_PID=$!
elif command_exists uvicorn; then
    uvicorn api:fastapi_app --host 0.0.0.0 --port 8001 --reload &
    BACKEND_PID=$!
else
    print_error "Cannot start backend server"
    exit 1
fi

# Wait for backend to start
sleep 3

if kill -0 $BACKEND_PID 2>/dev/null; then
    print_success "Backend server started! (PID: $BACKEND_PID)"
else
    print_error "Backend server failed to start"
    exit 1
fi

# Start frontend server
print_step "Starting frontend server..."
cd ../frontend

print_status "Starting frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

if kill -0 $FRONTEND_PID 2>/dev/null; then
    print_success "Frontend server started! (PID: $FRONTEND_PID)"
else
    print_error "Frontend server failed to start"
    exit 1
fi

# Display status
echo ""
echo "=================================================================="
print_success "🎉 CODEBASE ANALYTICS IS RUNNING!"
echo "=================================================================="
print_status "📊 Backend API: http://localhost:8001"
print_status "🌐 Frontend App: http://localhost:3000"
print_status "📂 Project Directory: $(pwd)/.."
print_status "🌿 Branch: $(git branch --show-current)"
echo ""
print_status "📋 Process Information:"
print_status "  Backend PID: $BACKEND_PID"
print_status "  Frontend PID: $FRONTEND_PID"
echo ""
print_warning "Press Ctrl+C to stop both servers"
echo "=================================================================="

# Show live logs
print_step "Showing live logs (Press Ctrl+C to stop)..."
echo ""

# Function to monitor both processes and show their output
monitor_processes() {
    while true; do
        # Check if processes are still running
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            print_error "Backend process died!"
            break
        fi
        
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            print_error "Frontend process died!"
            break
        fi
        
        sleep 5
    done
}

# Start monitoring in background
monitor_processes &
MONITOR_PID=$!

# Wait for interrupt
wait

# Cleanup
cleanup
