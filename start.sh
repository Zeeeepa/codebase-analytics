#!/bin/bash

# Codebase Analytics - Start Script
# Provides multiple deployment options for backend and frontend

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to print header
print_header() {
    echo ""
    print_color $CYAN "=================================="
    print_color $CYAN "  Codebase Analytics Launcher"
    print_color $CYAN "=================================="
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()
    
    if ! command_exists "node"; then
        missing_deps+=("node")
    fi
    
    if ! command_exists "npm"; then
        missing_deps+=("npm")
    fi
    
    if ! command_exists "python3"; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists "pip"; then
        missing_deps+=("pip")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_color $RED "❌ Missing dependencies: ${missing_deps[*]}"
        print_color $YELLOW "Please install the missing dependencies and try again."
        exit 1
    fi
}

# Function to install backend dependencies
install_backend_deps() {
    print_color $BLUE "📦 Installing backend dependencies..."
    cd backend
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_color $YELLOW "⚠️  No requirements.txt found, installing common dependencies..."
        pip install fastapi uvicorn modal graph-sitter
    fi
    cd ..
    print_color $GREEN "✅ Backend dependencies installed"
}

# Function to install frontend dependencies
install_frontend_deps() {
    print_color $BLUE "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    print_color $GREEN "✅ Frontend dependencies installed"
}

# Function to start backend locally
start_backend_local() {
    print_color $BLUE "🚀 Starting backend locally..."
    cd backend
    
    # Check if uvicorn is available, fallback to python
    if command_exists "uvicorn"; then
        print_color $GREEN "🌐 Backend running at: http://localhost:8000"
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    else
        print_color $GREEN "🌐 Backend running at: http://localhost:8000"
        python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    fi
}

# Function to start backend with Modal
start_backend_modal() {
    print_color $BLUE "☁️  Starting backend with Modal..."
    
    # Check if modal is installed
    if ! command_exists "modal"; then
        print_color $YELLOW "📦 Installing Modal..."
        pip install modal
    fi
    
    cd backend
    print_color $GREEN "🌐 Starting Modal deployment..."
    print_color $CYAN "📝 Note: Modal will provide the live URL once deployed"
    modal serve api.py
}

# Function to build and start frontend
start_frontend() {
    local backend_url=${1:-"http://localhost:8000"}
    
    print_color $BLUE "🎨 Building and starting frontend..."
    cd frontend
    
    # Set backend URL environment variable
    export REACT_APP_BACKEND_URL="$backend_url"
    export VITE_BACKEND_URL="$backend_url"
    
    # Build the frontend
    print_color $BLUE "🔨 Building frontend..."
    npm run build
    
    # Start the frontend
    print_color $GREEN "🌐 Frontend will be available at: http://localhost:3000"
    print_color $CYAN "🔗 Backend URL configured as: $backend_url"
    
    # Check if we have a start script, otherwise use serve
    if npm run start >/dev/null 2>&1; then
        npm run start
    elif command_exists "serve"; then
        serve -s build -l 3000
    else
        print_color $YELLOW "📦 Installing serve to host the built frontend..."
        npm install -g serve
        serve -s build -l 3000
    fi
}

# Function to get Modal URL from deployment
get_modal_url_auto() {
    print_color $BLUE "🔍 Detecting Modal deployment URL..."
    
    # Start Modal in background and capture output
    cd backend
    
    # Create a temporary file to capture Modal output
    local temp_file=$(mktemp)
    
    print_color $YELLOW "🚀 Starting Modal deployment..."
    modal serve api.py > "$temp_file" 2>&1 &
    local modal_pid=$!
    
    # Wait for Modal to start and extract URL
    local max_wait=60  # Wait up to 60 seconds
    local wait_time=0
    local modal_url=""
    
    while [ $wait_time -lt $max_wait ]; do
        if grep -q "https://.*\.modal\.run" "$temp_file"; then
            # Extract the main app URL (not individual function URLs)
            modal_url=$(grep "https://.*\.modal\.run" "$temp_file" | grep -E "(fastapi_modal_app|main)" | head -1 | sed 's/.*\(https:\/\/[^[:space:]]*\.modal\.run\).*/\1/')
            
            if [ -n "$modal_url" ]; then
                print_color $GREEN "✅ Modal URL detected: $modal_url"
                break
            fi
        fi
        
        sleep 2
        wait_time=$((wait_time + 2))
        print_color $CYAN "⏳ Waiting for Modal deployment... (${wait_time}s)"
    done
    
    # Clean up temp file
    rm -f "$temp_file"
    cd ..
    
    if [ -z "$modal_url" ]; then
        print_color $RED "❌ Could not detect Modal URL automatically"
        print_color $YELLOW "📝 Please provide the Modal backend URL manually:"
        print_color $CYAN "   (e.g., https://your-app--endpoint.modal.run)"
        read -p "Modal URL: " modal_url
        
        if [ -z "$modal_url" ]; then
            print_color $RED "❌ No URL provided. Using default local backend."
            modal_url="http://localhost:8000"
        fi
    fi
    
    # Store the Modal PID for cleanup
    MODAL_PID=$modal_pid
    
    echo "$modal_url"
}

# Function to get Modal URL from user (fallback)
get_modal_url_manual() {
# Function to get Modal URL from user

    echo ""
    print_color $YELLOW "📝 Please provide the Modal backend URL:"
    print_color $CYAN "   (e.g., https://your-app--endpoint.modal.run)"
    read -p "Modal URL: " modal_url
    
    if [ -z "$modal_url" ]; then
        print_color $RED "❌ No URL provided. Using default local backend."
        echo "http://localhost:8000"
    else
        echo "$modal_url"
    fi
}

# Function to run option in background and wait
run_in_background() {
    local option=$1
    local backend_url=$2
    
    case $option in
        4)
            # Backend local + Frontend
            print_color $BLUE "🔄 Starting backend locally in background..."
            start_backend_local &
            BACKEND_PID=$!
            
            # Wait a moment for backend to start
            sleep 5
            
            print_color $BLUE "🔄 Starting frontend..."
            start_frontend "http://localhost:8000"
            ;;
        5)
            # Backend modal + Frontend
            print_color $BLUE "🔄 Starting Modal backend and frontend..."
            print_color $CYAN "🤖 Automatically detecting Modal URL..."
            
            modal_url=$(get_modal_url_auto)

            start_frontend "$modal_url"
            ;;
    esac
}

# Function to cleanup background processes
cleanup() {
    if [ ! -z "$BACKEND_PID" ]; then
        print_color $YELLOW "🛑 Stopping background backend process..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$MODAL_PID" ]; then
        print_color $YELLOW "🛑 Stopping Modal deployment..."
        kill $MODAL_PID 2>/dev/null || true
    fi
    

    exit 0
}

# Trap cleanup on script exit
trap cleanup EXIT INT TERM

# Main menu function
show_menu() {
    print_header
    
    print_color $GREEN "Choose your deployment option:"
    echo ""
    print_color $BLUE "1) 🖥️  Backend - Local Development"
    print_color $PURPLE "2) ☁️  Backend - Modal Cloud"
    print_color $CYAN "3) 🎨 Frontend - Build and Start"
    print_color $YELLOW "4) 🔄 Backend Local + Frontend"
    print_color $GREEN "5) ���� Backend Modal + Frontend"

    echo ""
    print_color $RED "0) ❌ Exit"
    echo ""
}

# Main execution
main() {
    # Check dependencies first
    check_dependencies
    
    while true; do
        show_menu
        read -p "Enter your choice (0-5): " choice
        
        case $choice in
            1)
                print_color $GREEN "🖥️  Starting backend locally..."
                install_backend_deps
                start_backend_local
                ;;
            2)
                print_color $PURPLE "☁️  Starting backend with Modal..."
                install_backend_deps
                start_backend_modal
                ;;
            3)
                print_color $CYAN "🎨 Building and starting frontend..."
                install_frontend_deps
                start_frontend
                ;;
            4)
                print_color $YELLOW "🔄 Starting backend local + frontend..."
                install_backend_deps
                install_frontend_deps
                run_in_background 4
                ;;
            5)
                print_color $GREEN "🌐 Starting backend modal + frontend..."
                install_frontend_deps
                run_in_background 5
                ;;
            0)
                print_color $RED "👋 Goodbye!"
                exit 0
                ;;
            *)
                print_color $RED "❌ Invalid option. Please choose 0-5."
                sleep 2
                ;;
        esac
        
        # If we reach here, the process has ended, show menu again
        echo ""
        print_color $YELLOW "Press Enter to return to menu..."
        read
    done
}

# Run main function
main "$@"

