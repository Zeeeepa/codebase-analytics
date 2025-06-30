#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_step "🚀 Starting Codebase Analytics Setup and Run Script"
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

# Navigate to WSL2 root and create folder
print_step "Setting up project directory..."
cd ~
if [ -d "ab" ]; then
    print_warning "Directory 'ab' already exists. Removing it..."
    rm -rf ab
fi

mkdir ab
cd ab
print_success "Created and entered directory: $(pwd)"

# Clone the project
print_step "Cloning the codebase-analytics project..."
git clone https://github.com/Zeeeepa/codebase-analytics.git
if [ $? -ne 0 ]; then
    print_error "Failed to clone repository"
    exit 1
fi

cd codebase-analytics
print_success "Repository cloned successfully!"

# Switch to treeview branch
print_step "Switching to treeview branch..."
git checkout treeview
if [ $? -ne 0 ]; then
    print_error "Failed to checkout treeview branch"
    exit 1
fi

print_success "Switched to treeview branch!"
git branch --show-current

# Create log directory
mkdir -p logs
LOG_DIR="$(pwd)/logs"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

print_status "Logs will be saved to:"
print_status "  Backend: $BACKEND_LOG"
print_status "  Frontend: $FRONTEND_LOG"

# Setup backend
print_step "Setting up backend..."
cd backend

# Check if virtual environment should be created
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        print_warning "Failed to create virtual environment, continuing without it..."
    else
        print_success "Virtual environment created!"
    fi
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated!"
fi

# Install Python dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
elif [ -f "../requirements.txt" ]; then
    print_status "Installing Python dependencies from root..."
    pip install -r ../requirements.txt
else
    print_warning "No requirements.txt found. Installing common dependencies..."
    pip install fastapi uvicorn codegen gitpython requests pydantic
    print_status "Installing visualization dependencies..."
    pip install networkx matplotlib
fi

print_success "Backend setup complete!"

# Setup frontend
print_step "Setting up frontend..."
cd ../frontend

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    print_error "Failed to install frontend dependencies"
    exit 1
fi

print_success "Frontend setup complete!"

# Start backend server
print_step "Starting backend server..."
cd ../backend

# Create a wrapper script for backend with logging
cat > run_backend.sh << 'EOF'
#!/bin/bash
echo "$(date): Starting backend server..." >> ../logs/backend.log
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Try different ways to run the API
if [ -f "api.py" ]; then
    echo "$(date): Running with python api.py..." >> ../logs/backend.log
    python api.py 2>&1 | tee -a ../logs/backend.log
elif command -v uvicorn >/dev/null 2>&1; then
    echo "$(date): Running with uvicorn..." >> ../logs/backend.log
    uvicorn api:fastapi_app --host 0.0.0.0 --port 8001 --reload 2>&1 | tee -a ../logs/backend.log
else
    echo "$(date): ERROR - Cannot start backend server" >> ../logs/backend.log
    exit 1
fi
EOF

chmod +x run_backend.sh

print_status "Starting backend server in background..."
./run_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    print_success "Backend server started successfully! (PID: $BACKEND_PID)"
    print_status "Backend logs: tail -f $BACKEND_LOG"
else
    print_error "Backend server failed to start"
    exit 1
fi

# Start frontend server
print_step "Starting frontend server..."
cd ../frontend

# Create a wrapper script for frontend with logging
cat > run_frontend.sh << 'EOF'
#!/bin/bash
echo "$(date): Starting frontend server..." >> ../logs/frontend.log
npm run dev 2>&1 | tee -a ../logs/frontend.log
EOF

chmod +x run_frontend.sh

print_status "Starting frontend server in background..."
./run_frontend.sh &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

# Check if frontend is running
if kill -0 $FRONTEND_PID 2>/dev/null; then
    print_success "Frontend server started successfully! (PID: $FRONTEND_PID)"
    print_status "Frontend logs: tail -f $FRONTEND_LOG"
else
    print_error "Frontend server failed to start"
    exit 1
fi

# Display status and URLs
echo ""
echo "=================================================================="
print_success "🎉 SETUP COMPLETE! Both servers are running!"
echo "=================================================================="
print_status "📊 Backend API: http://localhost:8001"
print_status "🌐 Frontend App: http://localhost:3000"
print_status "📂 Project Directory: $(pwd)/.."
print_status "🌿 Current Branch: $(git branch --show-current)"
echo ""
print_status "📋 Process Information:"
print_status "  Backend PID: $BACKEND_PID"
print_status "  Frontend PID: $FRONTEND_PID"
echo ""
print_status "📝 Live Logs:"
print_status "  Backend: tail -f $BACKEND_LOG"
print_status "  Frontend: tail -f $FRONTEND_LOG"
echo ""
print_warning "Press Ctrl+C to stop both servers and exit"
echo "=================================================================="

# Function to show live logs
show_logs() {
    echo ""
    print_step "Showing live logs (Ctrl+C to stop log viewing)..."
    echo ""
    
    # Create a function to show logs with prefixes
    (
        tail -f "$BACKEND_LOG" | sed 's/^/[BACKEND] /' &
        TAIL_BACKEND_PID=$!
        
        tail -f "$FRONTEND_LOG" | sed 's/^/[FRONTEND] /' &
        TAIL_FRONTEND_PID=$!
        
        # Wait for interrupt
        wait
        
        # Clean up tail processes
        kill $TAIL_BACKEND_PID $TAIL_FRONTEND_PID 2>/dev/null
    )
}

# Show initial log content
echo ""
print_step "Recent backend logs:"
if [ -f "$BACKEND_LOG" ]; then
    tail -n 10 "$BACKEND_LOG" | sed 's/^/  /'
else
    print_warning "Backend log file not found yet"
fi

echo ""
print_step "Recent frontend logs:"
if [ -f "$FRONTEND_LOG" ]; then
    tail -n 10 "$FRONTEND_LOG" | sed 's/^/  /'
else
    print_warning "Frontend log file not found yet"
fi

# Menu for user interaction
while true; do
    echo ""
    echo "=================================================================="
    print_status "Choose an option:"
    echo "  1) Show live logs"
    echo "  2) Check server status"
    echo "  3) Restart backend"
    echo "  4) Restart frontend"
    echo "  5) Open URLs in browser (if available)"
    echo "  6) Show recent logs"
    echo "  q) Quit and stop servers"
    echo "=================================================================="
    
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            show_logs
            ;;
        2)
            print_step "Checking server status..."
            if kill -0 $BACKEND_PID 2>/dev/null; then
                print_success "✅ Backend server is running (PID: $BACKEND_PID)"
            else
                print_error "❌ Backend server is not running"
            fi
            
            if kill -0 $FRONTEND_PID 2>/dev/null; then
                print_success "✅ Frontend server is running (PID: $FRONTEND_PID)"
            else
                print_error "❌ Frontend server is not running"
            fi
            ;;
        3)
            print_step "Restarting backend server..."
            if [ ! -z "$BACKEND_PID" ]; then
                kill $BACKEND_PID 2>/dev/null
                sleep 2
            fi
            cd ../backend
            ./run_backend.sh &
            BACKEND_PID=$!
            sleep 3
            if kill -0 $BACKEND_PID 2>/dev/null; then
                print_success "Backend server restarted! (PID: $BACKEND_PID)"
            else
                print_error "Failed to restart backend server"
            fi
            cd ../frontend
            ;;
        4)
            print_step "Restarting frontend server..."
            if [ ! -z "$FRONTEND_PID" ]; then
                kill $FRONTEND_PID 2>/dev/null
                sleep 2
            fi
            ./run_frontend.sh &
            FRONTEND_PID=$!
            sleep 5
            if kill -0 $FRONTEND_PID 2>/dev/null; then
                print_success "Frontend server restarted! (PID: $FRONTEND_PID)"
            else
                print_error "Failed to restart frontend server"
            fi
            ;;
        5)
            print_step "Attempting to open URLs in browser..."
            if command_exists xdg-open; then
                xdg-open http://localhost:3000 2>/dev/null &
                xdg-open http://localhost:8001 2>/dev/null &
                print_success "URLs opened in default browser"
            elif command_exists wslview; then
                wslview http://localhost:3000 2>/dev/null &
                wslview http://localhost:8001 2>/dev/null &
                print_success "URLs opened in Windows browser"
            else
                print_warning "No browser opener found. Please manually open:"
                print_status "  Frontend: http://localhost:3000"
                print_status "  Backend: http://localhost:8001"
            fi
            ;;
        6)
            print_step "Recent backend logs:"
            if [ -f "$BACKEND_LOG" ]; then
                tail -n 20 "$BACKEND_LOG" | sed 's/^/  /'
            else
                print_warning "Backend log file not found"
            fi
            
            echo ""
            print_step "Recent frontend logs:"
            if [ -f "$FRONTEND_LOG" ]; then
                tail -n 20 "$FRONTEND_LOG" | sed 's/^/  /'
            else
                print_warning "Frontend log file not found"
            fi
            ;;
        q|Q)
            print_step "Shutting down servers..."
            cleanup
            ;;
        *)
            print_warning "Invalid choice. Please try again."
            ;;
    esac
done
