#!/bin/bash

# Codebase Analytics Development Startup Script
echo "🚀 Starting Codebase Analytics Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is required but not installed.${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm is required but not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"

# Setup backend
echo -e "${BLUE}Setting up backend...${NC}"
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Check if backend can start
echo -e "${YELLOW}Testing backend import...${NC}"
python3 -c "import api; print('✅ Backend imports successfully')" || {
    echo -e "${RED}❌ Backend import failed${NC}"
    exit 1
}

cd ..

# Setup frontend
echo -e "${BLUE}Setting up frontend...${NC}"
cd frontend

# Install dependencies
echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
npm install

# Test frontend build
echo -e "${YELLOW}Testing frontend build...${NC}"
npm run build || {
    echo -e "${RED}❌ Frontend build failed${NC}"
    exit 1
}

cd ..

# Create frontend environment file if it doesn't exist
if [ ! -f "frontend/.env.local" ]; then
    echo "📝 Creating frontend/.env.local..."
    cat > frontend/.env.local << EOF
# Local development environment configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development

# Optional: Add any other environment-specific variables here
# NEXT_PUBLIC_MODAL_ENDPOINT=https://your-modal-app.modal.run
EOF
    echo "✅ Created frontend/.env.local"
fi

# Function to start backend
start_backend() {
    echo -e "${GREEN}🔧 Starting backend server on http://localhost:8000${NC}"
    cd backend
    source venv/bin/activate
    python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
}

# Function to start frontend
start_frontend() {
    echo -e "${GREEN}🎨 Starting frontend server on http://localhost:3000${NC}"
    cd frontend
    npm run dev
}

# Check if we should start both services
if [ "$1" = "backend" ]; then
    start_backend
elif [ "$1" = "frontend" ]; then
    start_frontend
else
    echo -e "${GREEN}🚀 Starting both backend and frontend servers...${NC}"
    echo -e "${YELLOW}Backend will be available at: http://localhost:8000${NC}"
    echo -e "${YELLOW}Frontend will be available at: http://localhost:3000${NC}"
    echo -e "${YELLOW}API Documentation: http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${BLUE}Press Ctrl+C to stop all servers${NC}"
    echo ""
    
    # Start backend in background
    (start_backend) &
    BACKEND_PID=$!
    
    # Wait a moment for backend to start
    sleep 3
    
    # Start frontend in background
    (start_frontend) &
    FRONTEND_PID=$!
    
    # Function to cleanup on exit
    cleanup() {
        echo -e "\n${YELLOW}Shutting down servers...${NC}"
        kill $BACKEND_PID 2>/dev/null
        kill $FRONTEND_PID 2>/dev/null
        echo -e "${GREEN}✅ Servers stopped${NC}"
        exit 0
    }
    
    # Set trap to cleanup on script exit
    trap cleanup SIGINT SIGTERM
    
    # Wait for both processes
    wait $BACKEND_PID $FRONTEND_PID
fi
