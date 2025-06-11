#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default ports
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Function to check if a port is available
check_port() {
  local port=$1
  if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
    return 1  # Port is in use
  else
    return 0  # Port is available
  fi
}

# Function to find an available port starting from a base port
find_available_port() {
  local base_port=$1
  local port=$base_port
  
  while ! check_port $port; do
    echo -e "${YELLOW}Port $port is already in use, trying next port...${NC}"
    port=$((port + 1))
    if [ $port -gt $((base_port + 20)) ]; then
      echo -e "${RED}Could not find an available port in range $base_port-$((base_port + 20))${NC}"
      return 1
    fi
  done
  
  echo $port
}

# Find available ports
BACKEND_PORT=$(find_available_port $BACKEND_PORT)
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to find available port for backend. Exiting.${NC}"
  exit 1
fi

FRONTEND_PORT=$(find_available_port $FRONTEND_PORT)
if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to find available port for frontend. Exiting.${NC}"
  exit 1
fi

echo -e "${GREEN}Using ports: Backend=$BACKEND_PORT, Frontend=$FRONTEND_PORT${NC}"

# Create a temporary .env file for the frontend to use the correct backend port
echo -e "${BLUE}Configuring frontend to use backend on port $BACKEND_PORT...${NC}"
if [ ! -d "frontend" ]; then
  echo -e "${RED}Frontend directory not found. Make sure you're running this script from the project root.${NC}"
  exit 1
fi

# Create or update .env.local file
cat > frontend/.env.local << EOF
# Local development environment configuration
NEXT_PUBLIC_API_URL=http://localhost:$BACKEND_PORT
NODE_ENV=development
EOF

# Start the backend API
echo -e "${BLUE}Starting backend API on port $BACKEND_PORT...${NC}"
cd backend
python api.py --port $BACKEND_PORT &
BACKEND_PID=$!
cd ..

# Wait for the backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 5

# Check if backend is running
if curl -s http://localhost:$BACKEND_PORT/health > /dev/null; then
  echo -e "${GREEN}✅ Backend API is running successfully on port $BACKEND_PORT!${NC}"
else
  echo -e "${RED}❌ Backend API failed to start. Check logs for errors.${NC}"
  echo -e "${YELLOW}Attempting to start backend in fallback mode...${NC}"
  
  # Kill the previous attempt if it's still running
  kill $BACKEND_PID 2>/dev/null
  
  # Try starting with explicit parameters
  cd backend
  python api.py --port $BACKEND_PORT --disable-graph-sitter &
  BACKEND_PID=$!
  cd ..
  
  sleep 5
  
  # Check again
  if curl -s http://localhost:$BACKEND_PORT/health > /dev/null; then
    echo -e "${GREEN}✅ Backend API is running in fallback mode on port $BACKEND_PORT!${NC}"
  else
    echo -e "${RED}❌ Backend API failed to start even in fallback mode. Exiting.${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
  fi
fi

# Start the frontend
echo -e "${BLUE}Starting frontend on port $FRONTEND_PORT...${NC}"
cd frontend
PORT=$FRONTEND_PORT npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle script termination
function cleanup {
  echo -e "${YELLOW}Stopping services...${NC}"
  kill $BACKEND_PID 2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  echo -e "${GREEN}✅ All services stopped${NC}"
  exit
}

# Set up trap to catch termination signals
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}Both services are running!${NC}"
echo -e "${BLUE}Backend API: ${NC}http://localhost:$BACKEND_PORT"
echo -e "${BLUE}API Documentation: ${NC}http://localhost:$BACKEND_PORT/docs"
echo -e "${BLUE}Frontend: ${NC}http://localhost:$FRONTEND_PORT"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Keep the script running
wait

