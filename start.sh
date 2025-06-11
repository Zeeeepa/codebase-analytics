#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fixed port for backend
BACKEND_PORT=8666
FRONTEND_PORT=8667

# Function to check if a port is available
check_port() {
  local port=$1
  # Try to bind to the port to see if it's available
  if (echo >/dev/tcp/localhost/$port) 2>/dev/null; then
    # If we can connect, the port is in use
    return 1
  else
    # If we can't connect, the port is available
    return 0
  fi
}

# Only find available port for frontend
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

# Check if backend port is available, if not, exit
if ! check_port $BACKEND_PORT; then
  echo -e "${RED}Port $BACKEND_PORT is already in use. Please free this port and try again.${NC}"
  exit 1
fi

# Find available port for frontend
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

# Kill any existing processes using our ports
echo -e "${BLUE}Ensuring ports are free...${NC}"
fuser -k $BACKEND_PORT/tcp 2>/dev/null
fuser -k $FRONTEND_PORT/tcp 2>/dev/null

# Start the backend API
echo -e "${BLUE}Starting backend API on port $BACKEND_PORT...${NC}"
cd backend
python api.py --port $BACKEND_PORT &
BACKEND_PID=$!
cd ..

# Wait for the backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
MAX_ATTEMPTS=10
ATTEMPT=1
BACKEND_STARTED=false

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  echo -n "."
  if curl -s http://localhost:$BACKEND_PORT/api/health > /dev/null; then
    echo ""
    echo -e "${GREEN}✅ Backend API is running successfully on port $BACKEND_PORT!${NC}"
    BACKEND_STARTED=true
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  sleep 2
done

if [ "$BACKEND_STARTED" = false ]; then
  echo ""
  echo -e "${RED}❌ Backend API failed to start. Check logs for errors.${NC}"
  kill $BACKEND_PID 2>/dev/null
  exit 1
fi

# Start the frontend
echo -e "${BLUE}Starting frontend on port $FRONTEND_PORT...${NC}"
cd frontend
# Explicitly set the PORT environment variable for Next.js
export PORT=$FRONTEND_PORT
echo -e "${YELLOW}Setting Next.js port to $FRONTEND_PORT${NC}"
# Use the PORT environment variable to set the Next.js port
PORT=$FRONTEND_PORT npm run dev &
FRONTEND_PID=$!
echo -e "${YELLOW}Frontend process started with PID: $FRONTEND_PID${NC}"
cd ..

# Wait for the frontend to start
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
MAX_ATTEMPTS=10
ATTEMPT=1
FRONTEND_STARTED=false

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  echo -n "."
  # Check if the Next.js process is still running
  if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo ""
    echo -e "${RED}❌ Frontend process died. Check logs for errors.${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
  fi
  
  # Try to connect to the frontend port
  if curl -s http://localhost:$FRONTEND_PORT > /dev/null; then
    echo ""
    echo -e "${GREEN}✅ Frontend is running successfully on port $FRONTEND_PORT!${NC}"
    FRONTEND_STARTED=true
    break
  fi
  ATTEMPT=$((ATTEMPT + 1))
  sleep 2
done

if [ "$FRONTEND_STARTED" = false ]; then
  echo ""
  echo -e "${RED}❌ Frontend failed to start on port $FRONTEND_PORT. Check logs for errors.${NC}"
  kill $BACKEND_PID 2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  exit 1
fi

echo -e "${GREEN}Both services are running!${NC}"
echo -e "${BLUE}Backend API: ${NC}http://localhost:$BACKEND_PORT"
echo -e "${BLUE}API Documentation: ${NC}http://localhost:$BACKEND_PORT/docs"
echo -e "${BLUE}Frontend: ${NC}http://localhost:$FRONTEND_PORT"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

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

# Keep the script running
wait
