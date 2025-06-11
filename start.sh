#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Start the backend API
echo -e "${BLUE}Starting backend API...${NC}"
cd backend
python api.py &
BACKEND_PID=$!
cd ..

# Wait for the backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
  echo -e "${GREEN}✅ Backend API is running successfully!${NC}"
else
  echo -e "${RED}❌ Backend API failed to start. Check logs for errors.${NC}"
  kill $BACKEND_PID 2>/dev/null
  exit 1
fi

# Start the frontend
echo -e "${BLUE}Starting frontend...${NC}"
cd frontend
npm run dev &
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
echo -e "${BLUE}Backend API: ${NC}http://localhost:8000"
echo -e "${BLUE}API Documentation: ${NC}http://localhost:8000/docs"
echo -e "${BLUE}Frontend: ${NC}http://localhost:3000"
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Keep the script running
wait

