#!/bin/bash

# Start the backend API
echo "Starting backend API..."
cd backend
python api.py &
BACKEND_PID=$!
cd ..

# Wait for the backend to start
echo "Waiting for backend to start..."
sleep 5

# Start the frontend
echo "Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Function to handle script termination
function cleanup {
  echo "Stopping services..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit
}

# Set up trap to catch termination signals
trap cleanup SIGINT SIGTERM

echo "Both services are running!"
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop both services"

# Keep the script running
wait

