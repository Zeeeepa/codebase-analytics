#!/bin/bash

# Enhanced Modal Deployment Script for Codebase Analytics
set -e

echo "🚀 Starting Enhanced Modal Deployment for Codebase Analytics"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_modal() {
    echo -e "${PURPLE}[MODAL]${NC} $1"
}

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    print_error "Modal CLI is not installed. Installing Modal..."
    pip install modal
fi

# Check if user is authenticated with Modal
if ! modal token list &> /dev/null; then
    print_warning "Not authenticated with Modal. Please run 'modal token new' first."
    print_status "Opening Modal authentication..."
    modal token new
fi

# Parse command line arguments
DEPLOY_TYPE="serve"
ENVIRONMENT="development"
FRONTEND_ONLY=false
BACKEND_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy)
            DEPLOY_TYPE="deploy"
            shift
            ;;
        --serve)
            DEPLOY_TYPE="serve"
            shift
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --frontend-only)
            FRONTEND_ONLY=true
            shift
            ;;
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --deploy            Deploy to Modal (persistent)"
            echo "  --serve             Serve with Modal (development)"
            echo "  --env ENVIRONMENT   Set environment (development|production) [default: development]"
            echo "  --frontend-only     Deploy only the frontend"
            echo "  --backend-only      Deploy only the backend"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Deployment type: $DEPLOY_TYPE"
print_status "Environment: $ENVIRONMENT"

# Deploy backend with Modal
if [ "$FRONTEND_ONLY" = false ]; then
    print_modal "Deploying backend to Modal..."
    
    cd backend
    
    if [ "$DEPLOY_TYPE" = "deploy" ]; then
        print_modal "Creating persistent Modal deployment..."
        modal deploy modal_api.py
        
        # Get the deployed URL
        MODAL_URL=$(modal app list | grep "enhanced-codebase-analytics" | awk '{print $3}' | head -1)
        if [ -z "$MODAL_URL" ]; then
            MODAL_URL="https://your-modal-app.modal.run"
        fi
        
        print_success "Backend deployed to Modal!"
        print_status "Modal URL: $MODAL_URL"
        
        # Save the URL for frontend configuration
        echo "$MODAL_URL" > ../modal_url.txt
        
    else
        print_modal "Starting Modal development server..."
        modal serve modal_api.py &
        MODAL_PID=$!
        
        # Wait for Modal to start
        sleep 10
        
        print_success "Modal development server started!"
        print_status "Modal URL: https://your-modal-app--fastapi-app-modal-dev.modal.run"
        
        # Save the development URL
        echo "https://your-modal-app--fastapi-app-modal-dev.modal.run" > ../modal_url.txt
    fi
    
    cd ..
fi

# Deploy frontend
if [ "$BACKEND_ONLY" = false ]; then
    print_status "Preparing frontend deployment..."
    
    cd frontend
    
    # Update the API URL to use Modal endpoint
    if [ -f "../modal_url.txt" ]; then
        MODAL_URL=$(cat ../modal_url.txt)
        print_status "Configuring frontend to use Modal URL: $MODAL_URL"
        
        # Update the next.config.js to proxy to Modal
        cat > next.config.js << EOF
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    appDir: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: '${MODAL_URL}/:path*',
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
EOF
    fi
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        print_status "Installing frontend dependencies..."
        npm install
    fi
    
    if [ "$ENVIRONMENT" = "production" ]; then
        print_status "Building frontend for production..."
        npm run build
        
        print_status "Starting production frontend server..."
        npm start &
        FRONTEND_PID=$!
        
        print_success "Frontend production server started!"
        print_status "Frontend URL: http://localhost:3000"
        
    else
        print_status "Starting frontend development server..."
        npm run dev &
        FRONTEND_PID=$!
        
        print_success "Frontend development server started!"
        print_status "Frontend URL: http://localhost:3000"
    fi
    
    cd ..
fi

# Create a cleanup function
cleanup() {
    print_status "Shutting down services..."
    
    if [ ! -z "$MODAL_PID" ]; then
        print_modal "Stopping Modal development server..."
        kill $MODAL_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        print_status "Stopping frontend server..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Clean up any remaining processes
    pkill -f "modal serve" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    pkill -f "npm start" 2>/dev/null || true
    
    print_success "Services stopped."
}

# Set up signal handlers
trap cleanup EXIT INT TERM

print_success "Enhanced Modal deployment completed! 🎉"

if [ "$DEPLOY_TYPE" = "deploy" ]; then
    print_modal "Persistent Modal deployment is now live!"
    if [ -f "modal_url.txt" ]; then
        MODAL_URL=$(cat modal_url.txt)
        print_status "Backend API: $MODAL_URL"
    fi
    if [ "$BACKEND_ONLY" = false ]; then
        print_status "Frontend: http://localhost:3000"
    fi
    
    print_status "Your application is now running with Modal serverless backend!"
    
else
    print_status "Development servers are running. Press Ctrl+C to stop all services."
    
    # Keep script running for development mode
    while true; do
        sleep 10
        
        # Check if Modal is still running (development mode)
        if [ ! -z "$MODAL_PID" ] && ! kill -0 $MODAL_PID 2>/dev/null; then
            print_warning "Modal development server has stopped"
            break
        fi
        
        # Check if frontend is still running
        if [ ! -z "$FRONTEND_PID" ] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
            print_warning "Frontend server has stopped"
            break
        fi
    done
fi

print_modal "Modal deployment script completed! 🚀"

