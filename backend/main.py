#!/usr/bin/env python3
"""
Unified entry point for the Codebase Analytics backend.
This script provides a single command to start the backend server with all necessary components.
"""

import os
import sys
import argparse
import uvicorn
import socket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the API modules
try:
    from api_simple import app as simple_app
except ImportError:
    print("Warning: api_simple.py not found or contains errors. Simple API will not be available.")
    simple_app = None

try:
    from api import fastapi_app as full_app
except ImportError:
    print("Warning: api.py not found or contains errors. Full API will not be available.")
    full_app = None

# Create a unified FastAPI application
app = FastAPI(
    title="Codebase Analytics API",
    description="Unified API for codebase analysis and visualization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Codebase Analytics API",
        "version": "1.0.0"
    }

# Mount the simple API if available
if simple_app:
    app.mount("/simple", simple_app)
    print("✅ Simple API mounted at /simple")

# Mount the full API if available
if full_app:
    app.mount("/full", full_app)
    print("✅ Full API mounted at /full")

def find_available_port(start_port=8000, max_port=8100):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

def main():
    """Main entry point for the backend server."""
    parser = argparse.ArgumentParser(description="Start the Codebase Analytics backend server")
    parser.add_argument("--port", type=int, default=None, help="Port to run the server on (default: auto-detect)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Find an available port if not specified
    port = args.port or find_available_port()

    print(f"🚀 Starting Codebase Analytics backend server on http://{args.host}:{port}")
    print(f"📚 API documentation available at http://{args.host}:{port}/docs")
    
    if simple_app:
        print(f"🔍 Simple API available at http://{args.host}:{port}/simple")
    
    if full_app:
        print(f"🔍 Full API available at http://{args.host}:{port}/full")

    # Start the server
    uvicorn.run(
        "main:app",
        host=args.host,
        port=port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()

