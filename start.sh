# Function to get Modal URL from deployment
get_modal_url_auto() {
    print_color $BLUE "🔍 Detecting Modal deployment URL..."
    
    # Start Modal in background and capture output
    cd backend
    
    # Create a temporary file to capture Modal output
    local temp_file=$(mktemp)
    
    print_color $YELLOW "🚀 Starting Modal deployment..."
    # Use unbuffered output and capture both stdout and stderr
    modal serve api.py 2>&1 | tee "$temp_file" &
    local modal_pid=$!
    
    # Wait for Modal to start and extract URL
    local max_wait=90  # Wait up to 90 seconds
    local wait_time=0
    local modal_url=""
    
    while [ $wait_time -lt $max_wait ]; do
        # Look for any Modal URL pattern with multiple strategies
        if [ -s "$temp_file" ]; then
            # Strategy 1: Look for FastAPI app URLs (most specific)
            modal_url=$(grep -o "https://[^[:space:]]*fastapi[^[:space:]]*\.modal\.run" "$temp_file" | head -1)
            
            # Strategy 2: Look for app URLs with common patterns
            if [ -z "$modal_url" ]; then
                modal_url=$(grep -o "https://[^[:space:]]*--[^[:space:]]*\.modal\.run" "$temp_file" | head -1)
            fi
            
            # Strategy 3: Look for any Modal URL if specific patterns not found
            if [ -z "$modal_url" ]; then
                modal_url=$(grep -o "https://[^[:space:]]*\.modal\.run" "$temp_file" | head -1)
            fi
            
            # Strategy 4: Look for URLs in specific Modal messages
            if [ -z "$modal_url" ]; then
                modal_url=$(grep -A 2 -B 2 "View app at\|Serving\|Running\|Available at\|App running at" "$temp_file" | grep -o "https://[^[:space:]]*\.modal\.run" | head -1)
            fi
            
            # Strategy 5: Look for URLs after "=>" or similar indicators
            if [ -z "$modal_url" ]; then
                modal_url=$(grep -o "=> https://[^[:space:]]*\.modal\.run" "$temp_file" | sed 's/=> //' | head -1)
            fi
            
            if [ -n "$modal_url" ]; then
                print_color $GREEN "✅ Modal URL detected: $modal_url"
                break
            fi
        fi
        
        sleep 3
        wait_time=$((wait_time + 3))
        print_color $CYAN "⏳ Waiting for Modal deployment... (${wait_time}s)"
        
        # Show recent output for debugging every 15 seconds
        if [ $((wait_time % 15)) -eq 0 ]; then
            print_color $YELLOW "📝 Recent Modal output:"
            tail -5 "$temp_file" | sed 's/^/   /'
        fi
    done
    
    # Clean up temp file but keep it for final debugging if needed
    if [ -z "$modal_url" ]; then
        print_color $RED "❌ Could not detect Modal URL automatically"
        print_color $YELLOW "📝 Full Modal output for debugging:"
        cat "$temp_file" | sed 's/^/   /'
        print_color $YELLOW "📝 Please provide the Modal backend URL manually:"
        print_color $CYAN "   (e.g., https://your-app--endpoint.modal.run)"
        read -p "Modal URL: " modal_url
        
        if [ -z "$modal_url" ]; then
            print_color $RED "❌ No URL provided. Using default local backend."
            modal_url="http://localhost:8000"
        fi
    fi
    
    rm -f "$temp_file"
    cd ..
    
    # Store the Modal PID for cleanup
    MODAL_PID=$modal_pid
    
    echo "$modal_url"
}
