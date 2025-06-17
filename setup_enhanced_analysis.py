#!/usr/bin/env python3
"""
Setup Script for Enhanced Codebase Analytics

This script sets up the enhanced analysis environment, installs dependencies,
and provides utilities for testing and deployment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any

def run_command(command: str, cwd: str = None) -> tuple:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "networkx>=3.0",
        "modal>=0.55.0",
        "codegen>=0.1.0"
    ]
    
    # Development dependencies
    dev_deps = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        print(f"   Installing {dep}...")
        success, stdout, stderr = run_command(f"pip install {dep}")
        if not success:
            print(f"❌ Failed to install {dep}")
            print(f"   Error: {stderr}")
            return False
    
    # Install development dependencies
    print("   Installing development dependencies...")
    for dep in dev_deps:
        success, stdout, stderr = run_command(f"pip install {dep}")
        if not success:
            print(f"⚠️  Warning: Failed to install {dep} (development dependency)")
    
    print("✅ Dependencies installed successfully")
    return True

def setup_environment():
    """Set up the environment configuration."""
    print("🔧 Setting up environment...")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Enhanced Codebase Analytics Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=1

# Analysis Configuration
MAX_ANALYSIS_TIME=300
ENABLE_CACHING=true
CACHE_TTL=3600

# Visualization Configuration
MAX_NODES=1000
DEFAULT_LAYOUT=hierarchical

# Security Configuration
ENABLE_SECURITY_SCAN=true
SECURITY_SCAN_DEPTH=medium

# Performance Configuration
ENABLE_PARALLEL_ANALYSIS=true
MAX_MEMORY_USAGE=1GB

# Modal Configuration (for deployment)
# MODAL_TOKEN_ID=your_token_id
# MODAL_TOKEN_SECRET=your_token_secret

# GitHub API (optional, for enhanced repository information)
# GITHUB_TOKEN=your_github_token
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env configuration file")
    else:
        print("✅ Environment configuration already exists")
    
    return True

def run_tests():
    """Run the test suite."""
    print("🧪 Running test suite...")
    
    # Check if test file exists
    test_file = Path("backend/test_enhanced_analysis.py")
    if not test_file.exists():
        print("❌ Test file not found")
        return False
    
    # Run tests
    success, stdout, stderr = run_command("python backend/test_enhanced_analysis.py")
    
    if success:
        print("✅ All tests passed")
        print(stdout)
    else:
        print("❌ Some tests failed")
        print(stderr)
    
    return success

def validate_installation():
    """Validate the installation by running a simple analysis."""
    print("🔍 Validating installation...")
    
    try:
        # Try importing the modules
        from backend.advanced_analysis import perform_comprehensive_analysis, AnalysisType
        from backend.enhanced_visualizations import create_comprehensive_dashboard_data
        
        print("✅ All modules imported successfully")
        
        # Try creating a mock analysis
        from backend.test_enhanced_analysis import MockCodebase
        
        codebase = MockCodebase()
        codebase.add_file("test.py", source="def test():\n    return True")
        codebase.add_function("test")
        
        # Run a simple analysis
        results = perform_comprehensive_analysis(codebase, [AnalysisType.CODE_QUALITY])
        
        if 'code_quality_metrics' in results:
            print("✅ Analysis functionality working")
        else:
            print("❌ Analysis functionality not working")
            return False
        
        # Try creating visualizations
        dashboard = create_comprehensive_dashboard_data(codebase)
        
        if 'metadata' in dashboard:
            print("✅ Visualization functionality working")
        else:
            print("❌ Visualization functionality not working")
            return False
        
        print("✅ Installation validation successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def create_sample_config():
    """Create sample configuration files."""
    print("📝 Creating sample configuration files...")
    
    # Sample analysis configuration
    analysis_config = {
        "default_analysis_types": [
            "dependency",
            "call_graph",
            "code_quality",
            "architectural"
        ],
        "security_analysis": {
            "enabled": True,
            "scan_depth": "medium",
            "exclude_patterns": ["test_*", "*_test.py"]
        },
        "performance_analysis": {
            "enabled": True,
            "complexity_threshold": 10,
            "hotspot_threshold": 0.8
        },
        "visualization": {
            "default_layout": "hierarchical",
            "max_nodes": 1000,
            "color_scheme": "default",
            "enable_animations": True
        }
    }
    
    with open("analysis_config.json", "w") as f:
        json.dump(analysis_config, f, indent=2)
    
    # Sample visualization configuration
    viz_config = {
        "dependency_graph": {
            "layout": "hierarchical",
            "node_sizing": "proportional",
            "edge_bundling": True,
            "highlight_cycles": True
        },
        "call_flow_diagram": {
            "layout": "directed",
            "show_entry_points": True,
            "show_leaf_functions": True,
            "max_call_chains": 5
        },
        "quality_heatmap": {
            "color_scale": "red_yellow_green",
            "show_file_names": True,
            "group_by_directory": True
        },
        "architectural_overview": {
            "component_detection": "automatic",
            "show_relationships": True,
            "pattern_highlighting": True
        }
    }
    
    with open("visualization_config.json", "w") as f:
        json.dump(viz_config, f, indent=2)
    
    print("✅ Sample configuration files created")

def start_development_server():
    """Start the development server."""
    print("🚀 Starting development server...")
    
    # Check if enhanced API exists
    api_file = Path("backend/enhanced_api.py")
    if not api_file.exists():
        print("❌ Enhanced API file not found")
        return False
    
    print("   Server will start on http://localhost:8001")
    print("   API documentation available at http://localhost:8001/docs")
    print("   Press Ctrl+C to stop the server")
    
    # Start the server
    try:
        subprocess.run([
            "python", "-m", "uvicorn", 
            "backend.enhanced_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    
    return True

def deploy_to_modal():
    """Deploy the enhanced API to Modal."""
    print("☁️  Deploying to Modal...")
    
    # Check if Modal is configured
    success, stdout, stderr = run_command("modal token show")
    if not success:
        print("❌ Modal not configured. Please run 'modal token new' first")
        return False
    
    # Deploy the enhanced API
    success, stdout, stderr = run_command("modal deploy backend/enhanced_api.py")
    
    if success:
        print("✅ Deployed to Modal successfully")
        print(stdout)
    else:
        print("❌ Deployment failed")
        print(stderr)
    
    return success

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic Analysis:")
    print("""
from backend.advanced_analysis import perform_comprehensive_analysis
from codegen.sdk.core.codebase import Codebase

# Load codebase
codebase = Codebase.from_repo("owner/repository")

# Perform comprehensive analysis
results = perform_comprehensive_analysis(codebase)

# Access results
dependency_analysis = results['dependency_analysis']
print(f"Total dependencies: {dependency_analysis.total_dependencies}")
""")
    
    print("\n2. Specific Analysis Types:")
    print("""
from backend.advanced_analysis import AnalysisType

# Perform only dependency and security analysis
analysis_types = [AnalysisType.DEPENDENCY, AnalysisType.SECURITY]
results = perform_comprehensive_analysis(codebase, analysis_types)
""")
    
    print("\n3. Creating Visualizations:")
    print("""
from backend.enhanced_visualizations import create_comprehensive_dashboard_data

# Create all visualizations
dashboard_data = create_comprehensive_dashboard_data(codebase)

# Access specific visualizations
dependency_graph = dashboard_data['dependency_graph']
quality_heatmap = dashboard_data['quality_heatmap']
""")
    
    print("\n4. API Usage:")
    print("""
# POST /analyze_comprehensive
curl -X POST "http://localhost:8001/analyze_comprehensive" \\
     -H "Content-Type: application/json" \\
     -d '{
       "repo_url": "owner/repository",
       "analysis_types": ["dependency", "code_quality"],
       "include_visualizations": true
     }'
""")

def main():
    """Main setup function."""
    print("🔧 Enhanced Codebase Analytics Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Create sample configurations
    create_sample_config()
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but continuing setup...")
    
    # Validate installation
    if not validate_installation():
        print("❌ Installation validation failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review the .env file and update configuration as needed")
    print("2. Review analysis_config.json and visualization_config.json")
    print("3. Start the development server: python setup_enhanced_analysis.py --start-server")
    print("4. Deploy to Modal: python setup_enhanced_analysis.py --deploy")
    
    show_usage_examples()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Codebase Analytics Setup")
    parser.add_argument("--start-server", action="store_true", help="Start development server")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Modal")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--validate", action="store_true", help="Validate installation only")
    
    args = parser.parse_args()
    
    if args.start_server:
        start_development_server()
    elif args.deploy:
        deploy_to_modal()
    elif args.test:
        run_tests()
    elif args.validate:
        validate_installation()
    else:
        main()

