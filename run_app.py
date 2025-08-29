#!/usr/bin/env python3
"""
Smart PDF Parser Application Launcher

Simple launcher script for the Smart PDF Parser Streamlit application.
"""

import sys
import subprocess
from pathlib import Path
import os


def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import docling
        from pathlib import Path
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False


def main():
    """Main launcher function."""
    print("🚀 Starting Smart PDF Parser...")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "src" / "ui" / "app.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print(f"Current directory: {current_dir}")
        print("Expected structure: src/ui/app.py")
        return 1
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Set up environment
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    
    # Add src to Python path for both this process and child processes
    src_path = str(current_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Set PYTHONPATH environment variable for Streamlit subprocess
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if current_pythonpath:
        os.environ['PYTHONPATH'] = f"{src_path}{os.pathsep}{current_pythonpath}"
    else:
        os.environ['PYTHONPATH'] = src_path
    
    print("✅ Environment configured")
    print("✅ Starting Streamlit server...")
    print("🌐 Opening browser at: http://localhost:8501")
    print("=" * 50)
    print("📝 Instructions:")
    print("1. Upload PDF files on the Parse page")
    print("2. Use the Search page to find content")
    print("3. Use the Verify page for visual verification")
    print("4. Export results on the Export page")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "src/ui/app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=false",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#667eea",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f0f2f6",
            "--theme.textColor=#262730"
        ]
        
        subprocess.run(cmd, cwd=current_dir)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down Smart PDF Parser...")
        print("Thank you for using Smart PDF Parser! 👋")
        return 0
    
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)