#!/usr/bin/env python3
"""
Smart PDF Parser Shutdown Script

Kills all running Streamlit instances and frees up port 8501.
"""

import subprocess
import sys
import os
import signal
import time


def find_streamlit_processes():
    """Find all running Streamlit processes."""
    try:
        # Find processes using port 8501
        result = subprocess.run(
            ["lsof", "-t", "-i:8501"], 
            capture_output=True, 
            text=True
        )
        port_pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Find Streamlit processes by command
        result = subprocess.run(
            ["pgrep", "-f", "streamlit"], 
            capture_output=True, 
            text=True
        )
        streamlit_pids = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Combine and deduplicate PIDs
        all_pids = list(set(port_pids + streamlit_pids))
        return [pid for pid in all_pids if pid and pid.isdigit()]
        
    except Exception as e:
        print(f"Error finding processes: {e}")
        return []


def kill_background_bash_processes():
    """Kill any background bash processes that might be running Streamlit."""
    try:
        # Kill the specific background bash process if it exists
        if os.path.exists("/tmp/streamlit_bash_pid"):
            with open("/tmp/streamlit_bash_pid", "r") as f:
                pid = f.read().strip()
                if pid.isdigit():
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"✅ Killed background bash process: {pid}")
                    except ProcessLookupError:
                        print(f"⚠️  Process {pid} already terminated")
            os.remove("/tmp/streamlit_bash_pid")
    except Exception as e:
        print(f"Error killing background processes: {e}")


def kill_processes(pids):
    """Kill processes by PID."""
    if not pids:
        print("No Streamlit processes found.")
        return True
    
    success = True
    for pid in pids:
        try:
            print(f"🛑 Killing process {pid}...")
            os.kill(int(pid), signal.SIGTERM)
            
            # Wait a moment for graceful shutdown
            time.sleep(1)
            
            # Check if process still exists
            try:
                os.kill(int(pid), 0)  # Just check if process exists
                # If we get here, process still exists, force kill
                print(f"⚡ Force killing process {pid}...")
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                # Process is gone, good
                pass
                
            print(f"✅ Process {pid} terminated")
            
        except ProcessLookupError:
            print(f"⚠️  Process {pid} already terminated")
        except PermissionError:
            print(f"❌ Permission denied to kill process {pid}")
            success = False
        except Exception as e:
            print(f"❌ Error killing process {pid}: {e}")
            success = False
    
    return success


def check_port_8501():
    """Check if port 8501 is still in use."""
    try:
        result = subprocess.run(
            ["lsof", "-t", "-i:8501"], 
            capture_output=True, 
            text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def main():
    """Main shutdown function."""
    print("🛑 Smart PDF Parser Shutdown Tool")
    print("=" * 50)
    
    # Kill background bash processes first
    kill_background_bash_processes()
    
    # Find and kill Streamlit processes
    pids = find_streamlit_processes()
    
    if pids:
        print(f"Found {len(pids)} process(es): {', '.join(pids)}")
        success = kill_processes(pids)
        
        # Wait a moment and check again
        time.sleep(2)
        
        if check_port_8501():
            print("⚠️  Port 8501 is still in use. Trying alternative methods...")
            
            # Try killing anything on port 8501 more aggressively
            try:
                subprocess.run(["pkill", "-f", "8501"], check=False)
                subprocess.run(["pkill", "-f", "streamlit"], check=False)
                time.sleep(1)
                
                if check_port_8501():
                    print("❌ Could not free port 8501. You may need to restart your terminal or use:")
                    print("   sudo lsof -t -i:8501 | xargs sudo kill -9")
                else:
                    print("✅ Port 8501 is now free!")
            except Exception as e:
                print(f"Error with alternative shutdown: {e}")
        else:
            print("✅ All processes terminated successfully!")
            print("✅ Port 8501 is now free!")
    else:
        if check_port_8501():
            print("⚠️  No Streamlit processes found, but port 8501 is still in use.")
            print("This might be another application. Use:")
            print("   lsof -i:8501")
            print("to identify what's using the port.")
        else:
            print("✅ No Streamlit processes found and port 8501 is free!")
    
    print("=" * 50)
    print("Shutdown complete. You can now run the app again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)