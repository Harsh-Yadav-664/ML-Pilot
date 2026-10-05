import subprocess
import sys
import time
import os

def main():
    print("="*50)
    print("🚀 Starting MLPilot (Backend + Frontend)")
    print("="*50)
    print("Press Ctrl+C to shut down both servers.")
    
    # Ensure directories exist
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        print("Error: Must be run from the MLPilot root directory.")
        sys.exit(1)

    # Start Backend
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload"],
        cwd="backend"
    )
    
    # Start Frontend (npm needs a shell on Windows; on Linux/macOS shell=True would drop "run dev")
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        shell=(os.name == "nt"),
    )
    
    try:
        # Keep the main thread alive while subprocesses run
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print("🛑 Shutting down MLPilot servers...")
        print("="*50)
        backend.terminate()
        frontend.terminate()
        backend.wait()
        
        print("✅ Shutdown complete. Have a great day!")
        sys.exit(0)

if __name__ == "__main__":
    main()
