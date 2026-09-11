"""
Root Launcher for Jindal Stainless Steel Defect Detection Dashboard.
Runs from either the root NEU-DET folder or steel_defect_detection folder.
"""

import os
import sys
import subprocess
from pathlib import Path

# Locate project directory
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "steel_defect_detection" if (ROOT / "steel_defect_detection").exists() else ROOT
APP_PATH = PROJECT_DIR / "dashboard" / "app.py"

# Add Python user scripts to PATH
user_scripts = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Python" / "Python314" / "Scripts"
if user_scripts.exists():
    os.environ["PATH"] = str(user_scripts) + os.pathsep + os.environ.get("PATH", "")

if __name__ == "__main__":
    print(f"============================================================")
    print(f"  Launching Jindal Stainless AI Defect Detection Platform   ")
    print(f"  Team: GenCoders (Lipi Bagarti & Kartik Ranjan Singh)       ")
    print(f"============================================================")
    print(f"Dashboard File: {APP_PATH}")
    print(f"Server URL:     http://localhost:8501\n")

    # Try running with python -m streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        "8501",
        "--server.headless",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    try:
        subprocess.run(cmd, cwd=str(PROJECT_DIR))
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
