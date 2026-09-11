"""
Convenience launcher for the Jindal Stainless Steel Surface Defect Detection Dashboard.

Usage:
    python run_dashboard.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    app_path = ROOT / "dashboard" / "app.py"
    print(f"Launching Jindal Defect Detection Dashboard from {app_path}...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501", "--server.headless", "false"]
    subprocess.run(cmd)
