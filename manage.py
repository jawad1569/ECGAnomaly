"""
Windows-friendly management script for ECG Anomaly Detection MLOps project.
Replaces Makefile commands for Windows users.

Usage:
    python manage.py data       # Validate PTB-XL and build features
    python manage.py train      # AutoML training + MLflow + save model
    python manage.py dev         # Start FastAPI dev server
    python manage.py test        # Run tests
    python manage.py lint        # Run linters
    python manage.py format      # Format code
    python manage.py docker-build  # Build the API Docker image
    python manage.py docker-run    # Run it on http://localhost:8000
    python manage.py docker-stop   # Stop and remove the container
"""

import subprocess
import sys


def run_command(cmd, description, cwd=None):
    """Execute a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    if cwd:
        print(f"Working Directory: {cwd}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✅ {description} completed successfully")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    commands = {
        "train": {"cmd": "python src/pipeline/train.py", "desc": "Training Pipeline"},
        "dev": {
            "cmd": "python -m uvicorn src.api.main:app --reload",
            "desc": "FastAPI Development Server",
        },
        "test": {"cmd": "pytest -v", "desc": "Test Suite"},
        "lint": {"cmd": "ruff check .", "desc": "Code Linting"},
        "format": {"cmd": "black .", "desc": "Code Formatting"},
        "clean": {
            "cmd": (
                "python -c \"import shutil, glob; [shutil.rmtree(d, ignore_errors=True) for d in "
                "glob.glob('**/__pycache__', recursive=True) "
                "+ glob.glob('**/.pytest_cache', recursive=True)]\""
            ),
            "desc": "Clean Cache Files",
        },
        "check": {"cmd": "pre-commit run --all-files", "desc": "Pre-commit Hooks"},
        "docker-build": {"cmd": "docker build -t ecg-api:latest .", "desc": "Build Docker image"},
        "docker-run": {
            "cmd": "docker run -d --name ecg-api -p 8000:8000 ecg-api:latest",
            "desc": "Run API container on http://localhost:8000",
        },
        "docker-stop": {"cmd": "docker rm -f ecg-api", "desc": "Stop and remove API container"},
        "data": {
            "cmd": "python src/pipeline/prepare_data.py",
            "desc": "Data Preparation (validate + features)",
        },
    }

    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

    cmd_info = commands[command]
    cwd = cmd_info.get("cwd", None)
    run_command(cmd_info["cmd"], cmd_info["desc"], cwd=cwd)


if __name__ == "__main__":
    main()
