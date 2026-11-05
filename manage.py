"""
Windows-friendly management script for HEARTSIGHT MLOps project.
Replaces Makefile commands for Windows users.

Usage:
    python manage.py train      # Run training pipeline
    python manage.py dev         # Start FastAPI dev server
    python manage.py test        # Run tests
    python manage.py lint        # Run linters
    python manage.py format      # Format code
"""

import sys
import subprocess


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
        "lint": {"cmd": "ruff check . && black --check .", "desc": "Code Linting"},
        "format": {"cmd": "black .", "desc": "Code Formatting"},
        "clean": {
            "cmd": "python -c \"import shutil; import glob; [shutil.rmtree(d) for d in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(d) for d in glob.glob('**/.pytest_cache', recursive=True)]\"",
            "desc": "Clean Cache Files",
        },
        "check": {"cmd": "pre-commit run --all-files", "desc": "Pre-commit Hooks"},
        "metrics": {
            "cmd": "python src/monitoring/prometheus_metrics.py",
            "desc": "Start Prometheus Metrics Exporter (port 9000)",
        },
        "ingest": {
            "cmd": "python src/ingest.py",
            "desc": "Ingest PDFs and create vector database for RAG",
        },
        "rag": {
            "cmd": "python src/ingest.py",
            "desc": "Alias for ingest - build RAG vector database",
        },
        "ui": {
            "cmd": "npm run dev",
            "desc": "Start React frontend development server (port 3000)",
            "cwd": "ui",
        },
        "ui-build": {
            "cmd": "npm run build",
            "desc": "Build React frontend for production",
            "cwd": "ui",
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
