import argparse
import shutil
import subprocess
from pathlib import Path

import uvicorn

from robot_data_studio.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Robot Data Studio")
    parser.add_argument("--projects-dir", type=Path, default=Path.cwd() / "projects")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    npm = shutil.which("npm")
    if npm is None:
        parser.error("Node.js 20.19+ or 22.12+ and npm are required to build the frontend")
    try:
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
        subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Frontend build failed (exit {error.returncode}).\n")
    uvicorn.run(create_app(args.projects_dir, frontend / "dist"), host=args.host, port=args.port)
