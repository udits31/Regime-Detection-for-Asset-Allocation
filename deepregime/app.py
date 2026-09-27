#!/usr/bin/env python3
"""
app.py — DeepRegime Application Launcher

Usage:
    python app.py serve              # API + Dashboard
    python app.py serve --api-only   # FastAPI only
    python app.py serve --dash-only  # Streamlit only
"""

import argparse
import subprocess
import sys
import os


def serve(api_only=False, dash_only=False):
    procs = []
    if not dash_only:
        procs.append(subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ]))

    if not api_only:
        procs.append(subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "dashboard/app.py",
            "--server.port", "8501",
        ]))

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    srv = sub.add_parser("serve")
    srv.add_argument("--api-only",  action="store_true")
    srv.add_argument("--dash-only", action="store_true")
    args = parser.parse_args()

    if args.cmd == "serve":
        serve(args.api_only, args.dash_only)
