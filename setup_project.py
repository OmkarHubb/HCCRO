#!/usr/bin/env python3
"""
setup_project.py
================
Automated setup script for the Hierarchical Cognitive Cyber Resilience Optimization (HCCRO) framework.
Generates package directories, package __init__.py files, and standard configuration files.
"""

import os
from pathlib import Path

DIRECTORIES = [
    "config",
    "data/raw",
    "data/processed",
    "src",
    "src/core",
    "src/stages",
    "src/simulation",
    "src/utils",
    "tests",
]

INIT_FILES = [
    "config/__init__.py",
    "src/__init__.py",
    "src/core/__init__.py",
    "src/stages/__init__.py",
    "src/simulation/__init__.py",
    "src/utils/__init__.py",
    "tests/__init__.py",
]

def setup_hccro_project(target_dir: str = ".") -> None:
    base_path = Path(target_dir).resolve()
    print(f"=== Initializing HCCRO Framework Structure at: {base_path} ===")

    # 1. Create directory tree
    for rel_dir in DIRECTORIES:
        dpath = base_path / rel_dir
        dpath.mkdir(parents=True, exist_ok=True)
        print(f"[DIR CREATED] {dpath}")

    # 2. Touch __init__.py files
    for rel_init in INIT_FILES:
        ipath = base_path / rel_init
        if not ipath.exists():
            ipath.write_text('"""HCCRO Package Module."""\n', encoding="utf-8")
            print(f"[INIT CREATED] {ipath}")

    print("=== HCCRO Framework Setup Complete ===")

if __name__ == "__main__":
    setup_hccro_project()
