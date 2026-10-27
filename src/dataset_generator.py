"""CLI entrypoint script in root of src for dataset_generator."""

import os
import sys

# Ensure project root and src/ directory are in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from dataset_generator.cli import build_parser, main
except ImportError:
    from src.dataset_generator.cli import build_parser, main

if __name__ == "__main__":
    sys.exit(main())
