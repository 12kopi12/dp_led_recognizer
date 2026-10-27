"""Entry point for python -m src.dataset_generator."""

import sys

try:
    from ..dataset_generator import main
except (ImportError, ValueError):
    from src.dataset_generator import main

if __name__ == "__main__":
    sys.exit(main())
