"""CLI argument parsing and entrypoint execution for dataset_generator."""

import argparse
import logging
import os
import sys
from typing import List, Optional

# Ensure project root and src/ are in sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from .generator import DatasetGenerator, VALID_BOX_STATES, VALID_COLORS
except (ImportError, ValueError):
    from src.dataset_generator.generator import (
        DatasetGenerator,
        VALID_BOX_STATES,
        VALID_COLORS,
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser for dataset_generator."""
    parser = argparse.ArgumentParser(
        description="Dataset Generator for LED State Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--camera-type",
        "-t",
        default="ids",
        help="Camera driver type (e.g. 'ids', 'mock', 'opencv')",
    )
    parser.add_argument(
        "--camera-path",
        "-cp",
        required=True,
        help="Path to directory containing camera configuration profiles (.cset files)",
    )
    parser.add_argument(
        "--roi-path",
        "-rp",
        required=True,
        help="Path to JSON file containing the ROI configuration",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory where images and metadata.csv will be saved",
    )
    parser.add_argument(
        "--color",
        "-c",
        required=True,
        choices=sorted(VALID_COLORS),
        help=f"Target LED color/state: {sorted(VALID_COLORS)}",
    )
    parser.add_argument(
        "--box",
        "-b",
        required=True,
        choices=sorted(VALID_BOX_STATES),
        help=f"Laboratory box environment state: {sorted(VALID_BOX_STATES)}",
    )
    parser.add_argument(
        "--distance",
        "-d",
        required=True,
        help="Distance parameter string (e.g. '0', '10', '5')",
    )
    parser.add_argument(
        "--dev",
        required=True,
        help="Device identifier string (e.g. 'BPS', 'ODS')",
    )
    parser.add_argument(
        "--settle-delay",
        type=float,
        default=0.5,
        help="Settling time in seconds after applying a camera configuration profile",
    )
    parser.add_argument(
        "--discard-count",
        type=int,
        default=2,
        help="Number of transition frames to discard after reconfiguration",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution function for CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        generator = DatasetGenerator(camera_type=args.camera_type)
        generated = generator.run(
            camera_path=args.camera_path,
            roi_path=args.roi_path,
            output_dir=args.output_dir,
            color=args.color,
            box=args.box,
            distance=args.distance,
            dev=args.dev,
            settle_delay=args.settle_delay,
            discard_count=args.discard_count,
        )
        print(f"Successfully generated {len(generated)} image(s) in '{args.output_dir}'")
        return 0

    except Exception as e:
        logging.error("Dataset generation failed: %s", e)
        if args.verbose:
            logging.exception(e)
        return 1
