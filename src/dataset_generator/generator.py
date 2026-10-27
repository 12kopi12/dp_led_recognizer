"""Dataset generator for LED state recognition."""

import csv
from datetime import datetime, timezone
import glob
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

# Ensure project root and src are in sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.camera.factory import CameraFactory
    from src.camera.camera import ICamera
    from src.roi.manager import ROIManager
    from src.roi.roi import ROI, CircleROI
    from src.dataset_generator.cset_parser import (
        format_value_str,
        parse_cset_file,
        round_decimal,
    )
except ImportError:
    from camera.factory import CameraFactory
    from camera.camera import ICamera
    from roi.manager import ROIManager
    from roi.roi import ROI, CircleROI
    from dataset_generator.cset_parser import (
        format_value_str,
        parse_cset_file,
        round_decimal,
    )

logger = logging.getLogger(__name__)

# Valid discrete domains from dataset specification
VALID_COLORS: Set[str] = {"green", "red", "orange", "off"}
VALID_BOX_STATES: Set[str] = {"open", "closed", "light"}

# Default column order for metadata.csv
DEFAULT_METADATA_COLUMNS: List[str] = [
    "filename",
    "dev",
    "shape",
    "color",
    "exposuretime",
    "gain",
    "r",
    "g",
    "b",
    "distance",
    "box",
    "rep",
    "profile_name",
    "timestamp",
]


class DatasetGenerator:
    """Automates image acquisition and dataset compilation for LED recognition."""

    def __init__(
        self,
        camera_type: str = "ids",
        camera_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize DatasetGenerator.

        Args:
            camera_type: Driver identifier for CameraFactory (e.g. 'ids', 'mock').
            camera_config: Optional kwargs dictionary passed to the camera constructor.
        """
        self.camera_type = camera_type.lower()
        self.camera_config = dict(camera_config) if camera_config is not None else {}

    def run(
        self,
        camera_path: str,
        roi_path: str,
        output_dir: str,
        color: str,
        box: str,
        distance: str,
        dev: str,
        settle_delay: float = 0.5,
        discard_count: int = 2,
    ) -> List[str]:
        """Runs the capture pipeline across all camera profiles.

        Args:
            camera_path: Directory containing camera configuration files (.cset for IDS).
            roi_path: Path to the JSON configuration file containing a single ROI.
            output_dir: Directory where lossless PNGs and metadata.csv will be saved.
            color: Target LED state ('green', 'red', 'orange', 'off').
            box: Ambient box state ('open', 'closed', 'light').
            distance: Distance value (e.g. '0', '10', '5').
            dev: Device identifier (e.g. 'BPS', 'ODS').
            settle_delay: Pause in seconds after applying a camera profile (default: 0.5s).
            discard_count: Number of transition frames to discard after reconfiguration (default: 2).

        Returns:
            List of generated image file names.

        Raises:
            ValueError: If any input parameters fail validation.
            FileNotFoundError: If input paths do not exist.
            RuntimeError: If camera communication or capture fails.
        """
        # 1. Parameter Validation
        color_lower = color.strip().lower()
        if color_lower not in VALID_COLORS:
            raise ValueError(
                f"Invalid color '{color}'. Allowed values: {sorted(VALID_COLORS)}"
            )

        box_lower = box.strip().lower()
        if box_lower not in VALID_BOX_STATES:
            raise ValueError(
                f"Invalid box state '{box}'. Allowed values: {sorted(VALID_BOX_STATES)}"
            )

        dev_clean = dev.strip()
        if not dev_clean:
            raise ValueError("Device identifier 'dev' cannot be empty.")

        dist_clean = str(distance).strip()
        if not dist_clean:
            raise ValueError("Distance parameter cannot be empty.")

        if not os.path.isdir(camera_path):
            raise FileNotFoundError(f"Camera config directory not found: {camera_path}")

        if not os.path.isfile(roi_path):
            raise FileNotFoundError(f"ROI configuration file not found: {roi_path}")

        # 2. Discover and sort .cset profile files
        cset_files = sorted(glob.glob(os.path.join(camera_path, "*.cset")))
        if not cset_files:
            raise FileNotFoundError(
                f"No .cset configuration files found in camera_path: {camera_path}"
            )

        # 3. Load ROI configuration
        roi_manager = ROIManager()
        roi_manager.load_config(roi_path)
        if not roi_manager.rois:
            raise ValueError(f"No ROIs defined in configuration file: {roi_path}")

        roi: ROI = next(iter(roi_manager.rois.values()))
        # Normalize shape identifier strictly to 'circle' or 'rect'
        shape_norm = "circle" if "circle" in roi.shape else "rect"
        logger.info(
            "Loaded ROI '%s' (shape: %s, normalized: %s) from %s",
            roi.label,
            roi.shape,
            shape_norm,
            roi_path,
        )

        # 4. Prepare output directory and metadata CSV
        os.makedirs(output_dir, exist_ok=True)
        metadata_csv_path = os.path.join(output_dir, "metadata.csv")
        csv_columns = self._prepare_metadata_csv(metadata_csv_path)

        # 5. Connect and initialize camera
        camera: ICamera = CameraFactory.create_camera(
            self.camera_type, config=self.camera_config
        )
        logger.info("Opening camera of type '%s'...", self.camera_type)
        if not camera.open():
            raise RuntimeError(f"Failed to open camera of type '{self.camera_type}'")

        generated_files: List[str] = []

        try:
            logger.info("Starting acquisition across %d profile(s)...", len(cset_files))
            for profile_idx, cset_file in enumerate(cset_files, start=1):
                cset_filename = os.path.basename(cset_file)
                logger.info(
                    "[%d/%d] Processing profile: %s",
                    profile_idx,
                    len(cset_files),
                    cset_filename,
                )

                # Parse parameters from .cset file
                params = parse_cset_file(cset_file)

                # Apply profile configuration to camera hardware
                success = camera.load_configuration(cset_file)
                if not success:
                    logger.warning(
                        "Camera driver reported failure loading profile '%s', proceeding with capture.",
                        cset_file,
                    )

                # Fixed settling delay for analog circuitry / auto-routines
                if settle_delay > 0:
                    time.sleep(settle_delay)

                # Drain stale buffers and discard transition frames
                self._flush_camera_pipeline(camera, discard_count)

                # Capture 2 repetitions (rep 0 and rep 1)
                for rep_idx in (0, 1):
                    frame = camera.get_frame()
                    if frame is None:
                        raise RuntimeError(
                            f"Failed to capture frame for profile '{cset_filename}', repetition {rep_idx}"
                        )

                    # Validate resolution if reference resolution is specified in ROI config
                    roi_manager.validate_resolution(frame)

                    # Extract ROI crop (apply circular black mask if circle shape)
                    if shape_norm == "circle" and isinstance(roi, CircleROI):
                        cropped_roi = roi.crop(frame, mask=True)
                    else:
                        cropped_roi = roi.crop(frame)

                    # Construct standardized filename:
                    # {dev}_{shape}_{color}_exp{exposuretime}_gain{gain}_r{r}_g{g}_b{b}_dist{dist}cm_{box}_rep{idx}.png
                    exp_str = format_value_str(params["exposuretime"])
                    gain_str = format_value_str(params["gain"])
                    r_str = format_value_str(params["r"])
                    g_str = format_value_str(params["g"])
                    b_str = format_value_str(params["b"])
                    dist_str = format_value_str(dist_clean)

                    filename = (
                        f"{dev_clean}_{shape_norm}_{color_lower}_"
                        f"exp{exp_str}_gain{gain_str}_"
                        f"r{r_str}_g{g_str}_b{b_str}_"
                        f"dist{dist_str}cm_{box_lower}_rep{rep_idx}.png"
                    )

                    file_path = os.path.join(output_dir, filename)

                    # Save lossless PNG
                    save_success = cv2.imwrite(file_path, cropped_roi)
                    if not save_success:
                        raise IOError(f"Failed to write image file: {file_path}")

                    # Write record to metadata.csv
                    row_data: Dict[str, Any] = {
                        "filename": filename,
                        "dev": dev_clean,
                        "shape": shape_norm,
                        "color": color_lower,
                        "exposuretime": params["exposuretime"],
                        "gain": round_decimal(params["gain"]),
                        "r": round_decimal(params["r"]),
                        "g": round_decimal(params["g"]),
                        "b": round_decimal(params["b"]),
                        "distance": round_decimal(dist_clean),
                        "box": box_lower,
                        "rep": rep_idx,
                        "profile_name": params["profile_name"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    self._append_metadata_row(metadata_csv_path, csv_columns, row_data)
                    generated_files.append(filename)
                    logger.debug("Saved image: %s", filename)

            logger.info(
                "Acquisition completed successfully. %d images generated in '%s'.",
                len(generated_files),
                output_dir,
            )
            return generated_files

        finally:
            logger.info("Closing camera...")
            camera.close()

    def _flush_camera_pipeline(self, camera: ICamera, discard_count: int) -> None:
        """Flushes hardware queue and discards transition frames after reconfiguration."""
        # Use native driver buffer flush if available
        if hasattr(camera, "flush_buffers"):
            try:
                camera.flush_buffers(discard_count=discard_count)
            except Exception as e:
                logger.debug("Error during camera.flush_buffers(): %s", e)

        # Fallback/safeguard: actively grab and discard frames
        for _ in range(discard_count):
            try:
                camera.get_frame(timeout_ms=500)
            except Exception as e:
                logger.debug("Discard frame read timed out or failed: %s", e)
                break

    def _prepare_metadata_csv(self, csv_path: str) -> List[str]:
        """Prepares metadata.csv, ensuring valid headers and column ordering.

        If the file already exists and is non-empty, its existing header is preserved.
        Otherwise, DEFAULT_METADATA_COLUMNS is written as header.

        Returns:
            List of column names in order.
        """
        if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    if header and header[0] == "filename":
                        return header
                except StopIteration:
                    pass

        # Write fresh header
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(DEFAULT_METADATA_COLUMNS)

        return DEFAULT_METADATA_COLUMNS

    def _append_metadata_row(
        self, csv_path: str, columns: List[str], data: Dict[str, Any]
    ) -> None:
        """Appends a single metadata row to metadata.csv according to column order."""
        row = [data.get(col, "") for col in columns]
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
