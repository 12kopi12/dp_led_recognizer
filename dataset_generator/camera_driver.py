"""Camera driver module for dataset acquisition.

Manages camera connection via OpenCV DirectShow backend, safe configuration
of camera properties (brightness, contrast, sharpness, saturation, exposure),
and frame buffer flushing for hardware stabilization.
"""

import argparse
import logging
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CameraDriver:
    """Manages OpenCV camera capture with DirectShow backend and parameter controls."""

    # Reference values defined by specification
    REFERENCE_VALUES: Dict[str, float] = {
        "brightness": 0.0,
        "contrast": 30.0,
        "sharpness": 0.0,
        "saturation": 100.0,
    }

    # Safe operating ranges (±20 from reference values, lower clamped at 0)
    PARAMETER_RANGES: Dict[str, Tuple[float, float]] = {
        "brightness": (0.0, 20.0),
        "contrast": (10.0, 50.0),
        "sharpness": (0.0, 20.0),
        "saturation": (80.0, 120.0),
    }

    # Mapping of parameter names to OpenCV property identifiers
    PROPERTY_MAP: Dict[str, int] = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
        "saturation": cv2.CAP_PROP_SATURATION,
        "exposure": cv2.CAP_PROP_EXPOSURE,
    }

    def __init__(
        self,
        camera_index: int = 1,
        buffer_flush_count: int = 4,
        disable_auto_exposure: bool = True,
        disable_auto_wb: bool = True,
    ) -> None:
        """Initialize the CameraDriver.

        Args:
            camera_index: Camera device index (default: 1).
            buffer_flush_count: Number of frames to discard after changing parameters (default: 4, range 3-5).
            disable_auto_exposure: Whether to disable camera auto exposure (manual mode).
            disable_auto_wb: Whether to disable camera auto white balance.
        """
        self.camera_index = camera_index
        self.buffer_flush_count = max(3, min(5, buffer_flush_count))
        self.disable_auto_exposure = disable_auto_exposure
        self.disable_auto_wb = disable_auto_wb

        self.cap: Optional[cv2.VideoCapture] = None
        self.baseline_exposure: Optional[float] = None
        self.exposure_range: Tuple[float, float] = (-7.0, -5.0)

    def open(self) -> bool:
        """Opens camera using cv2.VideoCapture(camera_index, cv2.CAP_DSHOW).

        Returns:
            True if camera opened successfully, False otherwise.
        """
        logger.info("Opening camera index %d with cv2.CAP_DSHOW...", self.camera_index)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            logger.error("Failed to open camera index %d.", self.camera_index)
            return False

        # Disable auto exposure if requested (in DirectShow OpenCV, 0.25 specifies manual mode)
        if self.disable_auto_exposure:
            logger.info("Disabling auto exposure (setting manual mode)...")
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        # Disable auto white balance if requested
        if self.disable_auto_wb:
            logger.info("Disabling auto white balance...")
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)

        # Query baseline exposure from camera
        exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        if exp is not None and exp != -1.0:
            self.baseline_exposure = float(exp)
            self.exposure_range = (self.baseline_exposure - 1.0, self.baseline_exposure + 1.0)
            logger.info(
                "Detected baseline exposure: %.1f (safe range: [%.1f, %.1f])",
                self.baseline_exposure,
                self.exposure_range[0],
                self.exposure_range[1],
            )
        else:
            # Fallback exposure if camera doesn't report it
            self.baseline_exposure = -6.0
            self.exposure_range = (-7.0, -5.0)
            logger.warning(
                "Could not query camera exposure. Using fallback baseline %.1f (range [%.1f, %.1f]).",
                self.baseline_exposure,
                self.exposure_range[0],
                self.exposure_range[1],
            )

        # Perform initial buffer flush
        self.flush_buffer()
        return True

    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            logger.info("Releasing camera %d...", self.camera_index)
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "CameraDriver":
        if not self.open():
            raise RuntimeError(f"Unable to open camera {self.camera_index}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def flush_buffer(self, count: Optional[int] = None) -> None:
        """Discards frames to allow camera hardware to stabilize after setting changes.

        Args:
            count: Number of frames to discard (default: self.buffer_flush_count, clamped to 3-5).
        """
        if not self.is_opened():
            return

        flush_n = count if count is not None else self.buffer_flush_count
        flush_n = max(3, min(5, flush_n))

        for _ in range(flush_n):
            # grab() is fast and discards frame from internal buffer without decoding
            self.cap.grab()
        # Small delay to ensure sensor parameter synchronization
        time.sleep(0.05)

    def get_parameter(self, param_name: str) -> Optional[float]:
        """Safely reads a camera parameter.

        Args:
            param_name: One of 'brightness', 'contrast', 'sharpness', 'saturation', 'exposure'.

        Returns:
            Parameter value or None if unreadable.
        """
        if not self.is_opened():
            logger.warning("Camera is not opened.")
            return None

        param_lower = param_name.lower()
        if param_lower not in self.PROPERTY_MAP:
            logger.error("Unknown parameter name: %s", param_name)
            return None

        prop_id = self.PROPERTY_MAP[param_lower]
        val = self.cap.get(prop_id)
        return float(val) if val != -1.0 else None

    def get_all_parameters(self) -> Dict[str, Optional[float]]:
        """Reads all supported camera parameters.

        Returns:
            Dictionary with current values for all parameters.
        """
        return {p: self.get_parameter(p) for p in self.PROPERTY_MAP}

    def set_parameter(self, param_name: str, value: float, flush: bool = True) -> bool:
        """Safely sets a camera parameter.

        Args:
            param_name: Parameter to set ('brightness', 'contrast', 'sharpness', 'saturation', 'exposure').
            value: Desired numeric value.
            flush: Whether to flush buffer frames after setting.

        Returns:
            True if set operation succeeded, False otherwise.
        """
        if not self.is_opened():
            logger.error("Cannot set parameter %s: Camera not opened.", param_name)
            return False

        param_lower = param_name.lower()
        if param_lower not in self.PROPERTY_MAP:
            logger.error("Unknown parameter: %s", param_name)
            return False

        prop_id = self.PROPERTY_MAP[param_lower]

        # Clamp values to safe boundaries
        if param_lower in self.PARAMETER_RANGES:
            min_val, max_val = self.PARAMETER_RANGES[param_lower]
            clamped_val = max(min_val, min(max_val, float(value)))
        elif param_lower == "exposure":
            min_val, max_val = self.exposure_range
            clamped_val = max(min_val, min(max_val, float(value)))
        else:
            clamped_val = float(value)

        success = self.cap.set(prop_id, clamped_val)
        if not success:
            logger.warning("Camera driver rejected setting %s to %.2f", param_lower, clamped_val)

        if flush:
            self.flush_buffer()

        return success

    # --- Explicit parameter getters and setters ---

    def get_brightness(self) -> Optional[float]:
        """Gets current brightness value."""
        return self.get_parameter("brightness")

    def set_brightness(self, value: float, flush: bool = True) -> bool:
        """Sets brightness value clamped to safe range [0, 20]."""
        return self.set_parameter("brightness", value, flush=flush)

    def get_contrast(self) -> Optional[float]:
        """Gets current contrast value."""
        return self.get_parameter("contrast")

    def set_contrast(self, value: float, flush: bool = True) -> bool:
        """Sets contrast value clamped to safe range [10, 50]."""
        return self.set_parameter("contrast", value, flush=flush)

    def get_sharpness(self) -> Optional[float]:
        """Gets current sharpness value."""
        return self.get_parameter("sharpness")

    def set_sharpness(self, value: float, flush: bool = True) -> bool:
        """Sets sharpness value clamped to safe range [0, 20]."""
        return self.set_parameter("sharpness", value, flush=flush)

    def get_saturation(self) -> Optional[float]:
        """Gets current saturation value."""
        return self.get_parameter("saturation")

    def set_saturation(self, value: float, flush: bool = True) -> bool:
        """Sets saturation value clamped to safe range [80, 120]."""
        return self.set_parameter("saturation", value, flush=flush)

    def get_exposure(self) -> Optional[float]:
        """Gets current exposure value."""
        return self.get_parameter("exposure")

    def set_exposure(self, value: float, flush: bool = True) -> bool:
        """Sets exposure value clamped to safe range [baseline-1, baseline+1]."""
        return self.set_parameter("exposure", value, flush=flush)

    def set_parameters(self, params: Dict[str, float], flush: bool = True) -> Dict[str, bool]:
        """Sets multiple camera parameters at once and performs a single buffer flush.

        Args:
            params: Dictionary mapping parameter names to values.
            flush: Whether to flush buffer after applying all parameters.

        Returns:
            Dictionary mapping each parameter name to success boolean.
        """
        results = {}
        for name, val in params.items():
            results[name] = self.set_parameter(name, val, flush=False)

        if flush:
            self.flush_buffer()

        return results

    def reset_to_reference(self, flush: bool = True) -> None:
        """Resets camera parameters to reference default values."""
        ref_params = dict(self.REFERENCE_VALUES)
        if self.baseline_exposure is not None:
            ref_params["exposure"] = self.baseline_exposure

        logger.info("Resetting camera parameters to reference values: %s", ref_params)
        self.set_parameters(ref_params, flush=flush)

    def get_random_parameters(self, include_exposure: bool = True) -> Dict[str, float]:
        """Generates random parameter values within the defined safe ranges.

        Ranges:
            brightness: [0, 20]
            contrast:   [10, 50]
            sharpness:  [0, 20]
            saturation: [80, 120]
            exposure:   [baseline - 1, baseline + 1]

        Returns:
            Dictionary of randomized parameter values.
        """
        random_params = {
            "brightness": float(random.randint(int(self.PARAMETER_RANGES["brightness"][0]), int(self.PARAMETER_RANGES["brightness"][1]))),
            "contrast": float(random.randint(int(self.PARAMETER_RANGES["contrast"][0]), int(self.PARAMETER_RANGES["contrast"][1]))),
            "sharpness": float(random.randint(int(self.PARAMETER_RANGES["sharpness"][0]), int(self.PARAMETER_RANGES["sharpness"][1]))),
            "saturation": float(random.randint(int(self.PARAMETER_RANGES["saturation"][0]), int(self.PARAMETER_RANGES["saturation"][1]))),
        }

        if include_exposure and self.baseline_exposure is not None:
            exp_choices = [
                self.baseline_exposure - 1.0,
                self.baseline_exposure,
                self.baseline_exposure + 1.0,
            ]
            random_params["exposure"] = float(random.choice(exp_choices))

        return random_params

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the camera.

        Returns:
            Tuple of (success_bool, frame_ndarray_or_None).
        """
        if not self.is_opened():
            logger.error("Camera is not opened.")
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from camera.")
            return False, None

        return True, frame


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for camera driver CLI."""
    parser = argparse.ArgumentParser(description="Camera Driver CLI for DirectShow video capture.")
    parser.add_argument(
        "--camera-index",
        "-c",
        type=int,
        default=1,
        help="Camera device index (default: 1)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open live preview window to view feed and test settings",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print current camera properties and safe ranges",
    )
    parser.add_argument(
        "--capture",
        type=str,
        default=None,
        help="Capture a single frame and save to the specified file path",
    )
    parser.add_argument(
        "--test-random",
        action="store_true",
        help="Test generating and applying random parameters with buffer flush",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for camera driver inspection and preview."""
    args = parse_args()

    driver = CameraDriver(camera_index=args.camera_index)
    if not driver.open():
        logger.error(
            "Could not connect to camera index %d. If testing locally without secondary camera, try --camera-index 0.",
            args.camera_index,
        )
        sys.exit(1)

    try:
        if args.info or (not args.preview and not args.capture and not args.test_random):
            print("\n=== Camera Parameters ===")
            params = driver.get_all_parameters()
            for k, v in params.items():
                print(f"  {k:12s}: {v}")
            print("\n=== Defined Safe Ranges ===")
            for k, (low, high) in driver.PARAMETER_RANGES.items():
                print(f"  {k:12s}: [{low:.1f}, {high:.1f}]")
            print(f"  {'exposure':12s}: [{driver.exposure_range[0]:.1f}, {driver.exposure_range[1]:.1f}]")
            print("==========================\n")

        if args.test_random:
            print("Testing randomized parameters:")
            rand_p = driver.get_random_parameters()
            print("Generated:", rand_p)
            driver.set_parameters(rand_p)
            print("Applied successfully with buffer flush.")
            driver.reset_to_reference()
            print("Reset back to reference values.")

        if args.capture:
            ret, frame = driver.read_frame()
            if ret and frame is not None:
                cv2.imwrite(args.capture, frame)
                logger.info("Saved captured frame to: %s", args.capture)
            else:
                logger.error("Failed to capture frame.")

        if args.preview:
            window_name = f"Camera Preview (Index {args.camera_index})"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print("Preview running. Press [R] to randomize parameters, [0] to reset, [Q] to quit.")

            while True:
                ret, frame = driver.read_frame()
                if not ret or frame is None:
                    break

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("r"):
                    rand_p = driver.get_random_parameters()
                    logger.info("Applying random settings: %s", rand_p)
                    driver.set_parameters(rand_p)
                elif key == ord("0"):
                    logger.info("Resetting to reference settings.")
                    driver.reset_to_reference()

            cv2.destroyAllWindows()

    finally:
        driver.release()


if __name__ == "__main__":
    main()
