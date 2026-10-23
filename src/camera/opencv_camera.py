"""OpenCV Camera driver using DirectShow / V4L backend.

Implements the ICamera interface for webcams and USB cameras via OpenCV VideoCapture,
supporting JSON configuration loading (e.g., prototyping/camera_settings.json).
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, override

import cv2
import numpy as np

try:
    from .camera import ICamera
except ImportError:
    from camera import ICamera

logger = logging.getLogger(__name__)


class OpenCVCamera(ICamera):
    """Camera driver wrapping OpenCV cv2.VideoCapture with configuration support."""

    PROPERTY_MAP: Dict[str, int] = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "sharpness": cv2.CAP_PROP_SHARPNESS,
        "saturation": cv2.CAP_PROP_SATURATION,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "auto_wb": cv2.CAP_PROP_AUTO_WB,
        "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
        "gamma": cv2.CAP_PROP_GAMMA,
        "fps": cv2.CAP_PROP_FPS,
        "frame_rate": cv2.CAP_PROP_FPS,
        "width": cv2.CAP_PROP_FRAME_WIDTH,
        "height": cv2.CAP_PROP_FRAME_HEIGHT,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        camera_index: int = 0,
        frame_width: Optional[int] = 1920,
        frame_height: Optional[int] = 1080,
        fourcc: Optional[str] = "MJPG",
        backend: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenCV camera driver.

        Args:
            config_path: Optional path to JSON camera configuration file.
            camera_index: Camera index (default: 0).
            frame_width: Target frame width (default: 1920).
            frame_height: Target frame height (default: 1080).
            fourcc: Video codec FourCC string (default: 'MJPG').
            backend: cv2 VideoCapture backend (default: cv2.CAP_DSHOW on Windows).
            **kwargs: Extra parameters passed to base class.
        """
        super().__init__(config_path=config_path, **kwargs)

        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fourcc = fourcc
        self.backend = backend if backend is not None else cv2.CAP_DSHOW

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened: bool = False

    @override
    def open(self) -> bool:
        """Opens camera via cv2.VideoCapture and applies resolution & settings.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._is_opened and self.cap is not None and self.cap.isOpened():
            return True

        logger.info("Opening OpenCV camera index %d (backend=%d)...", self.camera_index, self.backend)
        self.cap = cv2.VideoCapture(self.camera_index, self.backend)

        if not self.cap.isOpened():
            logger.error("Failed to open OpenCV camera at index %d.", self.camera_index)
            self.cap = None
            self._is_opened = False
            return False

        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.frame_width))
        if self.frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.frame_height))

        # Apply configuration file if provided
        if self.config_path:
            self.load_configuration(self.config_path)

        self._is_opened = True
        logger.info("OpenCV camera opened successfully.")
        return True

    @override
    def close(self) -> None:
        """Releases the VideoCapture instance."""
        if self.cap is not None:
            logger.info("Closing OpenCV camera index %d...", self.camera_index)
            self.cap.release()
            self.cap = None
        self._is_opened = False

    @override
    def is_opened(self) -> bool:
        """Checks if the camera device is open."""
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    @override
    def flush_buffers(self, discard_count: int = 2) -> None:
        """Flushes buffered frames in the DirectShow / V4L queue."""
        if not self.is_opened():
            return
        for _ in range(discard_count):
            self.cap.grab()

    @override
    def get_frame(self, timeout_ms: int = 1000, latest_only: bool = True) -> Optional[np.ndarray]:
        """Captures a frame from OpenCV VideoCapture.

        Args:
            timeout_ms: Timeout in milliseconds (unused by VideoCapture.read).
            latest_only: If True, grabs and discards any stale buffer before reading.

        Returns:
            Captured frame as BGR uint8 NumPy array, or None if reading failed.
        """
        if not self.is_opened():
            logger.error("Cannot capture: OpenCV camera is not opened.")
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame from OpenCV camera.")
            return None

        return frame

    @override
    def set_parameter(self, name: str, value: Any, flush: bool = True) -> bool:
        """Sets an OpenCV VideoCapture property."""
        if not self.is_opened():
            logger.error("Cannot set parameter: camera is not opened.")
            return False

        prop_id = self.PROPERTY_MAP.get(name.lower())
        if prop_id is None:
            logger.warning("Unsupported parameter name: '%s'", name)
            return False

        success = self.cap.set(prop_id, float(value))
        if not success:
            logger.warning("cv2.VideoCapture.set failed for property '%s' = %s", name, value)
        elif flush:
            self.flush_buffers(discard_count=2)

        return success

    @override
    def get_parameter(self, name: str) -> Any:
        """Retrieves an OpenCV VideoCapture property."""
        if not self.is_opened():
            logger.error("Cannot get parameter: camera is not opened.")
            return None

        prop_id = self.PROPERTY_MAP.get(name.lower())
        if prop_id is None:
            return None

        return self.cap.get(prop_id)

    @override
    def load_configuration(self, config_path: str) -> bool:
        """Loads camera settings from a JSON file.

        Supports structured configurations containing 'recommended_for_led' or flat
        key-value pairs of camera properties.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            True if successfully loaded and applied, False otherwise.
        """
        abs_path = os.path.abspath(config_path)
        if not os.path.isfile(abs_path):
            logger.error("Configuration file not found: %s", abs_path)
            return False

        self.config_path = abs_path
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            params = {}
            if "recommended_for_led" in data and isinstance(data["recommended_for_led"], dict):
                params.update(data["recommended_for_led"])
            elif isinstance(data, dict):
                params.update({k: v for k, v in data.items() if not isinstance(v, dict)})

            if self.is_opened():
                for param_name, param_value in params.items():
                    self.set_parameter(param_name, param_value, flush=False)
                self.flush_buffers(discard_count=2)
                logger.info("Applied %d settings from %s", len(params), abs_path)

            return True

        except Exception as e:
            logger.error("Failed to load JSON camera configuration from %s: %s", abs_path, e)
            return False
