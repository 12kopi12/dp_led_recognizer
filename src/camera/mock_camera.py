"""Mock Camera implementation for offline testing and development.

Allows running pipeline, ROI annotation, dataset generation, and Robot Framework
tests without physical camera hardware.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, override

import cv2
import numpy as np

try:
    from .camera import ICamera
except ImportError:
    from camera import ICamera

logger = logging.getLogger(__name__)


class MockCamera(ICamera):
    """Mock camera that simulates camera acquisition from image files or synthetic generation."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        frames_dir: Optional[str] = None,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize MockCamera.

        Args:
            config_path: Optional path to JSON configuration file.
            frames_dir: Directory containing pre-recorded image files to cycle through.
            width: Simulated frame width in pixels (default: 1920).
            height: Simulated frame height in pixels (default: 1080).
            fps: Simulated frame rate (default: 30.0).
            **kwargs: Extra parameters passed to base class.
        """
        super().__init__(config_path=config_path, **kwargs)

        self.frames_dir = frames_dir
        self.width = width
        self.height = height
        self.fps = fps

        self._is_opened: bool = False
        self._frame_files: List[str] = []
        self._current_index: int = 0
        self._parameters: Dict[str, Any] = {
            "exposure": -5.0,
            "gain": 1.0,
            "brightness": 0.0,
            "contrast": 20.0,
            "saturation": 20.0,
            "fps": fps,
            "width": width,
            "height": height,
        }

    @override
    def open(self) -> bool:
        """Opens mock camera and indexes frames directory if provided."""
        if self._is_opened:
            return True

        if self.frames_dir and os.path.isdir(self.frames_dir):
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            self._frame_files = [
                os.path.join(self.frames_dir, f)
                for f in sorted(os.listdir(self.frames_dir))
                if f.lower().endswith(valid_exts)
            ]
            logger.info("MockCamera loaded %d frames from %s", len(self._frame_files), self.frames_dir)
        else:
            self._frame_files = []
            logger.info("MockCamera running in synthetic frame generation mode (%dx%d)", self.width, self.height)

        if self.config_path:
            self.load_configuration(self.config_path)

        self._is_opened = True
        self._current_index = 0
        return True

    @override
    def close(self) -> None:
        """Closes mock camera."""
        self._is_opened = False
        self._current_index = 0

    @override
    def is_opened(self) -> bool:
        """Returns True if mock camera is open."""
        return self._is_opened

    @override
    def flush_buffers(self, discard_count: int = 2) -> None:
        """Mock flush: advances index if pre-recorded frames are present."""
        if self._frame_files:
            self._current_index = (self._current_index + discard_count) % len(self._frame_files)

    @override
    def get_frame(self, timeout_ms: int = 1000, latest_only: bool = True) -> Optional[np.ndarray]:
        """Returns the next pre-recorded frame or a generated synthetic frame."""
        if not self._is_opened:
            return None

        # 1. If frame files are available, read sequentially in a loop
        if self._frame_files:
            file_path = self._frame_files[self._current_index % len(self._frame_files)]
            self._current_index += 1
            frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return frame

        # 2. Otherwise generate synthetic frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background dark gray
        frame[:] = (30, 30, 30)

        # Draw simulated board and test LED
        center_x, center_y = self.width // 2, self.height // 2
        cv2.circle(frame, (center_x, center_y), 40, (0, 255, 0), -1)  # Green LED
        cv2.putText(
            frame,
            f"MockCamera Frame {self._current_index}",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (200, 200, 200),
            2,
        )
        self._current_index += 1
        return frame

    @override
    def set_parameter(self, name: str, value: Any) -> bool:
        """Updates simulated parameter value."""
        self._parameters[name.lower()] = value
        return True

    @override
    def get_parameter(self, name: str) -> Any:
        """Retrieves simulated parameter value."""
        return self._parameters.get(name.lower())

    @override
    def load_configuration(self, config_path: str) -> bool:
        """Loads simulated parameters from JSON file."""
        abs_path = os.path.abspath(config_path)
        if not os.path.isfile(abs_path):
            return False

        self.config_path = abs_path
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(v, dict):
                        self.set_parameter(k, v)
            return True
        except Exception as e:
            logger.error("Failed to load mock config from %s: %s", abs_path, e)
            return False
