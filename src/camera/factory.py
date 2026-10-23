"""Factory Method implementation for instantiating camera drivers.

Allows dynamic selection of camera backend (IDS Peak, OpenCV, or Mock)
based on runtime configuration or test environment.
"""

from typing import Any, Dict, Optional

from .camera import ICamera
from .ids_camera import IDSCamera
from .mock_camera import MockCamera
from .opencv_camera import OpenCVCamera


class CameraFactory:
    """Factory creating camera driver instances implementing ICamera."""

    @staticmethod
    def create_camera(camera_type: str, config: Optional[Dict[str, Any]] = None) -> ICamera:
        """Instantiates a camera driver based on type string and configuration.

        Args:
            camera_type: Camera backend identifier ('ids', 'opencv', 'mock').
            config: Optional dictionary of keyword arguments passed to the camera constructor
                (e.g., config_path, camera_index, serial_number, frame_width, etc.).

        Returns:
            An instance implementing the ICamera interface.

        Raises:
            ValueError: If an unsupported camera type is requested.
        """
        cfg = config or {}
        normalized_type = camera_type.strip().lower()

        if normalized_type in ("ids", "idscamera", "ids_peak"):
            return IDSCamera(**cfg)
        elif normalized_type in ("opencv", "opencvcamera", "cv2", "webcam"):
            return OpenCVCamera(**cfg)
        elif normalized_type in ("mock", "mockcamera", "offline", "test"):
            return MockCamera(**cfg)
        else:
            supported = ["ids", "opencv", "mock"]
            raise ValueError(
                f"Unknown camera_type '{camera_type}'. Supported camera types: {supported}"
            )
