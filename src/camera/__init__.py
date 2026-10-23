"""Camera abstraction package.

Provides unified hardware camera drivers and factory for the LED recognizer pipeline.
"""

from .camera import Camera, ICamera
from .factory import CameraFactory
from .ids_camera import IDSCamera
from .mock_camera import MockCamera
from .opencv_camera import OpenCVCamera

__all__ = [
    "ICamera",
    "Camera",
    "IDSCamera",
    "OpenCVCamera",
    "MockCamera",
    "CameraFactory",
]
