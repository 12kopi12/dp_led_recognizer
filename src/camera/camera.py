"""Hardware camera abstraction interface.

Defines the unified ICamera interface that all camera drivers (IDS, OpenCV,
Mock) implement to ensure consistent frame acquisition, parameter tuning,
and configuration loading across runtime and testing environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class ICamera(ABC):
    """Abstract base interface for all camera implementations.

    Provides a unified contract for:
      - Connecting and disconnecting hardware (`open`, `close`, `is_opened`)
      - Frame acquisition returning standard BGR numpy arrays (`get_frame`)
      - Hardware queue flushing and latency compensation (`flush_buffers`)
      - Runtime parameter adjustments (`set_parameter`, `get_parameter`)
      - Camera configuration loading from files (`load_configuration`, e.g. .cset, .json)
      - Pythonic context management (`with camera: ...`)
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize base camera attributes.

        Args:
            config_path: Optional path to a camera configuration file
                (e.g., .cset for IDS cameras, .json for OpenCV/Mock cameras).
            **kwargs: Implementation-specific options (device index, serial number, etc.).
        """
        self.config_path: Optional[str] = config_path

    @abstractmethod
    def open(self) -> bool:
        """Opens connection to the camera hardware and starts acquisition pipeline.

        If a configuration path was provided at initialization, it should be
        applied automatically during open().

        Returns:
            True if the camera was successfully opened and started, False otherwise.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Stops frame acquisition and safely releases all camera resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Checks if the camera device is currently open and ready to capture frames.

        Returns:
            True if the camera is open and active, False otherwise.
        """
        pass

    @abstractmethod
    def get_frame(self, timeout_ms: int = 1000, latest_only: bool = True) -> Optional[np.ndarray]:
        """Captures a single frame from the camera stream.

        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds.
            latest_only: If True, discards older buffered frames that accumulated
                while the application was idle, returning the freshest possible frame.

        Returns:
            Captured frame as a BGR uint8 NumPy array (H x W x 3) suitable for
            OpenCV processing, or None if acquisition timed out or failed.
        """
        pass

    def flush_buffers(self, discard_count: int = 2) -> None:
        """Flushes queued and in-flight frames to ensure subsequent frames reflect recent changes.

        Default implementation is a no-op; drivers with internal queues (IDS, OpenCV)
        override this to drain stale hardware buffers.

        Args:
            discard_count: Number of transition frames to discard to allow sensor registers
                to settle.
        """
        pass

    @abstractmethod
    def set_parameter(self, name: str, value: Any) -> bool:
        """Sets a camera configuration parameter.

        Args:
            name: Parameter name (normalized name such as 'exposure', 'gain',
                or camera-specific node/property name).
            value: Target value to set.

        Returns:
            True if the parameter was successfully set, False otherwise.
        """
        pass

    @abstractmethod
    def get_parameter(self, name: str) -> Any:
        """Retrieves the current value of a camera parameter.

        Args:
            name: Parameter name (normalized or camera-specific).

        Returns:
            Current parameter value, or None if the parameter does not exist or is unavailable.
        """
        pass

    @abstractmethod
    def load_configuration(self, config_path: str) -> bool:
        """Loads and applies camera parameters from an external configuration file.

        For IDS cameras, this corresponds to GenApi persistence files (.cset).
        For OpenCV/Mock cameras, this corresponds to JSON configuration files.

        Args:
            config_path: Absolute or relative path to the configuration file.

        Returns:
            True if the configuration was successfully loaded and applied, False otherwise.
        """
        pass

    def __enter__(self) -> "ICamera":
        """Context manager entry point: opens the camera."""
        if not self.open():
            raise RuntimeError(f"Failed to open camera: {self.__class__.__name__}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point: closes the camera."""
        self.close()


# Backward compatibility alias
Camera = ICamera
