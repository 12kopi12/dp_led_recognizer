"""ROI (Region of Interest) domain models for LED recognition.

Defines the abstract base ROI class and concrete implementations (RectROI, CircleROI)
with boundary-safe cropping, circular masking, and JSON serialization.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class ROI(ABC):
    """Abstract base class representing a Region of Interest on an image.

    Subclasses must implement:
      - `crop(image)`: extracts the image region defined by the ROI.
      - `bbox`: returns the bounding box `(x, y, width, height)`.
      - `center`: returns `(cx, cy)`.
      - `area`: returns the area in pixels.
    """

    def __init__(
        self,
        label: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """Initialize base ROI properties.

        Args:
            label: Unique identifier or name of the ROI (e.g. 'pwr_led').
            tags: Optional tags/categories (e.g. ['status', 'critical']).
            description: Optional human-readable description.
        """
        self.label = label
        self.tags = list(tags) if tags is not None else []
        self.description = description

    @property
    @abstractmethod
    def shape(self) -> str:
        """Returns the shape identifier string ('rect' or 'circle')."""
        ...

    @property
    @abstractmethod
    def bbox(self) -> Tuple[int, int, int, int]:
        """Returns the bounding box as (x, y, width, height)."""
        ...

    @property
    @abstractmethod
    def center(self) -> Tuple[int, int]:
        """Returns the center coordinates as (cx, cy)."""
        ...

    @property
    @abstractmethod
    def area(self) -> float:
        """Returns the geometric area of the ROI in pixels."""
        ...

    @abstractmethod
    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crops and extracts this ROI from the provided image.

        Args:
            image: Source image (H x W or H x W x C NumPy array).

        Returns:
            Extracted image patch of fixed dimensions matching the ROI bounding box.
        """
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ROI to a dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any], label: Optional[str] = None) -> "ROI":
        """Polymorphic factory constructing a concrete ROI instance from standard dictionary.

        Args:
            data: Dictionary containing ROI parameters ('shape', 'geometry', etc.).
            label: Optional override for the ROI label.

        Returns:
            An instance of RectROI or CircleROI.

        Raises:
            ValueError: If the shape is unsupported or required geometry keys are missing.
        """
        effective_label = label or data.get("label", "")
        tags = data.get("tags", [])
        desc = data.get("description", "")

        shape = str(data.get("shape", "")).strip().lower()
        geom = data.get("geometry", {})
        if not isinstance(geom, dict):
            raise ValueError(f"Expected 'geometry' to be a dict, got {type(geom)}: {data}")

        if shape == "circle":
            if not all(k in geom for k in ("cx", "cy", "radius")):
                raise ValueError(f"Circle geometry missing required keys ('cx', 'cy', 'radius') in: {data}")

            return CircleROI(
                center_x=int(geom["cx"]),
                center_y=int(geom["cy"]),
                radius=int(geom["radius"]),
                label=effective_label,
                mask_outside=bool(data.get("mask_outside", True)),
                tags=tags,
                description=desc,
            )

        elif shape == "rect":
            if not all(k in geom for k in ("x", "y", "width", "height")):
                raise ValueError(f"Rect geometry missing required keys ('x', 'y', 'width', 'height') in: {data}")

            return RectROI(
                x=int(geom["x"]),
                y=int(geom["y"]),
                width=int(geom["width"]),
                height=int(geom["height"]),
                label=effective_label,
                tags=tags,
                description=desc,
            )

        else:
            raise ValueError(f"Unsupported ROI shape '{shape}'. Expected 'rect' or 'circle'.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(label='{self.label}', bbox={self.bbox})"


class RectROI(ROI):
    """Rectangular Region of Interest."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        label: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """Initialize RectROI.

        Args:
            x: Top-left X coordinate.
            y: Top-left Y coordinate.
            width: Width of the rectangle (must be > 0).
            height: Height of the rectangle (must be > 0).
            label: Unique identifier or name.
            tags: Optional tags/categories.
            description: Optional description.
        """
        super().__init__(label=label, tags=tags, description=description)
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive integers, got w={width}, h={height}")

        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    @property
    def shape(self) -> str:
        return "rect"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Safely crops the rectangle from image with boundary zero-padding.

        If the rectangle partially or entirely extends outside the frame, the
        returned array always preserves the exact (height, width) dimensions,
        filling out-of-boundary regions with zeros (black).

        Args:
            image: Source frame (2D or 3D NumPy array).

        Returns:
            Cropped image patch of shape (height, width) or (height, width, C).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image)}")

        img_h, img_w = image.shape[:2]
        num_channels = image.shape[2] if image.ndim == 3 else None

        if num_channels is not None:
            crop_canvas = np.zeros((self.height, self.width, num_channels), dtype=image.dtype)
        else:
            crop_canvas = np.zeros((self.height, self.width), dtype=image.dtype)

        # Intersect bounds
        x_min = self.x
        y_min = self.y
        x_max = self.x + self.width
        y_max = self.y + self.height

        img_x1 = max(0, x_min)
        img_y1 = max(0, y_min)
        img_x2 = min(img_w, x_max)
        img_y2 = min(img_h, y_max)

        dst_x1 = img_x1 - x_min
        dst_y1 = img_y1 - y_min
        dst_x2 = dst_x1 + (img_x2 - img_x1)
        dst_y2 = dst_y1 + (img_y2 - img_y1)

        if img_x2 > img_x1 and img_y2 > img_y1:
            if num_channels is not None:
                crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = image[img_y1:img_y2, img_x1:img_x2, :]
            else:
                crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[img_y1:img_y2, img_x1:img_x2]

        return crop_canvas

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "shape": "rect",
            "geometry": {
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
            },
            "bbox": [self.x, self.y, self.width, self.height],
            "tags": self.tags,
            "description": self.description,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RectROI):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
            and self.label == other.label
            and self.tags == other.tags
            and self.description == other.description
        )


class CircleROI(ROI):
    """Circular Region of Interest with optional background masking."""

    def __init__(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        label: str = "",
        mask_outside: bool = True,
        tags: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        """Initialize CircleROI.

        Args:
            center_x: Center X coordinate (cx).
            center_y: Center Y coordinate (cy).
            radius: Radius in pixels (must be > 0).
            label: Unique identifier or name.
            mask_outside: If True, pixels outside the circle are masked to pure black (0).
            tags: Optional tags/categories.
            description: Optional description.
        """
        super().__init__(label=label, tags=tags, description=description)
        if radius <= 0:
            raise ValueError(f"Radius must be a positive integer, got radius={radius}")

        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.radius = int(radius)
        self.mask_outside = bool(mask_outside)

    @property
    def cx(self) -> int:
        """Alias for center_x."""
        return self.center_x

    @property
    def cy(self) -> int:
        """Alias for center_y."""
        return self.center_y

    @property
    def shape(self) -> str:
        return "circle"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        diameter = 2 * self.radius
        return (self.center_x - self.radius, self.center_y - self.radius, diameter, diameter)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.center_x, self.center_y)

    @property
    def area(self) -> float:
        return float(math.pi * (self.radius ** 2))

    def crop(self, image: np.ndarray, mask: Optional[bool] = None) -> np.ndarray:
        """Safely crops the bounding square from image and optionally applies circular mask.

        Boundary conditions are safely zero-padded, always returning a patch
        of size (2*r, 2*r) or (2*r, 2*r, C).

        Args:
            image: Source frame (2D or 3D NumPy array).
            mask: Override for circular masking. If None, uses self.mask_outside.

        Returns:
            Cropped (and masked) square image patch of dimension (2*r, 2*r).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image)}")

        img_h, img_w = image.shape[:2]
        diameter = 2 * self.radius
        num_channels = image.shape[2] if image.ndim == 3 else None

        if num_channels is not None:
            crop_canvas = np.zeros((diameter, diameter, num_channels), dtype=image.dtype)
        else:
            crop_canvas = np.zeros((diameter, diameter), dtype=image.dtype)

        x_min = self.center_x - self.radius
        y_min = self.center_y - self.radius
        x_max = self.center_x + self.radius
        y_max = self.center_y + self.radius

        img_x1 = max(0, x_min)
        img_y1 = max(0, y_min)
        img_x2 = min(img_w, x_max)
        img_y2 = min(img_h, y_max)

        dst_x1 = img_x1 - x_min
        dst_y1 = img_y1 - y_min
        dst_x2 = dst_x1 + (img_x2 - img_x1)
        dst_y2 = dst_y1 + (img_y2 - img_y1)

        if img_x2 > img_x1 and img_y2 > img_y1:
            if num_channels is not None:
                crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2, :] = image[img_y1:img_y2, img_x1:img_x2, :]
            else:
                crop_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[img_y1:img_y2, img_x1:img_x2]

        apply_mask = self.mask_outside if mask is None else bool(mask)
        if apply_mask:
            circle_mask = np.zeros((diameter, diameter), dtype=np.uint8)
            cv2.circle(circle_mask, (self.radius, self.radius), self.radius, 255, -1)
            if num_channels is not None:
                crop_canvas = cv2.bitwise_and(crop_canvas, crop_canvas, mask=circle_mask)
            else:
                crop_canvas = cv2.bitwise_and(crop_canvas, circle_mask)

        return crop_canvas

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "shape": "circle",
            "geometry": {
                "cx": self.center_x,
                "cy": self.center_y,
                "radius": self.radius,
            },
            "bbox": list(self.bbox),
            "mask_outside": self.mask_outside,
            "tags": self.tags,
            "description": self.description,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CircleROI):
            return False
        return (
            self.center_x == other.center_x
            and self.center_y == other.center_y
            and self.radius == other.radius
            and self.mask_outside == other.mask_outside
            and self.label == other.label
            and self.tags == other.tags
            and self.description == other.description
        )
