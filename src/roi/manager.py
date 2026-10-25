"""ROI manager for persisting, organizing, and cropping multiple ROIs."""

from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    from .roi import CircleROI, RectROI, ROI
except (ImportError, ValueError):
    from roi import CircleROI, RectROI, ROI

logger = logging.getLogger(__name__)


class ROIManager:
    """Manages collections of ROIs, camera resolution validation, and config persistence."""

    def __init__(
        self,
        rois: Optional[Union[Dict[str, ROI], List[ROI]]] = None,
        reference_resolution: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ROIManager.

        Args:
            rois: Optional initial collection of ROIs (dict mapping label -> ROI or list of ROIs).
            reference_resolution: Optional (width, height) tuple specifying the expected camera resolution.
            metadata: Optional additional metadata dictionary.
        """
        self._rois: Dict[str, ROI] = {}
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}

        if reference_resolution is not None:
            self.set_reference_resolution(*reference_resolution)

        if rois is not None:
            if isinstance(rois, dict):
                for label, roi in rois.items():
                    if not roi.label:
                        roi.label = label
                    self.add_roi(roi)
            elif isinstance(rois, list):
                for roi in rois:
                    self.add_roi(roi)

    @property
    def rois(self) -> Dict[str, ROI]:
        """Returns the dictionary mapping labels to ROI objects."""
        return self._rois

    def add_roi(self, roi: ROI, overwrite: bool = True) -> None:
        """Adds an ROI to the manager.

        Args:
            roi: The ROI instance to register.
            overwrite: If False, raises ValueError if the label is already registered.

        Raises:
            ValueError: If label is missing, or already registered when overwrite=False.
        """
        if not roi.label:
            # Generate fallback unique label
            idx = 1
            while f"roi_{idx}" in self._rois:
                idx += 1
            roi.label = f"roi_{idx}"

        if not overwrite and roi.label in self._rois:
            raise ValueError(f"ROI with label '{roi.label}' already exists.")

        self._rois[roi.label] = roi

    def remove_roi(self, label: str) -> bool:
        """Removes an ROI by label.

        Args:
            label: Label of the ROI to remove.

        Returns:
            True if removed, False if label was not found.
        """
        if label in self._rois:
            del self._rois[label]
            return True
        return False

    def get_roi(self, label: str) -> Optional[ROI]:
        """Retrieves an ROI by label.

        Args:
            label: Label to look up.

        Returns:
            The ROI instance, or None if not found.
        """
        return self._rois.get(label)

    def has_roi(self, label: str) -> bool:
        """Checks if an ROI with the given label exists."""
        return label in self._rois

    def clear(self) -> None:
        """Removes all registered ROIs."""
        self._rois.clear()

    def list_labels(self) -> List[str]:
        """Returns a list of all registered ROI labels."""
        return list(self._rois.keys())

    def crop(self, label: str, image: np.ndarray) -> np.ndarray:
        """Crops a single ROI by its label.

        Args:
            label: Target ROI label.
            image: Source frame.

        Returns:
            Cropped image patch.

        Raises:
            KeyError: If label is not registered.
        """
        if label not in self._rois:
            raise KeyError(f"ROI '{label}' is not registered.")
        return self._rois[label].crop(image)

    def crop_all(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Crops all registered ROIs from the provided image.

        Args:
            image: Source frame (NumPy array).

        Returns:
            Dictionary mapping ROI labels to their cropped image patches.
        """
        return {label: roi.crop(image) for label, roi in self._rois.items()}

    def set_reference_resolution(self, width: int, height: int) -> None:
        """Sets the reference resolution in metadata."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Resolution dimensions must be positive integers, got w={width}, h={height}")
        self.metadata["reference_resolution"] = {"width": int(width), "height": int(height)}

    def get_reference_resolution(self) -> Optional[Tuple[int, int]]:
        """Returns the reference resolution as (width, height), or None if not set."""
        res = self.metadata.get("reference_resolution")
        if isinstance(res, dict) and "width" in res and "height" in res:
            return (int(res["width"]), int(res["height"]))
        if isinstance(res, (list, tuple)) and len(res) == 2:
            return (int(res[0]), int(res[1]))
        return None

    def validate_resolution(
        self,
        image_or_resolution: Union[np.ndarray, Tuple[int, int], Dict[str, int]],
    ) -> bool:
        """Validates that a frame or resolution matches the stored reference resolution.

        If no reference resolution is set in metadata, validation always returns True.

        Args:
            image_or_resolution: NumPy image array, (width, height) tuple, or {'width': ..., 'height': ...}.

        Returns:
            True if dimensions match or reference resolution is unset, False otherwise.
        """
        expected = self.get_reference_resolution()
        if expected is None:
            return True

        expected_w, expected_h = expected

        if isinstance(image_or_resolution, np.ndarray):
            actual_h, actual_w = image_or_resolution.shape[:2]
        elif isinstance(image_or_resolution, (list, tuple)) and len(image_or_resolution) >= 2:
            actual_w, actual_h = int(image_or_resolution[0]), int(image_or_resolution[1])
        elif isinstance(image_or_resolution, dict):
            actual_w = int(image_or_resolution.get("width", 0))
            actual_h = int(image_or_resolution.get("height", 0))
        else:
            raise TypeError(f"Unsupported type for resolution validation: {type(image_or_resolution)}")

        matches = (actual_w == expected_w) and (actual_h == expected_h)
        if not matches:
            logger.warning(
                "Resolution mismatch: expected %dx%d, got %dx%d",
                expected_w, expected_h, actual_w, actual_h,
            )
        return matches

    def save_config(
        self,
        path: str,
        rois: Optional[Dict[str, ROI]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves ROIs and metadata to a JSON configuration file.

        Args:
            path: Target JSON file path.
            rois: Optional override dictionary of ROIs to save (defaults to self.rois).
            metadata: Optional override metadata (defaults to self.metadata).
        """
        target_rois = self._rois if rois is None else rois
        target_meta = dict(self.metadata)
        if metadata:
            target_meta.update(metadata)

        # Ensure timestamp is recorded
        if "updated_at" not in target_meta:
            target_meta["updated_at"] = datetime.now(timezone.utc).isoformat()

        config_data = {
            "version": target_meta.get("version", "1.0"),
            "metadata": target_meta,
            "rois": {label: roi.to_dict() for label, roi in target_rois.items()},
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d ROI(s) to %s", len(target_rois), path)

    def load_config(self, path: str) -> Dict[str, ROI]:
        """Loads ROIs and metadata from a standard JSON configuration file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Dictionary mapping labels to loaded ROI objects.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file does not conform to the standard format.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "rois" not in data or not isinstance(data["rois"], dict):
            raise ValueError(f"Invalid ROI config format in {path}: expected top-level 'rois' dictionary.")

        self.metadata.update(data.get("metadata", {}))
        loaded_rois: Dict[str, ROI] = {}
        for label, roi_dict in data["rois"].items():
            loaded_rois[label] = ROI.from_dict(roi_dict, label=label)

        self._rois.update(loaded_rois)
        logger.info("Loaded %d ROI(s) from %s", len(loaded_rois), path)
        return self._rois

    # -------------------------------------------------------------------------
    # Collection Protocol Helpers
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rois)

    def __iter__(self) -> Iterator[str]:
        return iter(self._rois)

    def __getitem__(self, label: str) -> ROI:
        return self._rois[label]

    def __setitem__(self, label: str, roi: ROI) -> None:
        roi.label = label
        self.add_roi(roi, overwrite=True)

    def __delitem__(self, label: str) -> None:
        del self._rois[label]

    def __contains__(self, label: object) -> bool:
        return label in self._rois

    def __repr__(self) -> str:
        return f"ROIManager(rois={list(self._rois.keys())}, ref_res={self.get_reference_resolution()})"
