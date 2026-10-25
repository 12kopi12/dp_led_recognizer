"""ROI (Region of Interest) management and annotation package.

Provides domain models (ROI, RectROI, CircleROI), storage manager (ROIManager),
and interactive GUI annotator (ROIAnnotator).
"""

from .annotator import ROIAnnotator
from .manager import ROIManager
from .roi import CircleROI, RectROI, ROI

__all__ = [
    "ROI",
    "RectROI",
    "CircleROI",
    "ROIManager",
    "ROIAnnotator",
]
