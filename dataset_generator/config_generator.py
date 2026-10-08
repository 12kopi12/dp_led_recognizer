"""ROI configuration generator module.

Allows the user to interactively select a Region of Interest (ROI) on an input image
using mouse gestures (supporting rectangle and circle shapes) and saves the definition
to a JSON configuration file.
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ROIConfigGenerator:
    """Interactive GUI tool for selecting ROI shapes (rectangle or circle) on an image."""

    def __init__(
        self,
        image_path: str,
        output_json_path: str,
        initial_shape: str = "rectangle",
        max_disp_w: int = 1280,
        max_disp_h: int = 800,
    ) -> None:
        """Initialize ROI config generator.

        Args:
            image_path: Path to the source image.
            output_json_path: Path to save the resulting JSON configuration.
            initial_shape: Starting shape mode ('rectangle' or 'circle').
            max_disp_w: Maximum display width for scaling large images.
            max_disp_h: Maximum display height for scaling large images.
        """
        self.image_path = image_path
        self.output_json_path = output_json_path
        self.active_shape = initial_shape.lower() if initial_shape.lower() in ("rectangle", "circle") else "rectangle"

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Failed to decode image from: {image_path}")

        self.img_h, self.img_w = self.img.shape[:2]

        # Calculate display scale factor for fitting on screen
        self.scale = min(max_disp_w / float(self.img_w), max_disp_h / float(self.img_h), 1.0)
        self.disp_w = int(round(self.img_w * self.scale))
        self.disp_h = int(round(self.img_h * self.scale))

        # Mouse drawing state
        self.drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.current_pt: Optional[Tuple[int, int]] = None

        # Confirmed ROI geometry (stored in original image coordinates)
        self.confirmed_roi: Optional[Dict[str, Any]] = None

    def _mouse_callback(self, event: int, x_disp: int, y_disp: int, flags: int, param: Any) -> None:
        """Handle OpenCV mouse events with coordinate rescaling."""
        x_orig = int(round(x_disp / self.scale))
        y_orig = int(round(y_disp / self.scale))

        # Clamp to image boundaries
        x_orig = max(0, min(self.img_w - 1, x_orig))
        y_orig = max(0, min(self.img_h - 1, y_orig))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x_orig, y_orig)
            self.current_pt = (x_orig, y_orig)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_pt = (x_orig, y_orig)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.current_pt = (x_orig, y_orig)
                self._commit_current_shape()

    def _commit_current_shape(self) -> None:
        """Calculates and stores the drawn shape in full-resolution coordinates."""
        if self.start_pt is None or self.current_pt is None:
            return

        x1, y1 = self.start_pt
        x2, y2 = self.current_pt

        if self.active_shape == "rectangle":
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            width = xmax - xmin
            height = ymax - ymin

            if width < 3 or height < 3:
                logger.warning("Rectangle too small, ignored.")
                return

            self.confirmed_roi = {
                "shape": "rectangle",
                "coordinates": {
                    "x": int(xmin),
                    "y": int(ymin),
                    "width": int(width),
                    "height": int(height),
                    "bbox": [int(xmin), int(ymin), int(width), int(height)],
                },
                "image_size": {
                    "width": int(width),
                    "height": int(height),
                },
            }
            logger.info("Selected rectangle: x=%d, y=%d, w=%d, h=%d", xmin, ymin, width, height)

        elif self.active_shape == "circle":
            cx, cy = x1, y1
            radius = int(round(math.hypot(x2 - x1, y2 - y1)))

            if radius < 2:
                logger.warning("Circle radius too small, ignored.")
                return

            diameter = int(2 * radius)
            self.confirmed_roi = {
                "shape": "circle",
                "coordinates": {
                    "cx": int(cx),
                    "cy": int(cy),
                    "radius": int(radius),
                    "bbox": [int(cx - radius), int(cy - radius), diameter, diameter],
                },
                "image_size": {
                    "width": diameter,
                    "height": diameter,
                },
            }
            logger.info("Selected circle: center=(%d, %d), radius=%d (bbox size %dx%d)", cx, cy, radius, diameter, diameter)

    def _render(self) -> np.ndarray:
        """Render the image with overlay annotations and instructions."""
        canvas = self.img.copy()
        line_thick = max(2, int(round(2 / self.scale)))

        # Draw confirmed ROI
        if self.confirmed_roi is not None:
            shape_type = self.confirmed_roi["shape"]
            coords = self.confirmed_roi["coordinates"]

            if shape_type == "rectangle":
                x, y, w, h = coords["x"], coords["y"], coords["width"], coords["height"]
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), line_thick)
                cv2.drawMarker(canvas, (x + w // 2, y + h // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, line_thick)
                cv2.putText(
                    canvas,
                    f"Rect: {w}x{h} px",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 / self.scale,
                    (0, 255, 0),
                    line_thick,
                )

            elif shape_type == "circle":
                cx, cy, r = coords["cx"], coords["cy"], coords["radius"]
                cv2.circle(canvas, (cx, cy), r, (0, 255, 255), line_thick)
                # Bounding box of the circle
                cv2.rectangle(canvas, (cx - r, cy - r), (cx + r, cy + r), (128, 128, 128), max(1, line_thick // 2), cv2.LINE_AA)
                cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, line_thick)
                cv2.putText(
                    canvas,
                    f"Circle: r={r} (size {2*r}x{2*r})",
                    (cx - r, max(25, cy - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 / self.scale,
                    (0, 255, 255),
                    line_thick,
                )

        # Draw live dragged shape
        if self.drawing and self.start_pt and self.current_pt:
            x1, y1 = self.start_pt
            x2, y2 = self.current_pt
            preview_color = (0, 165, 255)  # Orange

            if self.active_shape == "rectangle":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), preview_color, line_thick, cv2.LINE_AA)
            elif self.active_shape == "circle":
                r = int(round(math.hypot(x2 - x1, y2 - y1)))
                cv2.circle(canvas, (x1, y1), r, preview_color, line_thick, cv2.LINE_AA)
                cv2.line(canvas, (x1, y1), (x2, y2), preview_color, 1, cv2.LINE_AA)

        # Resize canvas for display
        disp_canvas = cv2.resize(canvas, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)

        # Draw HUD bar
        bar_height = 55
        hud = disp_canvas[:bar_height, :].copy()
        cv2.rectangle(hud, (0, 0), (self.disp_w, bar_height), (20, 20, 20), -1)
        disp_canvas[:bar_height, :] = cv2.addWeighted(hud, 0.85, disp_canvas[:bar_height, :], 0.15, 0)

        status_text = f"Mode: [{self.active_shape.upper()}]  |  Image: {self.img_w}x{self.img_h} px"
        cv2.putText(disp_canvas, status_text, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        controls_text = "[R] Rectangle  |  [C] Circle  |  [Z] Clear  |  [ENTER/S] Save & Exit  |  [Q/ESC] Cancel"
        cv2.putText(disp_canvas, controls_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 240, 180), 1, cv2.LINE_AA)

        return disp_canvas

    def run_gui(self) -> Optional[Dict[str, Any]]:
        """Launch the interactive ROI selection GUI window.

        Returns:
            Dictionary with ROI definition if saved, or None if cancelled.
        """
        window_name = "ROI Config Generator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.disp_w, self.disp_h)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "=" * 60)
        print(" ROI CONFIG GENERATOR - INTERACTIVE SELECTION")
        print("=" * 60)
        print(f" Source Image : {self.image_path} ({self.img_w}x{self.img_h} px)")
        print(f" Output JSON  : {self.output_json_path}")
        print(" Hotkeys:")
        print("   [R]       : Switch to Rectangle mode")
        print("   [C]       : Switch to Circle mode")
        print("   [Z]       : Clear current ROI")
        print("   [ENTER/S] : Save configuration to JSON and exit")
        print("   [Q/ESC]   : Cancel and exit without saving")
        print("=" * 60 + "\n")

        saved = False
        while True:
            disp_frame = self._render()
            cv2.imshow(window_name, disp_frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("r"), ord("R")):
                self.active_shape = "rectangle"
                logger.info("Switched shape mode to: RECTANGLE")
            elif key in (ord("c"), ord("C")):
                self.active_shape = "circle"
                logger.info("Switched shape mode to: CIRCLE")
            elif key in (ord("z"), ord("Z")):
                self.confirmed_roi = None
                logger.info("Cleared current ROI selection.")
            elif key in (13, ord("s"), ord("S")):  # Enter or S
                if self.confirmed_roi is None:
                    logger.warning("No ROI marked! Please draw an ROI before saving.")
                else:
                    self.save_json(self.output_json_path)
                    saved = True
                    break
            elif key in (27, ord("q"), ord("Q")):  # ESC or Q
                logger.info("Cancelled without saving.")
                break

        cv2.destroyAllWindows()
        return self.confirmed_roi if saved else None

    def save_json(self, destination_path: str) -> None:
        """Saves current confirmed ROI to the target JSON path."""
        if self.confirmed_roi is None:
            raise ValueError("No ROI configured to save.")

        payload = {
            "image_source": os.path.basename(self.image_path),
            "shape": self.confirmed_roi["shape"],
            "coordinates": self.confirmed_roi["coordinates"],
            "image_size": self.confirmed_roi["image_size"],
        }

        os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
        with open(destination_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        logger.info("Saved ROI configuration to: %s", destination_path)
        print(f"\nConfiguration successfully saved to:\n  {destination_path}\n")
        print(json.dumps(payload, indent=2))


def set_roi_programmatically(
    image_path: str,
    output_json_path: str,
    shape: str,
    coords: Tuple[int, ...],
) -> Dict[str, Any]:
    """Helper for setting and saving ROI programmatically (headless / automated tests).

    Args:
        image_path: Path to source image.
        output_json_path: Path to target JSON file.
        shape: 'rectangle' or 'circle'.
        coords: (x, y, w, h) for rectangle, or (cx, cy, r) for circle.

    Returns:
        Generated configuration dictionary.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    shape_lower = shape.lower()
    if shape_lower == "rectangle":
        if len(coords) != 4:
            raise ValueError("Rectangle requires 4 coordinates: (x, y, width, height)")
        x, y, w, h = coords
        config = {
            "image_source": os.path.basename(image_path),
            "shape": "rectangle",
            "coordinates": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "bbox": [int(x), int(y), int(w), int(h)],
            },
            "image_size": {
                "width": int(w),
                "height": int(h),
            },
        }
    elif shape_lower == "circle":
        if len(coords) != 3:
            raise ValueError("Circle requires 3 coordinates: (cx, cy, radius)")
        cx, cy, r = coords
        diameter = int(2 * r)
        config = {
            "image_source": os.path.basename(image_path),
            "shape": "circle",
            "coordinates": {
                "cx": int(cx),
                "cy": int(cy),
                "radius": int(r),
                "bbox": [int(cx - r), int(cy - r), diameter, diameter],
            },
            "image_size": {
                "width": diameter,
                "height": diameter,
            },
        }
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    logger.info("Saved ROI configuration to: %s", output_json_path)
    return config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ROI config generator."""
    parser = argparse.ArgumentParser(
        description="ROI Configuration Generator: Interactively select or define ROI (rectangle or circle) and export to JSON."
    )
    parser.add_argument(
        "image_pos",
        nargs="?",
        default=None,
        help="Path to input reference image (positional alternative to --image)",
    )
    parser.add_argument(
        "output_pos",
        nargs="?",
        default=None,
        help="Path to output JSON configuration file (positional alternative to --output)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to input reference image (e.g. setting_1.jpg)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output JSON configuration file",
    )
    parser.add_argument(
        "--shape",
        "-s",
        type=str,
        default="rectangle",
        choices=["rectangle", "circle"],
        help="Initial ROI shape mode ('rectangle' or 'circle', default: rectangle)",
    )
    # Headless / scriptable options
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Programmatic rectangle specification without GUI",
    )
    parser.add_argument(
        "--circle",
        nargs=3,
        type=int,
        metavar=("CX", "CY", "R"),
        help="Programmatic circle specification without GUI",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for ROI configuration generation."""
    args = parse_args()

    image_path = args.image or args.image_pos
    output_path = args.output or args.output_pos

    if not image_path or not output_path:
        print("Error: Both input image and output JSON path must be specified.")
        print("Usage: python config_generator.py <image_path> <output_json_path> [options]")
        print("   or: python config_generator.py --image <image_path> --output <output_json_path> [options]")
        sys.exit(1)

    # Headless mode if coordinates are passed via CLI
    if args.bbox:
        set_roi_programmatically(image_path, output_path, "rectangle", tuple(args.bbox))
        return

    if args.circle:
        set_roi_programmatically(image_path, output_path, "circle", tuple(args.circle))
        return

    # Interactive GUI mode
    generator = ROIConfigGenerator(
        image_path=image_path,
        output_json_path=output_path,
        initial_shape=args.shape,
    )
    generator.run_gui()


if __name__ == "__main__":
    main()
