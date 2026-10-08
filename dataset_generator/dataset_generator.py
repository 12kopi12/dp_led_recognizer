"""Dataset generator module for CNN training and validation data.

Loads ROI configuration, controls camera parameters across defined variation ranges,
captures frames, crops Region of Interest (with circular masking if specified),
applies image augmentation (rotation, translation, noise, blur), and organizes
output images into training (train_data) and validation (test_data) directories.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Ensure module directory is in sys.path for direct script execution
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from camera_driver import CameraDriver
except ImportError:
    from .camera_driver import CameraDriver

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_roi_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates ROI configuration from JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Dictionary containing 'shape', 'coordinates', and 'image_size'.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "shape" not in config or "coordinates" not in config:
        raise ValueError(f"Invalid ROI config format in {config_path}. Missing 'shape' or 'coordinates'.")

    shape = str(config["shape"]).strip().lower()
    if shape not in ("rectangle", "circle"):
        raise ValueError(f"Unsupported shape '{shape}' in config. Must be 'rectangle' or 'circle'.")

    return config


def crop_roi(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Crops the specified ROI from the image.

    For circle ROIs, masks pixels outside the circle to pure black (0, 0, 0)
    and crops to the bounding box (2*r x 2*r).
    Handles edge boundary conditions safely with zero-padding.

    Args:
        image: Source image frame (H x W x C).
        config: ROI configuration dictionary.

    Returns:
        Cropped (and masked) image patch.
    """
    img_h, img_w = image.shape[:2]
    shape = str(config["shape"]).strip().lower()
    coords = config["coordinates"]

    if shape == "circle":
        # Extract center and radius
        if "cx" in coords and "cy" in coords:
            cx, cy = int(coords["cx"]), int(coords["cy"])
        elif "center" in coords:
            cx, cy = int(coords["center"][0]), int(coords["center"][1])
        else:
            raise ValueError("Circle config missing 'cx'/'cy' or 'center'.")

        r = int(coords["radius"])
        diameter = 2 * r

        # Bounding box coordinates
        x_min = cx - r
        y_min = cy - r
        x_max = cx + r
        y_max = cy + r

        # Canvas initialized with pure black
        crop = np.zeros((diameter, diameter, 3), dtype=np.uint8)

        # Overlapping intersection with image bounds
        img_x1 = max(0, x_min)
        img_y1 = max(0, y_min)
        img_x2 = min(img_w, x_max)
        img_y2 = min(img_h, y_max)

        dst_x1 = img_x1 - x_min
        dst_y1 = img_y1 - y_min
        dst_x2 = dst_x1 + (img_x2 - img_x1)
        dst_y2 = dst_y1 + (img_y2 - img_y1)

        if img_x2 > img_x1 and img_y2 > img_y1:
            crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[img_y1:img_y2, img_x1:img_x2]

        # Circular mask: 255 inside circle, 0 outside
        mask = np.zeros((diameter, diameter), dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)

        # Mask pixels outside circle to black
        crop_masked = cv2.bitwise_and(crop, crop, mask=mask)
        return crop_masked

    elif shape == "rectangle":
        if "bbox" in coords:
            x, y, w, h = [int(v) for v in coords["bbox"]]
        else:
            x = int(coords["x"])
            y = int(coords["y"])
            w = int(coords["width"])
            h = int(coords["height"])

        x_max = x + w
        y_max = y + h

        crop = np.zeros((h, w, 3), dtype=np.uint8)

        img_x1 = max(0, x)
        img_y1 = max(0, y)
        img_x2 = min(img_w, x_max)
        img_y2 = min(img_h, y_max)

        dst_x1 = img_x1 - x
        dst_y1 = img_y1 - y
        dst_x2 = dst_x1 + (img_x2 - img_x1)
        dst_y2 = dst_y1 + (img_y2 - img_y1)

        if img_x2 > img_x1 and img_y2 > img_y1:
            crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[img_y1:img_y2, img_x1:img_x2]

        return crop

    else:
        raise ValueError(f"Unsupported shape: {shape}")


def augment_image(image: np.ndarray, is_circle: bool = False) -> np.ndarray:
    """Applies data augmentation: rotation, slight shift, random noise, and blur.

    Args:
        image: Cropped ROI image patch (H x W x C).
        is_circle: Whether the ROI is a circle (ensures outer mask remains black).

    Returns:
        Augmented image patch.
    """
    h, w = image.shape[:2]
    aug = image.copy()

    # 1. Rotation & Slight Translation
    angle = random.uniform(-12.0, 12.0)
    max_shift_x = max(1.0, w * 0.05)
    max_shift_y = max(1.0, h * 0.05)
    dx = random.uniform(-max_shift_x, max_shift_x)
    dy = random.uniform(-max_shift_y, max_shift_y)

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 2] += dx
    rot_mat[1, 2] += dy

    border_mode = cv2.BORDER_CONSTANT if is_circle else cv2.BORDER_REFLECT_101
    aug = cv2.warpAffine(aug, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0))

    # 2. Random Blur (Gaussian or Box blur)
    blur_choice = random.random()
    if blur_choice < 0.45:
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.6, 1.4)
        aug = cv2.GaussianBlur(aug, (ksize, ksize), sigmaX=sigma)
    elif blur_choice < 0.70:
        aug = cv2.blur(aug, (3, 3))

    # 3. Random Noise (Gaussian noise)
    noise_sigma = random.uniform(3.0, 10.0)
    noise = np.random.normal(0, noise_sigma, aug.shape).astype(np.float32)
    aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. For circular ROI, strictly re-mask outside region to pure black (0, 0, 0)
    if is_circle:
        radius = min(w, h) // 2
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), radius, 255, -1)
        aug = cv2.bitwise_and(aug, aug, mask=mask)

    return aug


def get_target_folder_name(directory_path: str) -> str:
    """Extracts folder name from directory path (e.g. 'train_data/green' -> 'green')."""
    norm = os.path.normpath(directory_path)
    base = os.path.basename(norm)
    return base if base else "sample"


def generate_training_parameters(
    step_idx: int,
    total_steps: int,
    camera: Any,
    mode: str = "hybrid",
) -> Dict[str, float]:
    """Generates camera parameters for training step according to defined safe ranges.

    Ranges:
        brightness: [0, 20]
        contrast:   [10, 50]
        sharpness:  [0, 20]
        saturation: [80, 120]
        exposure:   [baseline - 1, baseline + 1]

    Args:
        step_idx: Current step index (1-based).
        total_steps: Total steps in the phase.
        camera: Camera driver or mock camera instance.
        mode: 'hybrid' (gradual progression + random jitter), 'gradual', or 'random'.

    Returns:
        Dictionary of parameter values.
    """
    if mode == "random":
        return camera.get_random_parameters()

    t = (step_idx - 1) / max(1, total_steps - 1)  # Normalized progression 0.0 -> 1.0

    # Base trajectories across the safe ranges
    # Multi-frequency oscillations ensure different parameters explore different combinations
    b_base = 0.0 + 20.0 * (0.5 + 0.5 * math.sin(2.0 * math.pi * t))
    c_base = 10.0 + 40.0 * t
    sh_base = 0.0 + 20.0 * (0.5 + 0.5 * math.cos(3.0 * math.pi * t))
    s_base = 80.0 + 40.0 * (0.5 + 0.5 * math.sin(4.0 * math.pi * t + math.pi / 4))

    if mode == "gradual":
        b = b_base
        c = c_base
        sh = sh_base
        s = s_base
    else:  # hybrid: gradual trend + random perturbation
        b = b_base + random.uniform(-4.0, 4.0)
        c = c_base + random.uniform(-6.0, 6.0)
        sh = sh_base + random.uniform(-4.0, 4.0)
        s = s_base + random.uniform(-8.0, 8.0)

    # Clamp to strict specification ranges
    b = float(max(0.0, min(20.0, round(b))))
    c = float(max(10.0, min(50.0, round(c))))
    sh = float(max(0.0, min(20.0, round(sh))))
    s = float(max(80.0, min(120.0, round(s))))

    params = {
        "brightness": b,
        "contrast": c,
        "sharpness": sh,
        "saturation": s,
    }

    # Exposure selection
    base_exp = getattr(camera, "baseline_exposure", -6.0)
    if base_exp is not None:
        if mode == "gradual":
            # Cycle through baseline - 1, baseline, baseline + 1
            exp_offset = [-1.0, 0.0, 1.0][step_idx % 3]
            params["exposure"] = float(base_exp + exp_offset)
        else:
            params["exposure"] = float(random.choice([base_exp - 1.0, base_exp, base_exp + 1.0]))

    return params


class MockCameraSource:
    """Mock camera source for simulation and testing when physical camera is unavailable."""

    def __init__(self, mock_image_path: str) -> None:
        self.mock_image_path = mock_image_path
        self.base_image = cv2.imread(mock_image_path)
        if self.base_image is None:
            raise FileNotFoundError(f"Mock image could not be loaded: {mock_image_path}")
        self.baseline_exposure = -6.0
        self.current_params = {
            "brightness": 0.0,
            "contrast": 30.0,
            "sharpness": 0.0,
            "saturation": 100.0,
            "exposure": -6.0,
        }

    def open(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def flush_buffer(self, count: Optional[int] = None) -> None:
        pass

    def reset_to_reference(self, flush: bool = True) -> None:
        self.current_params = {
            "brightness": 0.0,
            "contrast": 30.0,
            "sharpness": 0.0,
            "saturation": 100.0,
            "exposure": -6.0,
        }

    def get_random_parameters(self, include_exposure: bool = True) -> Dict[str, float]:
        return {
            "brightness": float(random.randint(0, 20)),
            "contrast": float(random.randint(10, 50)),
            "sharpness": float(random.randint(0, 20)),
            "saturation": float(random.randint(80, 120)),
            "exposure": float(random.choice([-7.0, -6.0, -5.0])),
        }

    def set_parameters(self, params: Dict[str, float], flush: bool = True) -> Dict[str, bool]:
        self.current_params.update(params)
        return {k: True for k in params}

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Simulate brightness, contrast, and saturation variations on base image
        img = self.base_image.copy().astype(np.float32)

        # Contrast and Brightness: output = alpha * input + beta
        alpha = float(self.current_params.get("contrast", 30.0)) / 30.0
        beta = float(self.current_params.get("brightness", 0.0))
        img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Saturation: convert to HSV, scale S channel
        sat_scale = float(self.current_params.get("saturation", 100.0)) / 100.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        simulated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Sharpness simulation
        sharp = float(self.current_params.get("sharpness", 0.0))
        if sharp > 0:
            blurred = cv2.GaussianBlur(simulated, (0, 0), 1.0)
            simulated = cv2.addWeighted(simulated, 1.0 + (sharp / 40.0), blurred, -(sharp / 40.0), 0)

        return True, simulated


def run_training_phase(
    camera: Any,
    roi_config: Dict[str, Any],
    train_dir: str,
    train_steps: int = 100,
    variation_mode: str = "hybrid",
    preview: bool = False,
) -> int:
    """Executes training dataset generation phase (100 steps).

    In each step:
      - Gradually / randomly modifies camera parameters within defined safe ranges:
          Brightness: [0, 20]
          Contrast:   [10, 50]
          Sharpness:  [0, 20]
          Saturation: [80, 120]
          Exposure:   baseline ± 1
      - Flushes buffer frames.
      - Captures frame and crops ROI (masked to black outside circle if circle).
      - Saves clean image: <folder_name>_{idx:03d}.jpg
      - Creates augmented image: <folder_name>_{idx:03d}_aug.jpg

    Args:
        camera: CameraDriver or MockCameraSource instance.
        roi_config: ROI geometry dictionary.
        train_dir: Target folder for training images.
        train_steps: Number of steps to generate (default: 100).
        variation_mode: 'hybrid', 'gradual', or 'random'.
        preview: Whether to display captured crops in real-time.

    Returns:
        Total number of images created (clean + augmented).
    """
    os.makedirs(train_dir, exist_ok=True)
    folder_name = get_target_folder_name(train_dir)
    is_circle = roi_config["shape"].lower() == "circle"
    saved_count = 0

    print("\n" + "=" * 65)
    print(f" STARTING TRAINING PHASE: {train_steps} steps (Mode: {variation_mode})")
    print(f" Target Folder: {train_dir} (Prefix: {folder_name})")
    print("=" * 65)

    start_time = time.time()
    for idx in range(1, train_steps + 1):
        # 1. Update camera parameters within defined ranges
        params = generate_training_parameters(idx, train_steps, camera, mode=variation_mode)
        camera.set_parameters(params)

        # 2. Capture frame from camera
        ret, frame = camera.read_frame()
        if not ret or frame is None:
            logger.error("Failed to capture frame at training step %d", idx)
            continue

        # 3. Crop ROI (with black circular mask if circle)
        roi_clean = crop_roi(frame, roi_config)

        # 4. Save clean image: <folder_name>_{idx:03d}.jpg
        clean_filename = f"{folder_name}_{idx:03d}.jpg"
        clean_path = os.path.join(train_dir, clean_filename)
        cv2.imwrite(clean_path, roi_clean, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 5. Create augmented version and save: <folder_name>_{idx:03d}_aug.jpg
        roi_aug = augment_image(roi_clean, is_circle=is_circle)
        aug_filename = f"{folder_name}_{idx:03d}_aug.jpg"
        aug_path = os.path.join(train_dir, aug_filename)
        cv2.imwrite(aug_path, roi_aug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        saved_count += 2

        if idx % 10 == 0 or idx == train_steps or idx == 1:
            exp_str = f"Exp:{params.get('exposure', 0):.1f}" if "exposure" in params else ""
            print(
                f"  [Train {idx:03d}/{train_steps}] Saved {clean_filename} & {aug_filename} | "
                f"B:{params.get('brightness', 0):.0f} C:{params.get('contrast', 0):.0f} "
                f"S:{params.get('saturation', 0):.0f} Sh:{params.get('sharpness', 0):.0f} {exp_str}"
            )

        if preview:
            combined = np.hstack([roi_clean, roi_aug])
            cv2.imshow("Train Preview: Clean (left) | Augmented (right)", combined)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                logger.info("Preview interrupted by user.")
                break

    if preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f" Training phase completed in {elapsed:.1f}s: {saved_count} images created in {train_dir}\n")
    return saved_count


def run_validation_phase(
    camera: Any,
    roi_config: Dict[str, Any],
    test_dir: str,
    test_steps: int = 20,
    preview: bool = False,
) -> int:
    """Executes validation / test dataset generation phase (20 steps).

    In each step:
      - Sets random combinations of parameters within the same safe ranges.
      - Discards buffer frames.
      - Captures frame and crops ROI (masked to black outside circle if circle).
      - Saves clean (unaugmented) image: <folder_name>_{idx:03d}.jpg

    Args:
        camera: CameraDriver or MockCameraSource instance.
        roi_config: ROI geometry dictionary.
        test_dir: Target folder for validation images.
        test_steps: Number of steps to generate (default: 20).
        preview: Whether to display captured crops in real-time.

    Returns:
        Total number of images created.
    """
    os.makedirs(test_dir, exist_ok=True)
    folder_name = get_target_folder_name(test_dir)
    saved_count = 0

    print("\n" + "=" * 65)
    print(f" STARTING VALIDATION PHASE: {test_steps} steps (Randomized combinations)")
    print(f" Target Folder: {test_dir} (Prefix: {folder_name})")
    print("=" * 65)

    start_time = time.time()
    for idx in range(1, test_steps + 1):
        # 1. Random combinations within the same range
        params = camera.get_random_parameters()
        camera.set_parameters(params)

        # 2. Capture frame from camera
        ret, frame = camera.read_frame()
        if not ret or frame is None:
            logger.error("Failed to capture frame at validation step %d", idx)
            continue

        # 3. Crop ROI (with black circular mask if circle)
        roi_clean = crop_roi(frame, roi_config)

        # 4. Save clean (unaugmented) image: <folder_name>_{idx:03d}.jpg
        clean_filename = f"{folder_name}_{idx:03d}.jpg"
        clean_path = os.path.join(test_dir, clean_filename)
        cv2.imwrite(clean_path, roi_clean, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        saved_count += 1

        if idx % 5 == 0 or idx == test_steps or idx == 1:
            exp_str = f"Exp:{params.get('exposure', 0):.1f}" if "exposure" in params else ""
            print(
                f"  [Test {idx:03d}/{test_steps}] Saved {clean_filename} | "
                f"B:{params.get('brightness', 0):.0f} C:{params.get('contrast', 0):.0f} "
                f"S:{params.get('saturation', 0):.0f} Sh:{params.get('sharpness', 0):.0f} {exp_str}"
            )

        if preview:
            cv2.imshow("Test Preview: Clean", roi_clean)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                logger.info("Preview interrupted by user.")
                break

    if preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f" Validation phase completed in {elapsed:.1f}s: {saved_count} images created in {test_dir}\n")
    return saved_count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset generator."""
    parser = argparse.ArgumentParser(
        description="Dataset Generator for CNN training (train_data) and validation (test_data) datasets."
    )
    # Positional alternatives
    parser.add_argument(
        "train_dir_pos",
        nargs="?",
        default=None,
        help="Target folder for training data (positional alternative to --train-dir)",
    )
    parser.add_argument(
        "test_dir_pos",
        nargs="?",
        default=None,
        help="Target folder for validation/test data (positional alternative to --test-dir)",
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        help="Path to JSON file containing ROI configuration (positional alternative to --config)",
    )
    # Named flags
    parser.add_argument(
        "--train-dir",
        "-t",
        type=str,
        default=None,
        help="Target folder for training data (e.g. dataset_generator/train_data/green)",
    )
    parser.add_argument(
        "--test-dir",
        "-v",
        type=str,
        default=None,
        help="Target folder for validation/test data (e.g. dataset_generator/test_data/green)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to JSON file containing ROI configuration",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="Camera device index (default: 1)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=100,
        help="Number of steps for the training phase (default: 100)",
    )
    parser.add_argument(
        "--test-steps",
        type=int,
        default=20,
        help="Number of steps for the validation phase (default: 20)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "random", "gradual"],
        help="Parameter variation mode for training phase: 'hybrid' (default), 'random', or 'gradual'",
    )
    parser.add_argument(
        "--buffer-flush",
        type=int,
        default=4,
        help="Number of frames to flush after setting change (default: 4, range 3-5)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display OpenCV preview window during dataset generation",
    )
    parser.add_argument(
        "--mock-image",
        type=str,
        default=None,
        help="Optional path to a reference image to simulate camera captures offline",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for dataset generator."""
    args = parse_args()

    train_dir = args.train_dir or args.train_dir_pos
    test_dir = args.test_dir or args.test_dir_pos
    config_path = args.config or args.config_pos

    if not train_dir or not test_dir or not config_path:
        print("\nError: train_dir, test_dir, and config must all be specified.")
        print("Usage: python dataset_generator.py <train_dir> <test_dir> <config_json> [options]")
        print("   or: python dataset_generator.py --train-dir <train_dir> --test-dir <test_dir> --config <config_json> [options]")
        sys.exit(1)

    # 1. Load ROI configuration
    logger.info("Loading ROI configuration from: %s", config_path)
    roi_config = load_roi_config(config_path)
    shape = roi_config["shape"]
    logger.info("Loaded ROI shape: %s, image_size: %s", shape, roi_config.get("image_size"))

    # 2. Initialize camera or mock source
    camera = None
    if args.mock_image:
        logger.info("Running in mock mode using image: %s", args.mock_image)
        camera = MockCameraSource(args.mock_image)
    else:
        camera = CameraDriver(
            camera_index=args.camera_index,
            buffer_flush_count=args.buffer_flush,
        )
        if not camera.open():
            logger.error(
                "Failed to open camera index %d. If testing on a laptop with built-in camera, try --camera-index 0 or use --mock-image.",
                args.camera_index,
            )
            sys.exit(1)

    try:
        # Reset camera to reference parameters at start
        camera.reset_to_reference()

        # 3. Training Phase (100 steps)
        run_training_phase(
            camera=camera,
            roi_config=roi_config,
            train_dir=train_dir,
            train_steps=args.train_steps,
            variation_mode=args.mode,
            preview=args.preview,
        )

        # 4. Validation Phase (20 steps)
        run_validation_phase(
            camera=camera,
            roi_config=roi_config,
            test_dir=test_dir,
            test_steps=args.test_steps,
            preview=args.preview,
        )

        # Reset camera back to reference settings before closing
        camera.reset_to_reference()
        logger.info("Dataset generation completed successfully.")

    except KeyboardInterrupt:
        logger.warning("\nExecution cancelled by user. Resetting camera...")
        try:
            camera.reset_to_reference()
        except Exception:
            pass
    finally:
        camera.release()


if __name__ == "__main__":
    main()
