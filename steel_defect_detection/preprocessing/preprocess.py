"""
Shared preprocessing pipeline used at BOTH training-data-prep time and inference
time, so the model always sees images processed the same way.

Kept deliberately conservative (light denoise, moderate CLAHE) — see
docs/case_study_analysis.md Part 4 for why aggressive preprocessing risks erasing
subtle defect texture (crazing, hairline scratches).
"""

import cv2
import numpy as np
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Apply the standard preprocessing pipeline to a single image.

    Args:
        image: BGR or grayscale image as loaded by cv2.imread
        config: parsed config.yaml dict (loaded automatically if None)

    Returns:
        Preprocessed image (grayscale, CLAHE-enhanced, denoised, resized)
    """
    if config is None:
        config = load_config()
    cfg = config["preprocessing"]

    # 1. Grayscale
    if cfg.get("grayscale", True) and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Light denoise BEFORE contrast enhancement (avoid amplifying noise with CLAHE)
    if cfg.get("denoise", True):
        strength = cfg.get("denoise_strength", 3)
        strength = strength if strength % 2 == 1 else strength + 1  # must be odd
        method = cfg.get("denoise_method", "median")
        
        if method == "gaussian":
            image = cv2.GaussianBlur(image, (strength, strength), 0)
        else:
            image = cv2.medianBlur(image, strength)

    # 3. CLAHE contrast enhancement — boosts subtle low-contrast defects
    if cfg.get("apply_clahe", True):
        clahe = cv2.createCLAHE(
            clipLimit=cfg.get("clahe_clip_limit", 2.0),
            tileGridSize=tuple(cfg.get("clahe_tile_grid_size", [8, 8])),
        )
        image = clahe.apply(image)

    # 4. Resize to model input size
    target_size = tuple(cfg.get("resize", [640, 640]))
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert back to 3-channel for YOLO (which expects RGB-like input)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def preprocess_file(input_path: str, output_path: str, config: dict = None):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    processed = preprocess_image(image, config)
    cv2.imwrite(output_path, processed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a single image (CLAHE + denoise + resize)")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save preprocessed image")
    args = parser.parse_args()

    preprocess_file(args.input, args.output)
    print(f"Saved preprocessed image to {args.output}")
