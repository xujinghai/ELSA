#!/usr/bin/env python3
"""
Visualization script for Aeroscapes semantic segmentation predictions.
Generates side-by-side comparisons of original images, ground truth, and predictions.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

# ELSA imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elsa import add_maskformer2_config


# Aeroscapes color palette (BGR format for OpenCV)
# Original RGB values converted to BGR: (R, G, B) -> (B, G, R)
# Classes (11 classes, 0-10):
#   0: Background     - RGB(0,   0,   0)     -> BGR(0,   0,   0)
#   1: Person         - RGB(192, 128, 128)   -> BGR(128, 128, 192)
#   2: Bike           - RGB(0,   128, 0)     -> BGR(0,   128, 0)
#   3: Car            - RGB(128, 128, 128)   -> BGR(128, 128, 128)
#   4: Drone          - RGB(128, 0,   0)     -> BGR(0,   0,   128)
#   5: Boat           - RGB(0,   0,   128)   -> BGR(128, 0,   0)
#   6: Animal         - RGB(192, 0,   128)   -> BGR(128, 0,   192)
#   7: Obstacle       - RGB(192, 0,   0)     -> BGR(0,   0,   192)
#   8: Construction   - RGB(192, 128, 0)    -> BGR(0,   128, 192)
#   9: Vegetation     - RGB(0,   128, 128)   -> BGR(128, 128, 0)
#  10: Road           - RGB(128, 64,  128)   -> BGR(128, 64,  128)
AEROSCAPES_PALETTE = {
    0: (0, 0, 0),          # background - black
    1: (128, 128, 192),    # person - pink
    2: (0, 128, 0),        # bike - green
    3: (128, 128, 128),    # car - gray
    4: (0, 0, 128),        # drone - dark red
    5: (128, 0, 0),        # boat - dark blue
    6: (128, 0, 192),      # animal - purple
    7: (0, 0, 192),        # obstacle - red
    8: (0, 128, 192),      # construction - orange
    9: (128, 128, 0),      # vegetation - olive
    10: (128, 64, 128),    # road - purple
}


def setup_cfg(args):
    """Setup configuration from config file and command line arguments."""
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


def class_to_color(class_map, palette):
    """Convert class indices to RGB color image."""
    h, w = class_map.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in palette.items():
        mask = class_map == class_id
        color_map[mask] = color

    return color_map


def visualize_predictions(args):
    """Main visualization function."""
    # Setup
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create separate directory for prediction-only images
    pred_only_dir = output_dir / "predictions"
    pred_only_dir.mkdir(parents=True, exist_ok=True)

    # Get input images - Aeroscapes uses .jpg extension
    image_dir = Path(args.input)
    image_files = sorted(
        list(image_dir.glob('*.png')) +
        list(image_dir.glob('*.jpg')) +
        list(image_dir.glob('*.JPG'))
    )

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load original image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load {img_path}, skipping...")
            continue

        # Run prediction
        outputs = predictor(img)
        pred_sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()

        # Convert prediction to color
        pred_color = class_to_color(pred_sem_seg, AEROSCAPES_PALETTE)

        # Load ground truth if available
        if args.gt:
            # Aeroscapes masks are class-index PNGs (single channel)
            gt_path = None
            for ext in ['.png', '.PNG']:
                possible_path = Path(args.gt) / (img_path.stem + ext)
                if possible_path.exists():
                    gt_path = possible_path
                    break

            if gt_path and gt_path.exists():
                gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                if gt is None:
                    print(f"Failed to load ground truth {gt_path}")
                    gt_vis = np.zeros_like(img)
                else:
                    # Convert class-index mask to color image
                    gt_vis = class_to_color(gt, AEROSCAPES_PALETTE)
            else:
                print(f"Ground truth not found for: {img_path.name}")
                gt_vis = np.zeros_like(img)
        else:
            gt_vis = np.zeros_like(img)

        # Resize all to same height for concatenation
        h, w = img.shape[:2]
        gt_vis = cv2.resize(gt_vis, (w, h))
        pred_color = cv2.resize(pred_color, (w, h))

        # Create visualization: [Original | Ground Truth | Prediction]
        vis = np.concatenate([img, gt_vis, pred_color], axis=1)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)

        cv2.putText(vis, 'Original', (10, 30), font, font_scale, color, thickness)
        cv2.putText(vis, 'Ground Truth', (w + 10, 30), font, font_scale, color, thickness)
        cv2.putText(vis, 'Prediction', (2*w + 10, 30), font, font_scale, color, thickness)

        # Save combined visualization
        output_path = output_dir / (img_path.stem + '.png')
        cv2.imwrite(str(output_path), vis)

        # Save prediction-only image separately
        pred_only_path = pred_only_dir / (img_path.stem + '.png')
        cv2.imwrite(str(pred_only_path), pred_color)

    print(f"\nVisualization complete!")
    print(f"Combined visualizations saved to: {output_dir}")
    print(f"Prediction-only images saved to: {pred_only_dir}")
    print(f"Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Aeroscapes semantic segmentation predictions"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., configs/Aeroscapes/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml)"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (e.g., output_aeroscapes/model_0039999.pth)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input images directory (e.g., datasets/Aeroscapes/val/JPEGImages)"
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Path to ground truth labels directory (e.g., datasets/Aeroscapes/val/SegmentationClass)"
    )
    parser.add_argument(
        "--output",
        default="visualizations/aeroscapes",
        help="Path to output directory for visualizations (default: visualizations/aeroscapes)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to visualize (default: all)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    if args.gt and not os.path.exists(args.gt):
        print(f"Warning: Ground truth directory not found: {args.gt}")

    visualize_predictions(args)


if __name__ == "__main__":
    main()
