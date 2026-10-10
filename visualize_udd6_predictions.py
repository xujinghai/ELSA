#!/usr/bin/env python3
"""
Visualization script for UDD6 semantic segmentation predictions.
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


# UDD6 color palette (BGR format for OpenCV) - extracted from standard UDD6 labels
# Original RGB values converted to BGR: (R, G, B) -> (B, G, R)
# Classes:
#   0: Other      - RGB(0, 0, 0)       -> BGR(0, 0, 0)
#   1: Facade     - RGB(102, 102, 156) -> BGR(156, 102, 102)
#   2: Road       - RGB(128, 64, 128)  -> BGR(128, 64, 128)
#   3: Vegetation - RGB(107, 142, 35)  -> BGR(35, 142, 107)
#   4: Vehicle    - RGB(0, 0, 142)     -> BGR(142, 0, 0)
#   5: Roof       - RGB(70, 70, 70)    -> BGR(70, 70, 70)
UDD6_PALETTE = {
    0: (0, 0, 0),        # other - black (RGB: 0, 0, 0)
    1: (156, 102, 102),  # facade - grayish purple (RGB: 102, 102, 156)
    2: (128, 64, 128),   # road - purple (RGB: 128, 64, 128)
    3: (35, 142, 107),   # vegetation - olive green (RGB: 107, 142, 35)
    4: (142, 0, 0),      # vehicle - dark blue (RGB: 0, 0, 142)
    5: (70, 70, 70),     # roof - dark gray (RGB: 70, 70, 70)
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
    
    # Get input images - UDD6 uses .JPG extension
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
        pred_color = class_to_color(pred_sem_seg, UDD6_PALETTE)
        
        # Load ground truth if available
        if args.gt:
            # Try different possible filename patterns
            gt_path = None
            # Try same name with .png extension
            for ext in ['.png', '.PNG']:
                possible_path = Path(args.gt) / (img_path.stem + ext)
                if possible_path.exists():
                    gt_path = possible_path
                    break
            
            if gt_path and gt_path.exists():
                gt = cv2.imread(str(gt_path))
                if gt is None:
                    print(f"Failed to load ground truth {gt_path}")
                    gt_vis = np.zeros_like(img)
                else:
                    # UDD6 labels are already in RGB color format
                    gt_vis = gt
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
        description="Visualize UDD6 semantic segmentation predictions"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., configs/udd6/semantic-segmentation/elsa_R50_bs16_40k.yaml)"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (e.g., output_UDD/model_final.pth)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input images directory (e.g., datasets/UDD6/val/src)"
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Path to ground truth labels directory (e.g., datasets/UDD6/val/gt)"
    )
    parser.add_argument(
        "--output",
        default="visualizations/udd6",
        help="Path to output directory for visualizations (default: visualizations/udd6)"
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
