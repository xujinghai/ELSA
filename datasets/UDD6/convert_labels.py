"""
Convert UDD6 RGB labels to class index labels for semantic segmentation.

UDD6 Classes (6 classes):
    0: Other      - RGB(0, 0, 0)
    1: Facade     - RGB(102, 102, 156)
    2: Road       - RGB(128, 64, 128)
    3: Vegetation - RGB(107, 142, 35)
    4: Vehicle    - RGB(0, 0, 142)
    5: Roof       - RGB(70, 70, 70)
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# UDD6 color mapping: RGB -> class index
UDD6_COLOR_MAP = {
    (0, 0, 0): 0,        # Other
    (102, 102, 156): 1,  # Facade
    (128, 64, 128): 2,   # Road
    (107, 142, 35): 3,   # Vegetation
    (0, 0, 142): 4,      # Vehicle
    (70, 70, 70): 5,     # Roof
}


def rgb_to_class_index(rgb_label):
    """Convert RGB label image to class index image."""
    h, w = rgb_label.shape[:2]
    class_label = np.zeros((h, w), dtype=np.uint8)
    
    # Default to 255 (ignore) for unknown colors
    class_label.fill(255)
    
    for rgb, class_idx in UDD6_COLOR_MAP.items():
        # OpenCV uses BGR, so reverse RGB
        bgr = rgb[::-1]
        mask = np.all(rgb_label == bgr, axis=-1)
        class_label[mask] = class_idx
    
    return class_label


def convert_labels(input_dir, output_dir):
    """Convert all RGB labels in input_dir to class index labels in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    print(f"Converting {len(label_files)} labels from {input_dir} to {output_dir}")
    
    for label_file in tqdm(label_files):
        input_path = os.path.join(input_dir, label_file)
        output_path = os.path.join(output_dir, label_file)
        
        # Read RGB label
        rgb_label = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if rgb_label is None:
            print(f"Warning: Could not read {input_path}")
            continue
        
        # Convert to class index
        class_label = rgb_to_class_index(rgb_label)
        
        # Save as single-channel PNG
        cv2.imwrite(output_path, class_label)
    
    print(f"Conversion complete. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert UDD6 RGB labels to class indices')
    parser.add_argument('--dataset_path', type=str, 
                        default='datasets/UDD6',
                        help='Path to UDD6 dataset')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    
    # Convert train labels
    train_gt = os.path.join(dataset_path, 'train', 'gt')
    train_gt_idx = os.path.join(dataset_path, 'train', 'gt_idx')
    if os.path.exists(train_gt):
        convert_labels(train_gt, train_gt_idx)
    
    # Convert val labels
    val_gt = os.path.join(dataset_path, 'val', 'gt')
    val_gt_idx = os.path.join(dataset_path, 'val', 'gt_idx')
    if os.path.exists(val_gt):
        convert_labels(val_gt, val_gt_idx)


if __name__ == '__main__':
    main()
