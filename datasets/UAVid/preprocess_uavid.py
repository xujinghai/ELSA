"""
UAVid dataset preprocessing script.
Converts RGB label images to single-channel class index images.
"""
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# UAVid color to label mapping
COLOR_MAP = {
    (128, 0, 0): 0,      # Building
    (128, 64, 128): 1,   # Road
    (0, 128, 0): 2,      # Tree
    (128, 128, 0): 3,    # LowVeg
    (64, 0, 128): 4,     # Moving_Car
    (192, 0, 192): 5,    # Static_Car
    (64, 64, 0): 6,      # Human
    (0, 0, 0): 7,        # Clutter
    (255, 255, 255): 255 # Boundary (ignored)
}


def rgb_to_label(rgb_image):
    """Convert RGB label image to single-channel class indices."""
    h, w = rgb_image.shape[:2]
    label_image = np.zeros((h, w), dtype=np.uint8)
    
    for color, label_id in COLOR_MAP.items():
        # Find pixels matching this color
        mask = np.all(rgb_image == color, axis=2)
        label_image[mask] = label_id
    
    return label_image


def process_directory(input_dir, output_dir):
    """Process all label images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    label_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"Processing {len(label_files)} label images from {input_dir}...")
    
    for filename in tqdm(label_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read RGB label image
        rgb_label = np.array(Image.open(input_path))
        
        # Convert to single-channel indices
        label_indices = rgb_to_label(rgb_label)
        
        # Save as PNG
        Image.fromarray(label_indices).save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert UAVid RGB labels to class-index labels."
    )
    parser.add_argument(
        "--dataset_path",
        default="datasets/UAVid",
        help="Path to the UAVid dataset root.",
    )
    args = parser.parse_args()

    base_dir = args.dataset_path

    # Process training labels
    train_labels_in = os.path.join(base_dir, 'uavid_train', 'Labels')
    train_labels_out = os.path.join(base_dir, 'uavid_train', 'Labels_processed')
    process_directory(train_labels_in, train_labels_out)
    
    # Process validation labels
    val_labels_in = os.path.join(base_dir, 'uavid_val', 'Labels')
    val_labels_out = os.path.join(base_dir, 'uavid_val', 'Labels_processed')
    process_directory(val_labels_in, val_labels_out)
    
    print("\nDone! Processed labels saved to *_processed directories.")


if __name__ == '__main__':
    main()
