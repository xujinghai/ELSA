"""
Aeroscapes dataset preprocessing script.

Official Aeroscapes masks (SegmentationClass/*.png) are palettized PNGs whose
pixel values are already class indices (0-10, 255=unlabeled), so no
RGB-to-index conversion is needed. This script:
  1. Verifies the mask format (single-channel, valid class values).
  2. Splits the flat official layout (JPEGImages/ + SegmentationClass/ +
     ImageSets/trn.txt + val.txt) into train/val directories required by
     the load_sem_seg-based registration.

Output layout:
  datasets/Aeroscapes/train/JPEGImages + SegmentationClass
  datasets/Aeroscapes/val/JPEGImages + SegmentationClass
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

VALID_CLASSES = set(range(11))  # 0-10
IGNORE_LABEL = 255


def check_mask_format(mask_path):
    """Return (is_valid, invalid_values) for a class-index mask."""
    mask = np.array(Image.open(mask_path))
    unique = set(np.unique(mask).tolist())
    invalid = unique - VALID_CLASSES - {IGNORE_LABEL}
    return mask.ndim == 2 and not invalid, invalid


def read_split(split_file):
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def process_split(base_dir, split_name, file_stems):
    image_dir = os.path.join(base_dir, "JPEGImages")
    mask_dir = os.path.join(base_dir, "SegmentationClass")
    image_out = os.path.join(base_dir, split_name, "JPEGImages")
    mask_out = os.path.join(base_dir, split_name, "SegmentationClass")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    skipped = 0
    for stem in tqdm(file_stems, desc=f"Splitting {split_name}"):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(image_dir, stem + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        mask_path = os.path.join(mask_dir, stem + ".png")
        if img_path is None or not os.path.isfile(mask_path):
            skipped += 1
            continue

        valid, invalid = check_mask_format(mask_path)
        if not valid:
            raise ValueError(
                f"Mask {mask_path} is not a class-index image "
                f"(invalid values: {sorted(invalid)}). "
                "If masks are RGB color images, a color-to-index conversion is required first."
            )

        shutil.copy2(img_path, os.path.join(image_out, os.path.basename(img_path)))
        shutil.copy2(mask_path, os.path.join(mask_out, os.path.basename(mask_path)))

    if skipped:
        print(f"Warning: skipped {skipped} entries missing image or mask files.")


def main():
    parser = argparse.ArgumentParser(description="Split Aeroscapes into train/val directories.")
    parser.add_argument(
        "--dataset_path",
        default="datasets/Aeroscapes",
        help="Path to the official Aeroscapes dataset root.",
    )
    args = parser.parse_args()

    base_dir = args.dataset_path
    for split_name, split_file in [("train", "trn.txt"), ("val", "val.txt")]:
        split_path = os.path.join(base_dir, "ImageSets", split_file)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        file_stems = read_split(split_path)
        print(f"{split_name}: {len(file_stems)} entries from {split_path}")
        process_split(base_dir, split_name, file_stems)

    print("\nDone! Split datasets are ready under train/ and val/ directories.")


if __name__ == "__main__":
    main()
