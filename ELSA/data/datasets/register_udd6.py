# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# UDD6 dataset: 6 classes (0-5), no ignore_label (all pixels have valid class)
# Class 0 "Other" serves as background
UDD6_CATEGORIES = [
    {"name": "other", "id": 0},
    {"name": "facade", "id": 1},
    {"name": "road", "id": 2},
    {"name": "vegetation", "id": 3},
    {"name": "vehicle", "id": 4},
    {"name": "roof", "id": 5},
]


def _get_udd6_meta():
    stuff_classes = [k["name"] for k in UDD6_CATEGORIES]
    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_udd6(root):
    root = os.path.join(root, "UDD6")
    meta = _get_udd6_meta()
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, dirname, "src")
        gt_dir = os.path.join(root, dirname, "gt_idx")  # Use converted class index labels
        dataset_name = f"udd6_sem_seg_{name}"
        DatasetCatalog.register(
            dataset_name, 
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="JPG")
        )
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # Unmapped pixels are set to 255
            classes_to_exclude_from_miou=[0],  # Exclude 'other' class from mIoU calculation
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_udd6(_root)
