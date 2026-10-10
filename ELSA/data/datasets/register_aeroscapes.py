# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# Aeroscapes dataset: 11 classes (0-10), ignore_label=255 for unlabeled pixels
AEROSCAPES_CATEGORIES = [
    {"name": "background", "id": 0},
    {"name": "person", "id": 1},
    {"name": "bike", "id": 2},
    {"name": "car", "id": 3},
    {"name": "drone", "id": 4},
    {"name": "boat", "id": 5},
    {"name": "animal", "id": 6},
    {"name": "obstacle", "id": 7},
    {"name": "construction", "id": 8},
    {"name": "vegetation", "id": 9},
    {"name": "road", "id": 10},
]


def _get_aeroscapes_meta():
    stuff_classes = [k["name"] for k in AEROSCAPES_CATEGORIES]
    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_aeroscapes(root):
    root = os.path.join(root, "Aeroscapes")
    meta = _get_aeroscapes_meta()
    for name, dirname in (("train", "train"), ("val", "val")):
        image_dir = os.path.join(root, dirname, "JPEGImages")
        gt_dir = os.path.join(root, dirname, "SegmentationClass")
        dataset_name = f"aeroscapes_sem_seg_{name}"
        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # Unlabeled pixels are set to 255
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_aeroscapes(_root)
