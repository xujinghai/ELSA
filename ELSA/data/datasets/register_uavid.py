# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# UAVid dataset: 8 classes (0-7), ignore_label=255 for boundary
UAVID_CATEGORIES = [
    {"name": "building", "id": 0},
    {"name": "road", "id": 1},
    {"name": "tree", "id": 2},
    {"name": "low_vegetation", "id": 3},
    {"name": "moving_car", "id": 4},
    {"name": "static_car", "id": 5},
    {"name": "human", "id": 6},
    {"name": "clutter", "id": 7},
]


def _get_uavid_meta():
    stuff_classes = [k["name"] for k in UAVID_CATEGORIES]
    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_uavid(root):
    root = os.path.join(root, "UAVid")
    meta = _get_uavid_meta()
    for name, dirname in [("train", "uavid_train"), ("val", "uavid_val")]:
        image_dir = os.path.join(root, dirname, "Images")
        gt_dir = os.path.join(root, dirname, "Labels_processed")  # Use processed labels
        dataset_name = f"uavid_sem_seg_{name}"
        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # Boundary (RGB: 255,255,255) is label 255
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_uavid(_root)
