# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# Potsdam dataset: 6 classes (0-5), ignore_label=6 for black boundary lines
POTSDAM_CATEGORIES = [
    {"name": "impervious_surface", "id": 0},
    {"name": "building", "id": 1},
    {"name": "low_vegetation", "id": 2},
    {"name": "tree", "id": 3},
    {"name": "car", "id": 4},
    {"name": "clutter", "id": 5},
]


def _get_potsdam_meta():
    stuff_classes = [k["name"] for k in POTSDAM_CATEGORIES]
    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_potsdam(root):
    root = os.path.join(root, "Potsdam")
    meta = _get_potsdam_meta()
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, "img_dir", dirname)
        gt_dir = os.path.join(root, "ann_dir", dirname)
        dataset_name = f"potsdam_sem_seg_{name}"
        DatasetCatalog.register(
            dataset_name, 
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(dataset_name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=6,  # Black boundary lines (RGB: 0,0,0) are labeled as 6
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_potsdam(_root)
