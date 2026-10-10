# Copyright (c) Facebook, Inc. and its affiliates.
"""
Custom Semantic Segmentation Evaluator with F1 Score.
Extends detectron2's SemSegEvaluator to add Mean F1 metric.
"""
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union

import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator

try:
    import cv2
    _CV2_IMPORTED = True
except ImportError:
    _CV2_IMPORTED = False


class SemSegEvaluatorWithF1(SemSegEvaluator):
    """
    Semantic Segmentation Evaluator with Mean F1 Score.
    
    Inherits from SemSegEvaluator and adds F1-related metrics:
    - mF1: Mean F1 score averaged across classes
    - F1-{class_name}: Per-class F1 score
    - Precision-{class_name}: Per-class Precision
    - Recall-{class_name}: Per-class Recall
    """

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics plus F1 score.
        
        Original metrics:
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        
        Added metrics:
        * Mean F1 score averaged across classes (mF1)
        * Per-class F1, Precision, Recall
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        
        
        # Calculate standard metrics using all valid classes
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.any(acc_valid) else 0.0
        
        # For mIoU, check if dataset specifies classes to exclude
        classes_to_exclude_from_miou = getattr(
            MetadataCatalog.get(self._dataset_name), 
            'classes_to_exclude_from_miou', 
            []
        )
        
        # Create mask for classes to include in mIoU
        miou_mask = iou_valid.copy()
        for exclude_idx in classes_to_exclude_from_miou:
            if exclude_idx < len(miou_mask):
                miou_mask[exclude_idx] = False
        
        # Calculate mIoU excluding specified classes
        miou = np.sum(iou[miou_mask]) / np.sum(miou_mask) if np.any(miou_mask) else 0.0
        
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        # Calculate F1 Score
        # F1 = 2 * precision * recall / (precision + recall)
        # precision = tp / pos_pred, recall = tp / pos_gt
        precision = np.full(self._num_classes, np.nan, dtype=float)
        recall = np.full(self._num_classes, np.nan, dtype=float)
        f1 = np.full(self._num_classes, np.nan, dtype=float)
        
        precision_valid = pos_pred > 0
        precision[precision_valid] = tp[precision_valid] / pos_pred[precision_valid]
        recall[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]  # recall = acc
        
        # F1 calculation: avoid division by zero
        f1_valid = np.logical_and(precision_valid, acc_valid)
        denominator = precision + recall
        f1_computable = np.logical_and(f1_valid, denominator > 0)
        f1[f1_computable] = 2 * precision[f1_computable] * recall[f1_computable] / denominator[f1_computable]
        
        # For classes where precision or recall is 0 but valid, F1 should be 0
        f1[np.logical_and(f1_valid, denominator == 0)] = 0.0
        
        # Mean F1: calculate for all valid classes
        mf1 = np.nanmean(f1[f1_valid]) if np.any(f1_valid) else 0.0

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mF1"] = 100 * mf1  # Add Mean F1
        
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]
        
        # Add per-class F1, Precision, Recall
        for i, name in enumerate(self._class_names):
            res[f"F1-{name}"] = 100 * f1[i] if not np.isnan(f1[i]) else 0.0
            res[f"Precision-{name}"] = 100 * precision[i] if not np.isnan(precision[i]) else 0.0
            res[f"Recall-{name}"] = 100 * recall[i] if not np.isnan(recall[i]) else 0.0

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
