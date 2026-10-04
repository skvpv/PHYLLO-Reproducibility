"""Dataset-level mIoU from per-image confusion contributions.

Frozen dual metric (LD-A5 / ASR-005 / Mask_Encoding_Report):
  * mIoU_116     — labels 0..115 INCLUDING background (released code path).
  * mIoU_disease — labels 1..115, background (0) EXCLUDED (scientific primary).

Matches mmseg IoUMetric semantics: IoU per class is computed on DATASET-LEVEL summed
areas (sum of intersections / sum of unions across images), then averaged over classes
whose union > 0. This is NOT a per-image mIoU average. Storing per-image
area_intersect / area_union / area_pred / area_label lets the paired bootstrap resum
over sampled images and recompute a dataset-level mIoU per replicate exactly as the
EBSAP requires ("recompute ... from summed sampled confusion contributions").

NUM_CLASSES = 116 (background + 115 disease); ignore_index 255 is configured but absent
from all released masks (Mask_Encoding_Report), so no pixel is ignored.
"""
from __future__ import annotations
import numpy as np

NUM_CLASSES = 116
IGNORE_INDEX = 255
DISEASE_LABELS = np.arange(1, 116)          # 1..115
ALL_LABELS = np.arange(0, 116)              # 0..115


def per_image_confusion(pred: np.ndarray, label: np.ndarray,
                        num_classes: int = NUM_CLASSES,
                        ignore_index: int = IGNORE_INDEX):
    """Return (area_intersect, area_union, area_pred, area_label), each shape (C,).

    pred, label: integer label maps of identical shape. Pixels equal to ignore_index
    in the label are excluded (there are none in the released masks, but we honour the
    configured semantics exactly).
    """
    pred = np.asarray(pred).reshape(-1)
    label = np.asarray(label).reshape(-1)
    if pred.shape != label.shape:
        raise ValueError("pred and label must have identical shape")
    valid = label != ignore_index
    pred, label = pred[valid], label[valid]
    intersect = pred[pred == label]
    area_intersect = np.bincount(intersect, minlength=num_classes)[:num_classes]
    area_pred = np.bincount(pred, minlength=num_classes)[:num_classes]
    area_label = np.bincount(label, minlength=num_classes)[:num_classes]
    area_union = area_pred + area_label - area_intersect
    return (area_intersect.astype(np.int64), area_union.astype(np.int64),
            area_pred.astype(np.int64), area_label.astype(np.int64))


def miou_from_sums(area_intersect: np.ndarray, area_union: np.ndarray,
                   labels: np.ndarray) -> float:
    """mIoU over the given class labels using summed areas.

    Classes with union == 0 (absent in both pred and label over the summed images)
    are excluded from the mean, matching mmseg's nan-mean behaviour.
    """
    inter = np.asarray(area_intersect, dtype=np.float64)[labels]
    union = np.asarray(area_union, dtype=np.float64)[labels]
    present = union > 0
    if not present.any():
        return float("nan")
    iou = inter[present] / union[present]
    return float(iou.mean())


def miou_116(area_intersect: np.ndarray, area_union: np.ndarray) -> float:
    return miou_from_sums(area_intersect, area_union, ALL_LABELS)


def miou_disease(area_intersect: np.ndarray, area_union: np.ndarray) -> float:
    return miou_from_sums(area_intersect, area_union, DISEASE_LABELS)


def per_class_iou(area_intersect: np.ndarray, area_union: np.ndarray) -> np.ndarray:
    inter = np.asarray(area_intersect, dtype=np.float64)
    union = np.asarray(area_union, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, np.nan)
    return iou


def macc_from_sums(area_intersect: np.ndarray, area_label: np.ndarray,
                   labels: np.ndarray) -> float:
    inter = np.asarray(area_intersect, dtype=np.float64)[labels]
    target = np.asarray(area_label, dtype=np.float64)[labels]
    present = target > 0
    if not present.any():
        return float("nan")
    return float(np.mean(inter[present] / target[present]))


def macc_116(area_intersect: np.ndarray, area_label: np.ndarray) -> float:
    return macc_from_sums(area_intersect, area_label, ALL_LABELS)


def macc_disease(area_intersect: np.ndarray, area_label: np.ndarray) -> float:
    return macc_from_sums(area_intersect, area_label, DISEASE_LABELS)


def boundary_confusion(pred: np.ndarray, label: np.ndarray, tolerance: int = 2):
    """Return aggregate disease-boundary TP/FP/FN for boundary F1/IoU.

    Boundaries are morphological gradients of the foreground-vs-background masks. A
    prediction boundary is a match when it falls within ``tolerance`` pixels of a GT
    boundary, and vice versa. This is the fixed implementation used for both metrics.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    p = np.asarray(pred) > 0
    y = np.asarray(label) > 0
    pb = binary_dilation(p) ^ binary_erosion(p)
    yb = binary_dilation(y) ^ binary_erosion(y)
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    y_near = binary_dilation(yb, structure=structure)
    p_near = binary_dilation(pb, structure=structure)
    tp_pred = int(np.logical_and(pb, y_near).sum())
    tp_gt = int(np.logical_and(yb, p_near).sum())
    # Symmetric matches are averaged to avoid favouring either boundary density.
    tp = (tp_pred + tp_gt) / 2.0
    fp = float(pb.sum()) - tp_pred
    fn = float(yb.sum()) - tp_gt
    return float(tp), max(0.0, fp), max(0.0, fn)


def boundary_scores(tp: float, fp: float, fn: float):
    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    return {
        "boundary_iou": float(tp / denom_iou) if denom_iou else float("nan"),
        "boundary_f1": float(2 * tp / denom_f1) if denom_f1 else float("nan"),
    }
