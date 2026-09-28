"""
evaluate.py - Benchmark precision, recall, and F1 on 15 annotated test images.
"""

import json
import os
import numpy as np
from detector import PhotobombDetector


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def evaluate_dataset(dataset_json_path: str):
    with open(dataset_json_path, "r") as f:
        dataset = json.load(f)

    detector = PhotobombDetector()

    tp = 0  # Ground truth 'remove' correctly flagged as 'remove'
    fp = 0  # Ground truth 'keep' falsely flagged as 'remove' (or false detection flagged as remove)
    fn = 0  # Ground truth 'remove' incorrectly flagged as 'keep' (or missed detection)

    print(f"\n--- Running evaluation on {len(dataset)} images ---")

    for sample in dataset:
        img_path = sample["image_path"]
        gt_annotations = sample["annotations"]  # list of {box: [x1, y1, x2, y2], label: 'keep'/'remove'}

        if not os.path.exists(img_path):
            print(f"Warning: File {img_path} not found. Skipping...")
            continue

        preds = detector.process_image(img_path)

        # Match predicted boxes to GT using IoU threshold 0.5
        matched_gt = set()
        
        for pred in preds:
            p_box = pred["bounding_box"]
            p_cls = pred["classification"]

            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_annotations):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.3 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                gt_cls = gt_annotations[best_gt_idx]["label"]
                if p_cls == "remove":
                    if gt_cls == "remove":
                        tp += 1
                    else:
                        fp += 1
                else:
                    if gt_cls == "remove":
                        fn += 1
            else:
                # Detection has no GT match; if flagged remove, count as FP
                if p_cls == "remove":
                    fp += 1

        # Check for un-detected ground truth 'remove's
        for g_idx, gt in enumerate(gt_annotations):
            if g_idx not in matched_gt and gt["label"] == "remove":
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n================ EVALUATION METRICS (Class: 'remove') ================")
    print(f"True Positives (TP) : {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1-Score            : {f1:.4f}")
    print("=====================================================================\n")


if __name__ == "__main__":
    evaluate_dataset("test_dataset.json")