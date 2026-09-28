"""
detector.py - Photobomb Detection and Subject Classification Pipeline
"""

import cv2
import numpy as np
import json
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO


class PhotobombDetector:
    def __init__(self, model_name: str = "yolov8m.pt", conf_thresh: float = 0.20):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

        # Tuned weights
        self.weights = {
            "scale": 0.45,
            "center_dist": 0.20,
            "sharpness": 0.15,
            "baseline": 0.20
        }
        self.keep_threshold = 0.62

    def _compute_sharpness(self, crop: np.ndarray) -> float:
        """Compute sharpness using variance of Laplacian."""
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(np.clip(var / 500.0, 0.0, 1.0))

    def _extract_signals(self, boxes: List[Tuple[float, float, float, float]], 
                         img_w: int, img_h: int, image_bgr: np.ndarray) -> List[Dict[str, float]]:
        if not boxes:
            return []

        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        max_area = max(areas) if areas else 1.0

        sharpness_scores = []
        for (x1, y1, x2, y2) in boxes:
            ix1, iy1, ix2, iy2 = map(int, [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)])
            crop = image_bgr[iy1:iy2, ix1:ix2]
            sharpness_scores.append(self._compute_sharpness(crop))

        max_sharpness = max(sharpness_scores) if max(sharpness_scores) > 0 else 1.0

        signals_list = []
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        max_dist = np.sqrt(img_cx**2 + img_cy**2)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            y_bottom = y2

            # Linear relative scale ratio
            scale_ratio = areas[i] / max_area

            # Center proximity
            dist_to_center = np.sqrt((cx - img_cx)**2 + (cy - img_cy)**2)
            center_signal = max(0.0, 1.0 - (dist_to_center / max_dist))

            # Sharpness normalized against the sharpest detected person
            sharp_signal = sharpness_scores[i] / max_sharpness if max_sharpness > 0 else 0.0

            # Baseline depth (lower in image usually correlates with foreground)
            baseline_signal = np.clip(y_bottom / img_h, 0.0, 1.0)

            signals = {
                "scale_score": round(float(scale_ratio), 4),
                "center_proximity": round(float(center_signal), 4),
                "sharpness": round(float(sharp_signal), 4),
                "baseline_depth": round(float(baseline_signal), 4)
            }
            signals_list.append(signals)

        return signals_list

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        img_h, img_w = image_bgr.shape[:2]

        results = self.model(image_bgr, classes=[0], conf=self.conf_thresh, verbose=False)[0]

        detected_boxes = []
        det_confs = []
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            detected_boxes.append(coords)
            det_confs.append(conf)

        if not detected_boxes:
            return []

        signals_list = self._extract_signals(detected_boxes, img_w, img_h, image_bgr)

        output = []
        for idx, (box, det_conf, signals) in enumerate(zip(detected_boxes, det_confs, signals_list)):
            subjectness = (
                signals["scale_score"] * self.weights["scale"] +
                signals["center_proximity"] * self.weights["center_dist"] +
                signals["sharpness"] * self.weights["sharpness"] +
                signals["baseline_depth"] * self.weights["baseline"]
            )

            classification = "keep" if subjectness >= self.keep_threshold else "remove"

            if classification == "keep":
                confidence = float(np.clip((subjectness - self.keep_threshold) / (1.0 - self.keep_threshold + 1e-6), 0.5, 1.0))
            else:
                confidence = float(np.clip((self.keep_threshold - subjectness) / (self.keep_threshold + 1e-6), 0.5, 1.0))

            output.append({
                "instance_id": idx + 1,
                "bounding_box": [round(c, 2) for c in box],
                "classification": classification,
                "confidence_score": round(confidence, 4),
                "overall_subjectness": round(float(subjectness), 4),
                "signals_used": signals
            })

        return output


if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "images/clean_01.jpeg"
    detector = PhotobombDetector()
    results = detector.process_image(test_img)
    print(json.dumps(results, indent=2))