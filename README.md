# Sapien Assignment
# 1. Photobomb Detection & Subject Classification Pipeline

An automated computer vision pipeline designed to detect human instances in photographs, extract spatial and visual cues, classify each person as either `"keep"` (intended subject) or `"remove"` (photobomber), and output structured JSON results alongside visual benchmark evaluations.

---

## Project Structure

```text
photobomber_pipeline/
├── detector.py             # Core pipeline: YOLOv8 object detection & signal scoring
├── evaluate.py             # Evaluation suite computing Precision, Recall, and F1
├── label_ground_truth.py   # Interactive GUI annotator to build test_dataset.json
├── visualize.py            # Generates annotated output images and demo GIF
├── requirements.txt        # Python dependencies
├── test_dataset.json       # Ground truth annotations for 15 test images
├── demo_results.gif        # Animated demonstration of pipeline results
└── README.md               # Architecture report and setup documentation

