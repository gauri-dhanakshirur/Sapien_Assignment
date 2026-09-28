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

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
Clone or download the repository, navigate to the root directory, and install the required packages:
pip install -r requirements.txt

Usage Instructions

Step 1: Prepare Test Dataset & Ground Truth
Place your 15 test images inside the images/ directory.
Launch the interactive annotation tool:
python label_ground_truth.py

Controls:
Press k to mark a highlighted person as keep (intended subject).
Press r to mark a highlighted person as remove (photobomber).
Press s to skip false detection hits.
This outputs the ground truth dataset to test_dataset.json.

Step 2: Single Image Detection Pipeline
To process a single photograph and print structured JSON results to the console:
python detector.py images/clear_pb_01.jpeg

Step 3: Run Evaluation Benchmark
To test the pipeline against all 15 annotated images and compute performance metrics for the remove class:
python evaluate.py

Step 4: Generate Visual Outputs & GIF
To create annotated output images (annotated_outputs/) and build an animated summary GIF (demo_results.gif):
python visualize.py

1. Full report- Sapien-Photobomber Pipeline Report.pdf
2. Visualisation gif- demo_results.gif
