"""
label_ground_truth.py - Interactive tool to annotate your 15 images in 2 minutes.
"""
import os
import glob
import json
import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
image_files = sorted(
    glob.glob("images/*.jpg") + 
    glob.glob("images/*.jpeg") + 
    glob.glob("images/*.png")
)

if not image_files:
    print("No images found in images/ folder! Put your 15 images there first.")
    exit()

dataset = []

print("\nControls:")
print("Press 'k' -> classify as 'keep' (intended subject)")
print("Press 'r' -> classify as 'remove' (photobomber)")
print("Press 's' -> skip detection (if it's a false positive)\n")

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue

    results = model(img, classes=[0], conf=0.3, verbose=False)[0]
    annotations = []

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Display image with current person highlighted
        display = img.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(display, f"Person {idx+1}: [k]eep or [r]emove?", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Ground Truth Annotator", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('k'):
            annotations.append({"box": [x1, y1, x2, y2], "label": "keep"})
        elif key == ord('r'):
            annotations.append({"box": [x1, y1, x2, y2], "label": "remove"})
        # Pressing 's' skips without adding to ground truth

    category = "clear_photobomb" if "clear" in img_path else ("crowd" if "crowd" in img_path else "clean_negative")
    dataset.append({
        "image_path": img_path,
        "category": category,
        "annotations": annotations
    })

cv2.destroyAllWindows()

with open("test_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\nSaved valid ground-truth annotations for {len(dataset)} images to test_dataset.json!")