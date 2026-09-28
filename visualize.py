"""
visualize.py - Draws detections and creates an animated GIF.
"""

import cv2
import json
import glob
from PIL import Image
from detector import PhotobombDetector


def annotate_image(image_path: str, detector: PhotobombDetector, output_path: str):
    image = cv2.imread(image_path)
    results = detector.process_image(image_path)

    for item in results:
        x1, y1, x2, y2 = map(int, item["bounding_box"])
        cls = item["classification"]
        conf = item["confidence_score"]
        subj = item["overall_subjectness"]

        color = (0, 220, 0) if cls == "keep" else (0, 0, 230)  # BGR: Green vs Red
        label = f"ID:{item['instance_id']} [{cls.upper()}] conf:{conf:.2f} s:{subj:.2f}"

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + w + 10, y1), color, -1)
        cv2.putText(image, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imwrite(output_path, image)
    return image


def create_demo_gif(annotated_dir: str, gif_path: str = "demo_results.gif"):
    # Include .jpeg along with .jpg and .png
    files = sorted(
        glob.glob(f"{annotated_dir}/*.jpg") + 
        glob.glob(f"{annotated_dir}/*.jpeg") + 
        glob.glob(f"{annotated_dir}/*.png")
    )
    if not files:
        print("No annotated images found to build GIF!")
        return
    frames = [Image.open(f).resize((800, 600)) for f in files]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1200, loop=0)
    print(f"Demo GIF successfully created at: {gif_path}")


if __name__ == "__main__":
    detector = PhotobombDetector()
    import os
    os.makedirs("annotated_outputs", exist_ok=True)
    
    with open("test_dataset.json") as f:
        dataset = json.load(f)

    for sample in dataset:
        p = sample["image_path"]
        if os.path.exists(p):
            fname = os.path.basename(p)
            out_p = os.path.join("annotated_outputs", fname)
            annotate_image(p, detector, out_p)

    create_demo_gif("annotated_outputs", "demo_results.gif")