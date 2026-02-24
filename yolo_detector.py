from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # lightweight & fast

# COCO class IDs we care about
TARGET_CLASSES = {
    0: "person",
    43: "knife",
    44: "bottle",
    76: "scissors"
}

def detect_objects(frame):
    """
    Runs YOLOv8 on a single frame
    Returns filtered detections
    """
    results = model(frame, verbose=False)[0]

    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id in TARGET_CLASSES and confidence > 0.4:
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            detections.append({
                "class": TARGET_CLASSES[class_id],
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

    return detections