import cv2
import csv
import time

from ultralytics import YOLO
from feature_extractor import extract_features

# =========================
# LOAD YOLOv8 MODEL
# =========================
model = YOLO("yolov8n.pt")

# COCO classes we care about
TARGET_CLASSES = {
    0: "person",
    43: "knife",
    44: "bottle",
    76: "scissors"
}

# =========================
# OPEN LAPTOP WEBCAM
# =========================
cap = cv2.VideoCapture(0)  # 0 = laptop webcam

if not cap.isOpened():
    print("❌ ERROR: Webcam not accessible")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

# =========================
# CSV OUTPUT (LSTM INPUT)
# =========================
csv_file = open("output.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "frame_id",
    "person_count",
    "weapon_present",
    "weapon_count",
    "avg_person_area",
    "min_weapon_distance"
])

frame_id = 0
start_time = time.time()

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # -------------------------
    # YOLOv8 INFERENCE
    # -------------------------
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id in TARGET_CLASSES and confidence > 0.4:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": TARGET_CLASSES[class_id],
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

            # Draw bounding box (visual proof)
            color = (0, 255, 0) if TARGET_CLASSES[class_id] == "person" else (0, 0, 255)
            label = f"{TARGET_CLASSES[class_id]} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # -------------------------
    # FEATURE EXTRACTION
    # -------------------------
    features = extract_features(detections, frame.shape)

    # Write features to CSV (for LSTM)
    writer.writerow([
        frame_id,
        features["person_count"],
        features["weapon_present"],
        features["weapon_count"],
        features["avg_person_area"],
        features["min_weapon_distance"]
    ])

    # -------------------------
    # DISPLAY FEATURE INFO
    # -------------------------
    info_text = (
        f"Persons: {features['person_count']} | "
        f"Weapon: {features['weapon_present']} | "
        f"Min Dist: {features['min_weapon_distance']}"
    )

    cv2.putText(
        frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("YOLOv8 Crime Surveillance (Webcam)", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
csv_file.close()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
print(f"✅ Finished. Processed {frame_id} frames in {elapsed:.2f} seconds.")
print("📄 Features saved to output.csv (ready for LSTM).")