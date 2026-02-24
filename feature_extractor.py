import numpy as np
import math

def extract_features(detections, frame_shape):
    """
    Converts YOLO detections → numerical features
    """
    persons = []
    weapons = []

    for det in detections:
        if det["class"] == "person":
            persons.append(det)
        else:
            weapons.append(det)

    person_count = len(persons)
    weapon_count = len(weapons)
    weapon_present = 1 if weapon_count > 0 else 0

    # Average person bounding box area
    h, w = frame_shape[:2]
    areas = []

    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        area = ((x2 - x1) * (y2 - y1)) / (w * h)
        areas.append(area)

    avg_person_area = np.mean(areas) if areas else 0.0

    # Minimum distance between person and weapon
    min_distance = 1.0  # normalized

    for p in persons:
        px = (p["bbox"][0] + p["bbox"][2]) / 2
        py = (p["bbox"][1] + p["bbox"][3]) / 2

        for wpn in weapons:
            wx = (wpn["bbox"][0] + wpn["bbox"][2]) / 2
            wy = (wpn["bbox"][1] + wpn["bbox"][3]) / 2

            dist = math.dist((px, py), (wx, wy)) / max(frame_shape)
            min_distance = min(min_distance, dist)

    return {
        "person_count": person_count,
        "weapon_present": weapon_present,
        "weapon_count": weapon_count,
        "avg_person_area": round(avg_person_area, 4),
        "min_weapon_distance": round(min_distance, 4)
    }