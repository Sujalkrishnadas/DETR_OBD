"""
visualize_obb.py

Visualize DOTA Oriented Bounding Boxes (OBB)

Author: Sujal Krishna das
Project: Aircraft Research using DOTA
"""

from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------
# DATASET PATHS
# --------------------------------------------------------

IMAGE_PATH = Path("datasets/DOTA/train/images/P0187.png")
LABEL_PATH = Path("datasets/DOTA/train/labelTxt/P0187.txt")


# --------------------------------------------------------
# COLORS
# --------------------------------------------------------

CLASS_COLORS = {
    "plane": (0, 255, 0),
    "ship": (255, 0, 0),
    "storage-tank": (0, 255, 255),
    "baseball-diamond": (255, 255, 0),
    "tennis-court": (255, 0, 255),
    "basketball-court": (0, 165, 255),
    "ground-track-field": (128, 0, 255),
    "harbor": (255, 128, 0),
    "bridge": (255, 255, 255),
    "large-vehicle": (180, 105, 255),
    "small-vehicle": (255, 100, 100),
    "helicopter": (100, 255, 100),
    "roundabout": (100, 100, 255),
    "soccer-ball-field": (200, 200, 0),
    "swimming-pool": (0, 200, 200),
    "container-crane": (200, 0, 200),
}


# --------------------------------------------------------
# READ ANNOTATION
# --------------------------------------------------------

def read_annotation(label_path):

    objects = []

    with open(label_path, "r") as file:

        lines = file.readlines()

    # Skip metadata
    for line in lines[2:]:

        values = line.strip().split()

        if len(values) < 10:
            continue

        pts = np.array([
            [float(values[0]), float(values[1])],
            [float(values[2]), float(values[3])],
            [float(values[4]), float(values[5])],
            [float(values[6]), float(values[7])]
        ], dtype=np.int32)

        category = values[8]
        difficulty = int(values[9])

        objects.append({
            "polygon": pts,
            "category": category,
            "difficulty": difficulty
        })

    return objects


# --------------------------------------------------------
# DRAW OBB
# --------------------------------------------------------

def draw_objects(image, objects):

    for obj in objects:

        pts = obj["polygon"]
        category = obj["category"]
        difficulty = obj["difficulty"]

        color = CLASS_COLORS.get(category, (0, 255, 0))

        cv2.polylines(
            image,
            [pts],
            isClosed=True,
            color=color,
            thickness=2
        )

        x = int(pts[0][0])
        y = int(pts[0][1])

        label = f"{category} ({difficulty})"

        cv2.putText(
            image,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # Draw vertices
        for point in pts:

            cv2.circle(
                image,
                tuple(point),
                radius=4,
                color=(0, 0, 255),
                thickness=-1
            )

    return image


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

def main():

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    objects = read_annotation(LABEL_PATH)

    print(f"Objects Found : {len(objects)}")

    image = draw_objects(image, objects)

    cv2.imshow("DOTA OBB Visualization", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
