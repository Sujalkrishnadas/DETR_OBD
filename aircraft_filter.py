"""
aircraft_filter.py

Create a filtered DOTA dataset containing only selected classes.

Example:
    TARGET_CLASSES = {"plane"}

or

    TARGET_CLASSES = {"plane", "helicopter"}

Author: Sujal Krishna Das
"""

from pathlib import Path
import shutil

# ==========================================================
# CONFIGURATION
# ==========================================================

SOURCE_ROOT = Path("datasets/DOTA_split/val")

IMAGES_DIR = SOURCE_ROOT / "images"
LABELS_DIR = SOURCE_ROOT / "labelTxt"

OUTPUT_ROOT = Path("datasets/DOTA_aircraft_split/val")

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labelTxt"

# Change this whenever needed
TARGET_CLASSES = {
    "plane"
}

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# ==========================================================
# FILTER
# ==========================================================

images_copied = 0
labels_written = 0
objects_kept = 0
discarded_tiles=0

for label_file in sorted(LABELS_DIR.glob("*.txt")):

    with open(label_file, "r") as f:
        lines = f.readlines()

    header = lines[:2]
    objects = lines[2:]

    filtered_objects = []

    for obj in objects:

        values = obj.strip().split()

        if len(values) < 10:
            continue

        category = values[8]

        if category in TARGET_CLASSES:
            filtered_objects.append(obj)
            objects_kept += 1

    # Skip images without desired objects
    if len(filtered_objects) == 0:
        discarded_tiles += 1
        continue

    # Save annotation
    output_label = OUTPUT_LABELS / label_file.name

    with open(output_label, "w") as f:

        f.writelines(header)
        f.writelines(filtered_objects)

    labels_written += 1

    # Copy image
    image_name = label_file.stem + ".png"

    source_image = IMAGES_DIR / image_name
    destination_image = OUTPUT_IMAGES / image_name

    if source_image.exists():

        shutil.copy2(source_image, destination_image)
        images_copied += 1

# ==========================================================
# SUMMARY
# ==========================================================

print("=" * 70)
print("AIRCRAFT DATASET CREATED")
print("=" * 70)

print(f"Target Class        : {TARGET_CLASSES}")
print(f"Images Copied       : {images_copied}")
print(f"Labels Written      : {labels_written}")
print(f"Aircraft Objects    : {objects_kept}")
print(f"Discarded Tiles     : {discarded_tiles}")

print("\nDataset Location")
print(OUTPUT_ROOT.resolve())

print("=" * 70)