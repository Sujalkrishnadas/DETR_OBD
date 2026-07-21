"""
parse_annotation.py

Parses a single DOTA annotation (.txt) file and displays
all oriented bounding box (OBB) information.

Author: Sidharth V
Project: Aircraft Research using DOTA
"""

from pathlib import Path


def parse_annotation(annotation_file: Path):
    """
    Parse a DOTA annotation file.

    Parameters
    ----------
    annotation_file : Path
        Path to a DOTA .txt annotation file.

    Returns
    -------
    list
        List of dictionaries containing object annotations.
    """

    objects = []

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    # Skip the first two metadata lines
    for line in lines[2:]:

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) < 10:
            continue

        points = [
            (float(values[0]), float(values[1])),
            (float(values[2]), float(values[3])),
            (float(values[4]), float(values[5])),
            (float(values[6]), float(values[7]))
        ]

        category = values[8]
        difficulty = int(values[9])

        objects.append({
            "points": points,
            "category": category,
            "difficulty": difficulty
        })

    return objects


def print_annotation(objects):
    """
    Pretty-print parsed annotation.
    """

    print("=" * 70)
    print(f"Objects Found : {len(objects)}")
    print("=" * 70)

    for idx, obj in enumerate(objects, start=1):

        print(f"\nObject {idx}")

        print(f"Class      : {obj['category']}")
        print(f"Difficulty : {obj['difficulty']}")

        print("Vertices")

        for i, point in enumerate(obj["points"], start=1):
            print(f" P{i}: {point}")


if __name__ == "__main__":

    annotation_path = Path(
        "datasets/DOTA/train/labelTxt/P0000.txt"
    )

    objects = parse_annotation(annotation_path)

    print_annotation(objects)