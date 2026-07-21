"""
dataset_stats.py

Analyze the DOTA dataset.

Outputs:
1. Number of annotation files
2. Number of objects
3. Class distribution
4. Difficulty distribution
5. CSV summary

Author: Sujal Krishna Das
"""

from pathlib import Path
from collections import Counter
import pandas as pd


# --------------------------------------------------------
# DATASET PATH
# --------------------------------------------------------

LABEL_DIR = Path("datasets/DOTA/train/labelTxt")

# --------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------

class_counter = Counter()
difficulty_counter = Counter()

total_files = 0
total_objects = 0

for txt_file in sorted(LABEL_DIR.glob("*.txt")):

    total_files += 1

    with open(txt_file, "r") as f:
        lines = f.readlines()

    # Skip metadata lines
    for line in lines[2:]:

        values = line.strip().split()

        if len(values) < 10:
            continue

        category = values[8]
        difficulty = int(values[9])

        class_counter[category] += 1
        difficulty_counter[difficulty] += 1
        total_objects += 1


# --------------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------------

print("=" * 60)
print("DOTA DATASET STATISTICS")
print("=" * 60)

print(f"Annotation Files : {total_files}")
print(f"Total Objects    : {total_objects}")

print("\nClass Distribution")
print("-" * 60)

for cls, count in class_counter.most_common():
    print(f"{cls:25s} {count}")

print("\nDifficulty Distribution")
print("-" * 60)

for diff, count in sorted(difficulty_counter.items()):
    print(f"Difficulty {diff}: {count}")


# --------------------------------------------------------
# SAVE CSV
# --------------------------------------------------------

results = []

for cls, count in class_counter.most_common():

    percentage = (count / total_objects) * 100

    results.append({
        "Class": cls,
        "Objects": count,
        "Percentage": round(percentage, 2)
    })

df = pd.DataFrame(results)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "dataset_statistics.csv"

df.to_csv(csv_path, index=False)

print("\nCSV Saved:")
print(csv_path.resolve())

print("=" * 60)