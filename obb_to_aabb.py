from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new DOTA aircraft AABB dataset for RT-DETR by copying images "
            "and converting YOLO-OBB labels to YOLO AABB labels."
        )
    )
    parser.add_argument(
        "--src",
        default="datasets/DOTA_aircraft_split",
        help="Source dataset root containing train/ and val/ splits.",
    )
    parser.add_argument(
        "--dst",
        default="datasets/DOTA_aircraft_AABB",
        help="Destination dataset root to create.",
    )
    parser.add_argument(
        "--class-names",
        default="plane",
        help="Comma-separated class names for dataset.yaml. E.g. 'plane'.",
    )
    return parser.parse_args()


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def find_image_files(directory: Path) -> Iterable[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def convert_obb_line(line: str) -> str:
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Invalid label line, expected 9 values but got {len(parts)}")

    cls = parts[0]
    coords = [float(x) for x in parts[1:]]
    xs = coords[0::2]
    ys = coords[1::2]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"


def safe_copy_images(src_dir: Path, dst_dir: Path) -> int:
    copied = 0
    if not src_dir.exists():
        print(f"Warning: source image directory missing: {src_dir}")
        return copied

    for image_path in find_image_files(src_dir):
        try:
            shutil.copy2(image_path, dst_dir / image_path.name)
            copied += 1
        except Exception as exc:
            print(f"Warning: failed to copy {image_path}: {exc}")
    return copied


def convert_label_file(src_label: Path, dst_label: Path) -> tuple[int, int]:
    converted = 0
    skipped = 0
    lines = src_label.read_text(encoding="utf-8").splitlines()

    output_lines = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            output_lines.append(convert_obb_line(stripped))
            converted += 1
        except Exception as exc:
            print(f"Skipped {src_label.name}:{lineno} -> {exc}")
            skipped += 1

    dst_label.write_text("".join(output_lines), encoding="utf-8")
    return converted, skipped


def write_dataset_yaml(dst_root: Path, class_names: list[str]) -> Path:
    yaml_path = dst_root / "dataset.yaml"
    dataset_path = dst_root.resolve()

    names_text = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_text = (
        f"path: {dataset_path.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n\n"
        "names:\n"
        f"{names_text}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> int:
    args = parse_args()
    src_root = Path(args.src).expanduser()
    dst_root = Path(args.dst).expanduser()

    if not src_root.exists():
        print(f"Error: source dataset root does not exist: {src_root}")
        return 1

    print(f"Source dataset: {src_root}")
    print(f"Destination dataset: {dst_root}\n")

    total_labels = 0
    total_label_lines = 0
    total_label_skips = 0
    total_copied_images = 0

    for split in ["train", "val"]:
        split_src = src_root / split
        split_dst = dst_root / split
        images_src = split_src / "images"
        labels_src = split_src / "labels"
        images_dst = split_dst / "images"
        labels_dst = split_dst / "labels"

        ensure_dir(images_dst)
        ensure_dir(labels_dst)

        print(f"Processing split: {split}")
        copied_images = safe_copy_images(images_src, images_dst)
        total_copied_images += copied_images
        print(f"  Copied images: {copied_images}")

        if not labels_src.exists():
            print(f"  Warning: labels directory missing: {labels_src}")
            continue

        label_files = sorted(labels_src.glob("*.txt"))
        if not label_files:
            print(f"  Warning: no label files found in {labels_src}")

        for label_path in label_files:
            dst_label_path = labels_dst / label_path.name
            converted, skipped = convert_label_file(label_path, dst_label_path)
            total_labels += 1
            total_label_lines += converted
            total_label_skips += skipped
            status = "ok" if skipped == 0 else f"{skipped} skipped"
            print(f"  Converted {label_path.name}: {converted} boxes ({status})")

        print()

    yaml_path = write_dataset_yaml(dst_root, [name.strip() for name in args.class_names.split(",") if name.strip()])

    print("Conversion finished successfully.\n")
    print("Summary:")
    print(f"  Dataset root: {dst_root}")
    print(f"  Splits processed: train, val")
    print(f"  Images copied: {total_copied_images}")
    print(f"  Label files converted: {total_labels}")
    print(f"  Bounding boxes converted: {total_label_lines}")
    print(f"  Label lines skipped: {total_label_skips}")
    print(f"  YAML file: {yaml_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
