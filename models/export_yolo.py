#!/usr/bin/env python3
"""Export annotations.json to Ultralytics YOLO dataset format.

Uses text-file mode: dataset.yaml points train/val at .txt files listing
absolute image paths, so no image copying or symlinking is needed.

Label .txt files are written ALONGSIDE the source images (same date-tree,
same filename, .txt extension). This matches how Ultralytics resolves label
paths when image paths have no /images/ component: it simply swaps the
file extension.

The train/val split is done by calendar day (--val-after DATE or --val-days N
counts back from the latest annotated day).

Usage:
    python export_yolo.py annotations.json F:/cameras/northcam F:/cameras/northcam_labels \
        --classes deer person dog --val-days 14

    python export_yolo.py annotations.json F:/cameras/northcam F:/cameras/northcam_labels \
        --classes deer person dog --val-after 2025-11-01 --image-size 1920x1080
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import yaml

from app.image_loader import timestamp_path_regex


def load_raw_annotations(path: Path) -> tuple[dict, dict]:
    """Return (labels, images).

    labels: {label_id_str: class_name}
    images: {image_key: [{"bbox": [x,y,w,h], "label": label_id_str}, ...]}
    """
    with open(path) as f:
        raw = json.load(f)
    labels = raw.get("labels", {})
    images = {
        k: v for k, v in raw.items() if k not in ("labels", "through") and isinstance(v, list)
    }
    return labels, images


def image_key_to_date(key: str) -> str:
    """Extract YYYY-MM-DD from a key like '2024/01/15/123456.jpg'."""
    m = timestamp_path_regex.match(key)
    if m:
        return f"{m['year']}-{m['month']}-{m['day']}"
    return ""


def detect_image_size(image_dir: Path, image_key: str) -> tuple[int, int]:
    img = cv.imread(str(image_dir / image_key))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_dir / image_key}")
    h, w = img.shape[:2]
    return w, h


def main():
    parser = argparse.ArgumentParser(
        description="Export annotations.json to Ultralytics YOLO format (text-file mode)"
    )
    parser.add_argument("annotations", type=Path, help="Path to annotations.json")
    parser.add_argument("image_dir", type=Path, help="Root directory of source images")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for train.txt, val.txt, dataset.yaml. "
             "Label .txt files go alongside the images in image_dir.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        metavar="CLASS",
        help="Class names to export in order (become YOLO class IDs 0, 1, ...)",
    )

    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument(
        "--val-after",
        metavar="YYYY-MM-DD",
        help="Images on or after this date go to val; earlier dates go to train",
    )
    split_group.add_argument(
        "--val-days",
        type=int,
        metavar="N",
        help="Use the last N calendar days of annotated data as val",
    )

    parser.add_argument(
        "--image-size",
        metavar="WxH",
        help="Image dimensions as WxH (e.g. 1920x1080). "
        "Omit to auto-detect from the first annotated image.",
    )
    args = parser.parse_args()

    labels_dict, images_dict = load_raw_annotations(args.annotations)

    if not images_dict:
        print("No annotated images found in annotations file.")
        sys.exit(1)

    # Resolve selected classes -> annotation label IDs
    name_to_annot_id = {name: lid for lid, name in labels_dict.items()}
    unknown = [c for c in args.classes if c not in name_to_annot_id]
    if unknown:
        print(f"WARNING: classes not found in annotations: {unknown}")
        print(f"  Available: {sorted(labels_dict.values())}")
    annot_id_to_yolo = {
        name_to_annot_id[c]: i for i, c in enumerate(args.classes) if c in name_to_annot_id
    }
    if not annot_id_to_yolo:
        print("No selected classes found. Nothing to export.")
        sys.exit(1)

    # Image size
    if args.image_size:
        img_w, img_h = (int(v) for v in args.image_size.split("x"))
        print(f"Using specified image size: {img_w}x{img_h}")
    else:
        first_key = next(iter(images_dict))
        print(f"Auto-detecting image size from {first_key} ...")
        img_w, img_h = detect_image_size(args.image_dir, first_key)
        print(f"  Detected: {img_w}x{img_h}")

    # Determine val cutoff date
    all_dates = sorted(set(image_key_to_date(k) for k in images_dict if image_key_to_date(k)))
    if not all_dates:
        print("Could not parse dates from image keys. Expected format: YYYY/MM/DD/HHMMSS.jpg")
        sys.exit(1)

    if args.val_after:
        val_cutoff = args.val_after
    else:
        if args.val_days >= len(all_dates):
            print(
                f"WARNING: --val-days {args.val_days} >= total days {len(all_dates)}; using last day only for val"
            )
            val_cutoff = all_dates[-1]
        else:
            val_cutoff = all_dates[-args.val_days]

    print(
        f"Val split: dates >= {val_cutoff}"
        f"  ({sum(d >= val_cutoff for d in all_dates)} val days,"
        f" {sum(d < val_cutoff for d in all_dates)} train days)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_images, val_images = [], []
    class_counts = {"train": defaultdict(int), "val": defaultdict(int)}
    skipped = 0

    for image_key, boxes in images_dict.items():
        lines = []
        for box in boxes:
            lid = str(box.get("label") if isinstance(box, dict) else box["label"])
            if lid in (None, "None") or lid not in annot_id_to_yolo:
                continue
            bbox = box["bbox"] if isinstance(box, dict) else box.bbox
            x, y, bw, bh = bbox
            yolo_cls = annot_id_to_yolo[lid]
            cx = max(0.0, min(1.0, (x + bw / 2) / img_w))
            cy = max(0.0, min(1.0, (y + bh / 2) / img_h))
            nw = max(0.0, min(1.0, bw / img_w))
            nh = max(0.0, min(1.0, bh / img_h))
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            skipped += 1
            continue

        # Write label file NEXT TO the source image so Ultralytics can find it.
        # Ultralytics resolves labels by swapping the extension when there is no
        # /images/ component in the path: image.jpg -> image.txt
        label_path = args.image_dir / Path(image_key).with_suffix(".txt")
        label_path.write_text("\n".join(lines) + "\n")

        # Assign to split
        date = image_key_to_date(image_key)
        split = "val" if date >= val_cutoff else "train"
        abs_image = str((args.image_dir / image_key).resolve())
        if split == "train":
            train_images.append(abs_image)
        else:
            val_images.append(abs_image)

        for line in lines:
            cls_idx = int(line.split()[0])
            class_counts[split][args.classes[cls_idx]] += 1

    # Write train.txt / val.txt
    (args.output_dir / "train.txt").write_text("\n".join(train_images) + "\n")
    (args.output_dir / "val.txt").write_text("\n".join(val_images) + "\n")

    # Write dataset.yaml
    yaml_path = args.output_dir / "dataset.yaml"
    dataset_yaml = {
        "path": str(args.output_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(args.classes),
        "names": args.classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone.")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images  : {len(val_images)}")
    print(f"  Skipped (no matching class): {skipped}")
    print(f"  Annotations by class:")
    for cls in args.classes:
        t = class_counts["train"][cls]
        v = class_counts["val"][cls]
        print(f"    {cls}: {t} train, {v} val")
    print(f"  Label files written alongside images in: {args.image_dir.resolve()}")
    print(f"  dataset.yaml: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
