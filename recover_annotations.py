# Say app.annotator crashed and the annotations.json file is corrupted. Recover
# an annotations.json file from saved ultralytics-style annotations (which
# will be missing some classes, but oh well...)
import argparse
from collections import defaultdict
from pathlib import Path

import imagesize
from tqdm.auto import tqdm
from ultralytics.data.utils import check_det_dataset

from app.annotator import AnnotationFile
from app.utils import BoundingBox


def iter_yolo_annotations(ds: Path, root: Path):
    with open(ds, "r") as f:
        list_of_images = [l.strip() for l in f.readlines()]
    for image_file in tqdm(map(Path, list_of_images), total=len(list_of_images)):
        assert image_file.exists(), str(image_file)
        annot_file = image_file.with_suffix(".txt")
        assert annot_file.exists(), str(annot_file)
        img_w, img_h = imagesize.get(image_file)
        with open(annot_file, "r") as f:
            boxes_txt = [l.strip() for l in f.readlines()]
        for box in boxes_txt:
            label, cx, cy, w, h = box.split(" ")
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            yield str(image_file.relative_to(root)), BoundingBox(
                x=int(round((cx - w / 2) * img_w)),
                y=int(round((cy - h / 2) * img_h)),
                width=int(round(w * img_w)),
                height=int(round(h * img_h)),
                class_id=label,
            )


def yolo_to_annotations(yolo_dataset: dict, root: Path) -> AnnotationFile:
    annot = AnnotationFile()
    annot.labels = yolo_dataset["names"]
    labels_dict = defaultdict(list)

    through = ""
    for img, box in iter_yolo_annotations(yolo_dataset["train"], root):
        labels_dict[img].append(box)
        through = max(through, img)

    for img, box in iter_yolo_annotations(yolo_dataset["val"], root):
        labels_dict[img].append(box)
        through = max(through, img)

    annot.images = dict(labels_dict)
    annot.through = through

    return annot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yolo_dataset", type=Path, help="Path to YOLO dataset")
    parser.add_argument("destination", type=Path, help="Path to destination directory")
    args = parser.parse_args()

    out_file = args.destination / "annotations_recovery.json"
    assert args.yolo_dataset.exists()
    assert not out_file.exists()

    yolo_dataset = check_det_dataset(args.yolo_dataset)

    annot = yolo_to_annotations(yolo_dataset, root=args.destination)
    annot.save(out_file, through_key=sorted(annot.images.keys())[-1])


if __name__ == "__main__":
    main()
