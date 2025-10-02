import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pprint

import cv2 as cv

from background_model import TimestampAwareBackgroundSubtractor
from image_loader import get_all_timestamped_files_sorted

ANNOTATION_FILE = "annotations.json"


def load_annotations(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"labels": {}}


def save_annotations(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def prompt_label_name(label_id):
    name = input(f"Enter name for label {label_id}: ")
    return name.strip()


def main():
    parser = argparse.ArgumentParser(description="Image annotation tool")
    parser.add_argument("image_dir", type=Path, help="Directory with images")
    args = parser.parse_args()

    image_dir = args.image_dir
    images = list(get_all_timestamped_files_sorted(image_dir))
    if not images:
        print("No images found in", image_dir)
        sys.exit(1)

    ann_path = image_dir / ANNOTATION_FILE
    annotations = load_annotations(ann_path)
    labels = annotations.setdefault("labels", {})

    next_not_annotated = next(
        time_and_file
        for time_and_file in images
        if str(time_and_file[1].relative_to(image_dir)) not in annotations
    )
    model = TimestampAwareBackgroundSubtractor(area_threshold=50 * 50)

    cv.namedWindow("Annotator", cv.WINDOW_AUTOSIZE)

    pprint(labels)
    for timestamp, filename in images:
        key = str(filename.relative_to(image_dir))
        if key in annotations and isinstance(annotations[key], list):
            if next_not_annotated[0] - timestamp < model._history_seconds:
                model.apply(cv.imread(str(filename)), timestamp)
            continue  # already annotated

        img = cv.imread(str(filename))

        mask, blobs = model.applyWithStats(img, timestamp)
        if not blobs:
            annotations[key] = []
            cv.imshow("Annotator", img)
            cv.waitKey(1)
            continue

        blob_infos = []
        for blob in blobs:
            display = img.copy()
            x, y, w, h = blob.bbox
            cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow("Annotator", display)
            k = cv.waitKey(0)
            if k == ord("q"):
                save_annotations(ann_path, annotations)
                cv.destroyAllWindows()
                print(f"Annotations saved to {ann_path}")
                sys.exit(0)
            elif ord("0") <= k <= ord("9"):
                label_id = chr(k)
                if label_id not in labels:
                    cv.destroyWindow("Annotator")
                    label_name = prompt_label_name(label_id)
                    labels[label_id] = label_name
                    pprint(labels)
                blob_infos.append({"bbox": blob.bbox.tolist(), "label": label_id})
            elif k == 27:  # ESC to skip this one
                continue
        annotations[key] = blob_infos
        save_annotations(ann_path, annotations)  # save progress

    cv.destroyAllWindows()
    save_annotations(ann_path, annotations)
    print(f"Annotations saved to {ann_path}")


if __name__ == "__main__":
    main()
