import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from pprint import pprint

import cv2 as cv
import numpy as np
import yaml

from background_model import TimestampAwareBackgroundSubtractor, ForegroundBlob
from classifier import featurize, FEATURE_NAMES
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


def get_annotations(image, blobs, labels_dict, prior_annots):
    if len(blobs) == 0:
        cv.imshow("Annotator", image)
        cv.waitKey(1)
        return []

    blob_infos = []
    for blob in blobs:
        # Check if this blob matches any prior annotation (by IoU)
        matched_prior = None
        best_iou = 0.0
        for prior in prior_annots:
            iou = ForegroundBlob.iou(np.array(prior["bbox"]), blob.bbox)
            if iou > best_iou:
                best_iou = iou
                matched_prior = prior
        if matched_prior is not None and best_iou > 0.5:
            blob_infos.append(
                {"bbox": blob.bbox.tolist(), "label": matched_prior["label"]}
            )
            continue
        elif matched_prior is not None:
            print(f"NEAREST [IOU={best_iou:.2f}]", matched_prior)


        display = image.copy()
        x, y, w, h = blob.bbox
        cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow("Annotator", display)
        k = cv.waitKey(0)
        if k == ord("q"):
            return None
        elif ord("0") <= k <= ord("9"):
            label_id = chr(k)
            if label_id not in labels_dict:
                cv.destroyWindow("Annotator")
                label_name = prompt_label_name(label_id)
                labels_dict[label_id] = label_name
                pprint(labels_dict)
            blob_infos.append({"bbox": blob.bbox.tolist(), "label": label_id})

    return blob_infos


def iter_files_with_flags(annotations, image_dir):
    for time_and_file in get_all_timestamped_files_sorted(image_dir):
        timestamp, filename = time_and_file
        key = str(filename.relative_to(image_dir))
        needs_annotation = key not in annotations or not isinstance(
            annotations[key], list
        )
        needs_features = needs_annotation
        if not needs_annotation and len(annotations[key]) > 0:
            for info in annotations[key]:
                if "features" not in info or len(info["features"]) != len(FEATURE_NAMES):
                    needs_features = True
                    break

        yield timestamp, filename, needs_annotation, needs_features


def main():
    parser = argparse.ArgumentParser(description="Image annotation tool")
    parser.add_argument("image_dir", type=Path, help="Directory with images")
    parser.add_argument(
        "bg_model_config", type=Path, help="Path to background model config"
    )
    parser.add_argument(
        "--prior-annotations",
        default=None,
        type=Path,
        help="Path to existing annotations to default to",
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    images = list(get_all_timestamped_files_sorted(image_dir))
    if not images:
        print("No images found in", image_dir)
        sys.exit(1)

    prior_annotations = {}
    if args.prior_annotations is not None:
        if not args.prior_annotations.exists():
            print("Prior annotations file not found:", args.prior_annotations)
            sys.exit(1)
        with open(args.prior_annotations, "r") as f:
            prior_annotations = json.load(f)

    if not args.bg_model_config.exists():
        print("Background model config file not found:", args.bg_model_config)
        sys.exit(1)

    with open(args.bg_model_config, "r") as f:
        bg_config = yaml.safe_load(f)

    model = TimestampAwareBackgroundSubtractor(
        history_seconds=bg_config["history_seconds"],
        var_threshold=bg_config["var_threshold"],
        detect_shadows=bg_config["detect_shadows"],
        area_threshold=bg_config["area_threshold"],
        shadow_correlation_threshold=bg_config["shadow_correlation_threshold"],
        morph_radius=bg_config["morph_radius"],
        morph_thresh=bg_config["morph_thresh"],
        morph_iters=bg_config["morph_iters"],
        default_fps=bg_config["default_fps"],
        region_of_interest=cv.imread(bg_config["region_of_interest"], cv.IMREAD_GRAYSCALE),
        night_mode_kwargs={
            k[6:]: v for k, v in bg_config.items() if k.startswith("night_")
        },
        # debug_dir=Path("debug"),
    )

    ann_path = image_dir / ANNOTATION_FILE
    annotations = load_annotations(ann_path)
    labels = annotations.setdefault("labels", {})

    # As we iterate, we only load images from disk if (a) they need processing or (b) they are
    # part of the recent history of a file that needs processing, such that the background model
    # is up to date. When we skip a file, we add it to recent_skipped. When we process a file,
    # we maybe need to load some of the recent_skipped files to update the background model. When
    # an image is processed, we clear recent_skipped. This way, recent_skipped always contains
    # the set of files *since* the last processed file.
    recent_skipped = deque()

    pprint(labels)
    i = 0
    for timestamp, filename, needs_annotation, needs_features in iter_files_with_flags(
        annotations, image_dir
    ):
        key = str(filename.relative_to(image_dir))
        if not needs_annotation and not needs_features:
            recent_skipped.append((timestamp, filename))
            continue

        # Burn through all skipped files and use any that are recent enough to update the BG model
        while recent_skipped:
            ts, fn = recent_skipped.popleft()
            # If skipped file is part of this file's context, load it to update the model
            if ts >= timestamp - model.history_seconds:
                img = cv.imread(str(fn))
                model.apply(img, ts)

        i += 1
        if i % 100 == 0:
            print(f"Completed thru {i} images, current: {filename}")
            save_annotations(ann_path, annotations)

        # Once we reach here, we can assume that the BG model is up-to-date and that we just need
        # to load and process this file.
        img = cv.imread(str(filename))

        mask, blobs = model.applyWithStats(img, timestamp)

        # Skip blobs that are almost certainly shadows
        blobs = [b for b in blobs if np.mean(b.shadow_correlation()) < 0.5]

        if needs_annotation:
            annots = get_annotations(img, blobs, labels, prior_annotations.get(key, []))
            if annots is None:
                save_annotations(ann_path, annotations)
                cv.destroyAllWindows()
                print(f"Annotations saved to {ann_path}")
                sys.exit(0)
            else:
                annotations[key] = annots

        if needs_features:
            annots = annotations.get(key, [])
            for ann in annots:
                # Find matching blob by closest bbox
                best_blob, best_iou = None, 0.5
                for blob in blobs:
                    iou = ForegroundBlob.iou(np.array(ann["bbox"]), blob.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_blob = blob
                if best_blob is not None:
                    ann["features"] = featurize(best_blob).tolist()
                else:
                    ann["features"] = []

    cv.destroyAllWindows()
    save_annotations(ann_path, annotations)
    print(f"Annotations saved to {ann_path}")


if __name__ == "__main__":
    main()
