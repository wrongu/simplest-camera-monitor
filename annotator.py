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

from background_model import TimestampAwareBackgroundSubtractor, BoundingBox
from image_loader import get_all_timestamped_files_sorted

ANNOTATION_FILE = "annotations.json"


def load_annotations(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"labels": {}}


def save_annotations(path, data, through_key=None):
    if through_key is not None:
        data["through"] = through_key
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def prompt_label_name(label_id):
    name = input(f"Enter name for label {label_id}: ")
    return name.strip()


class Interface(object):
    COLOR_LABELED = (0, 255, 0)
    COLOR_UNLABELED = (0, 0, 255)
    COLOR_ACTIVE = (255, 255, 0)

    def __init__(self, image: cv.Mat, bboxes: list[BoundingBox], labels_dict: dict):
        self.image = image
        self.annotations = list(bboxes)
        self.labels_dict = labels_dict
        self._active_bbox = None
        self._drag_start = None
        self.last_keypress = -1

        cv.namedWindow("Interface", cv.WINDOW_AUTOSIZE)
        cv.setMouseCallback("Interface", self.on_mouse)

        self.select_next()

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_RBUTTONDOWN:
            self._drag_start = (x, y)
            # create a new annotation/new bbox. Overwriting an existing one if one was already
            # selected.
            if self._active_bbox is not None:
                self._active_bbox.set_bounds(x, y, x, y)
            else:
                self._active_bbox = BoundingBox(x, y, 0, 0)
        elif event == cv.EVENT_MOUSEMOVE:
            if self._drag_start is not None:
                self._active_bbox.set_bounds(*self._drag_start, x, y)
        elif event == cv.EVENT_RBUTTONUP:
            self._drag_start = None
            # Done dragging. Pop this bbox from any in the current list; will add it back in a
            # moment if it was big enough
            if self._active_bbox in self.annotations:
                self.annotations.remove(self._active_bbox)
            if self._active_bbox.area < 500:
                self._active_bbox = None
            else:
                # add (or re-add) this to the list. keep it active.
                self.annotations.append(self._active_bbox)
        elif event == cv.EVENT_LBUTTONUP:
            self.save_active()
            # Select whatever bbox was clicked, or deselect if none is nearby
            dist, closest = float("inf"), None
            for bbox in self.annotations:
                d = bbox.distance(x, y)
                if d < dist:
                    dist, closest = d, bbox
            if dist < 5:
                self._active_bbox = closest
            else:
                self._active_bbox = None

    def init_with_prior_annotations(self, prior_annots, iou_threshold=0.5):
        if len(prior_annots) == 0 or len(self.annotations) == 0:
            return
        for blob in self.annotations:
            # Check if this blob matches any prior annotation (by IoU)
            matched_prior = None
            best_iou = 0.0
            for prior in prior_annots:
                iou = BoundingBox.iou(BoundingBox(*prior["bbox"]), blob)
                if iou > best_iou:
                    best_iou = iou
                    matched_prior = prior
            if matched_prior is not None and best_iou > iou_threshold:
                blob.class_id = matched_prior["label"]

    def refresh_display(self):
        display = self.image.copy()
        for blob in self.annotations:
            if blob is self._active_bbox:
                continue

            blob.draw(
                display,
                color=(
                    Interface.COLOR_LABELED
                    if blob.class_id is not None
                    else Interface.COLOR_UNLABELED
                ),
            )

        if self._active_bbox is not None:
            self._active_bbox.draw(display, color=Interface.COLOR_ACTIVE)

        cv.imshow("Interface", display)

    def save_active(self):
        if self._active_bbox is not None:
            if self._active_bbox not in self.annotations:
                self.annotations.append(self._active_bbox)

    def select_next(self):
        self.save_active()
        if len(self.annotations) > 0:
            if self._active_bbox is not None:
                idx = self.annotations.index(self._active_bbox)
                idx = (idx + 1) % len(self.annotations)
                self._active_bbox = self.annotations[idx]
            else:
                self._active_bbox = self.annotations[0]

    def run_interactive(self, extra_quit_keys=None) -> None | list[BoundingBox]:
        quit_keys = [3, 13, 27]  # enter, return, escape
        if extra_quit_keys is not None:
            quit_keys.extend(extra_quit_keys)

        while True:
            self.refresh_display()
            self.last_keypress = cv.waitKey(10)
            if ord("0") <= self.last_keypress <= ord("9"):
                # Label whatever is active
                if self._active_bbox is not None:
                    self._active_bbox.class_id = chr(self.last_keypress)
                    if self._active_bbox not in self.annotations:
                        self.annotations.append(self._active_bbox)
                    self.select_next()
            elif self.last_keypress in [40, 127]:
                # Delete or backspace
                if self._active_bbox is not None:
                    if self._active_bbox in self.annotations:
                        self.annotations.remove(self._active_bbox)
                    self._active_bbox = None
                self.select_next()
            elif self.last_keypress == 9:
                # tab -> cycle
                self.select_next()
            elif self.last_keypress in quit_keys:
                return [b for b in self.annotations if b.class_id is not None]
            elif self.last_keypress != -1:
                print(f"WTF? {self.last_keypress} pressed")


def iter_files_with_flags(annotations, image_dir):
    up_through = annotations.get("through", None)
    skipping = up_through is not None
    for time_and_file in get_all_timestamped_files_sorted(image_dir):
        timestamp, filename = time_and_file
        key = str(filename.relative_to(image_dir))
        if key >= up_through:
            skipping = False

        needs_annotation = key not in annotations or not isinstance(
            annotations[key], list
        )
        # Hack: old annotations may exist but involve a label that has since changed
        if not needs_annotation:
            needs_annotation = any(a["label"] in ["0", "4", "2"] for a in annotations[key])

        yield timestamp, filename, needs_annotation and not skipping


def main(
    image_dir: Path,
    bg_model: TimestampAwareBackgroundSubtractor,
    prior_annotations: dict,
    skip_no_motion: bool = False,
    paused: bool = False,
):
    annot_file = image_dir / ANNOTATION_FILE
    annotations: dict[str, dict | list] = load_annotations(annot_file)
    labels = annotations.setdefault("labels", {})

    # As we iterate, we only load images from disk if (a) they need processing or (b) they are
    # part of the recent history of a file that needs processing, such that the background model
    # is up to date. When we skip a file, we add it to recent_skipped. When we process a file,
    # we maybe need to load some of the recent_skipped files to update the background model. When
    # an image is processed, we clear recent_skipped. This way, recent_skipped always contains
    # the set of files *since* the last processed file.
    recent_skipped = deque()
    history_left = deque(maxlen=200)
    history_right = deque(maxlen=200)

    pprint(labels)
    i = 0
    quit_requested = False
    for timestamp, filename, needs_annotation in iter_files_with_flags(
        annotations, image_dir
    ):
        if quit_requested:
            break

        key = str(filename.relative_to(image_dir))
        if not needs_annotation:
            recent_skipped.append((timestamp, filename))
            continue

        # Burn through all skipped files and use any that are recent enough to update the BG model
        while recent_skipped:
            ts, fn = recent_skipped.popleft()
            # If skipped file is part of this file's context, load it to update the model
            if ts >= timestamp - 2 * bg_model.history_seconds:
                img = cv.imread(str(fn))
                _, blobs = bg_model.applyWithStats(img, ts)
                history_left.append((key, img, blobs))

        i += 1
        if i % 100 == 0:
            print(f"Completed thru {i} images, current: {filename}")
            save_annotations(annot_file, annotations, through_key=key)

        # Once we reach here, we can assume that the BG model is up-to-date and that we just need
        # to load and process this file.
        img = cv.imread(str(filename))

        _, blobs = bg_model.applyWithStats(img, timestamp)

        # Skip blobs that are almost certainly shadows
        blobs = [b for b in blobs if np.mean(b.shadow_correlation()) < 0.5]

        if skip_no_motion:
            if len(blobs) == 0:
                needs_annotation = False
                cv.imshow("Interface", img)
                k = cv.waitKey(1)
                if k == ord(" "):
                    paused = True

        paused = paused or needs_annotation
        # loop preconditions:
        # - (key, img, blobs) is 'current'
        # - history_right is empty, history_left is strictly earlier
        # after the loop, this must all remain true. during the pause loop, history will get weird
        while paused:
            if key in annotations:
                init_bboxes = [BoundingBox.from_dict(a) for a in annotations[key]]
            else:
                init_bboxes = [blob.bbox for blob in blobs]
            interface = Interface(img, init_bboxes, labels)
            if key in prior_annotations and key not in annotations:
                interface.init_with_prior_annotations(prior_annotations[key])
            result: list[BoundingBox] = interface.run_interactive(
                extra_quit_keys=[ord(","), ord("."), ord(" ")]
            )
            if interface.last_keypress == 27:
                quit_requested = True
                break
            elif len(result) > 0:
                annotations[key] = [b.to_dict() for b in result]

            if interface.last_keypress == ord(" "):
                while len(history_right) > 0:
                    history_left.append((key, img, blobs))
                    key, img, blobs = history_right.pop()
                paused = False
            elif interface.last_keypress == ord(","):
                if len(history_left) > 0:
                    history_right.append((key, img, blobs))
                    key, img, blobs = history_left.pop()
            elif interface.last_keypress == ord("."):
                if len(history_right) > 0:
                    history_left.append((key, img, blobs))
                    key, img, blobs = history_right.pop()
                else:
                    # This breaks the current 'paused' loop and returns control to the outer 'for'
                    # loop, thus loading the next image that is not yet loaded, but then we'll hit
                    # the 'while paused' block again on the next frame
                    break

        history_left.append((key, img, blobs))

    cv.destroyAllWindows()
    save_annotations(annot_file, annotations, through_key=key)
    print(f"Annotations saved to {annot_file}")


if __name__ == "__main__":
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
    parser.add_argument(
        "--skip-no-motion",
        action="store_true",
        help="If True, only prompt the user to label if motion is detected.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="If set, start the script in 'paused' mode."
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    images = list(get_all_timestamped_files_sorted(image_dir))
    if not images:
        print("No images found in", image_dir)
        sys.exit(1)

    if not args.bg_model_config.exists():
        print("Background model config file not found:", args.bg_model_config)
        sys.exit(1)

    with open(args.bg_model_config, "r") as f:
        bg_config = yaml.safe_load(f)

    model = TimestampAwareBackgroundSubtractor(
        history_seconds=bg_config["history_seconds"],
        var_threshold=bg_config["var_threshold"],
        detect_shadows=bg_config["detect_shadows"],
        prep_hist_eq=bg_config["prep_hist_eq"],
        prep_blur=bg_config["prep_blur"],
        area_threshold=bg_config["area_threshold"],
        shadow_correlation_threshold=bg_config["shadow_correlation_threshold"],
        morph_radius=bg_config["morph_radius"],
        morph_thresh=bg_config["morph_thresh"],
        morph_iters=bg_config["morph_iters"],
        default_fps=bg_config["default_fps"],
        region_of_interest=cv.imread(
            bg_config["region_of_interest"], cv.IMREAD_GRAYSCALE
        ),
        night_mode_kwargs={
            k[6:]: v for k, v in bg_config.items() if k.startswith("night_")
        },
    )

    prior_annotations = {}
    if args.prior_annotations is not None:
        if not args.prior_annotations.exists():
            print("Prior annotations file not found:", args.prior_annotations)
            sys.exit(1)
        with open(args.prior_annotations, "r") as f:
            prior_annotations = json.load(f)

    main(image_dir, model, prior_annotations, args.skip_no_motion, args.paused)
