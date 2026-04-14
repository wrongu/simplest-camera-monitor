import argparse
import json
import os
import signal
import sys
import threading
import webbrowser
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from pprint import pprint
from typing import Optional

import cv2 as cv
import numpy as np
import yaml
from flask import Flask, Response, jsonify, render_template, request

from background_model import BoundingBox, TimestampAwareBackgroundSubtractor
from image_loader import get_all_timestamped_files_sorted

ANNOTATION_FILE = "annotations.json"


def load_annotations(path: str | Path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"labels": {}}


def save_annotations(path, data, through_key=None):
    if through_key is not None:
        data["through"] = through_key
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Compute IoU between two bbox dicts (each with 'bbox': [x,y,w,h])."""
    x1, y1, w1, h1 = b1.x, b1.y, b1.width, b1.height
    x2, y2, w2, h2 = b2.x, b2.y, b2.width, b2.height
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _apply_prior_annotations(blobs: list[BoundingBox], prior_annots: list[BoundingBox], iou_threshold=0.5):
    """Pre-populate blob labels from prior annotations using IoU matching."""
    if not prior_annots or not blobs:
        return
    for blob in blobs:
        best_iou, best_prior = 0.0, None
        for prior in prior_annots:
            iou = _iou(blob, prior)
            if iou > best_iou:
                best_iou, best_prior = iou, prior
        if best_prior is not None and best_iou > iou_threshold:
            blob.class_id = best_prior.class_id


class AnnotatorState:
    """All mutable state for the annotation session, protected by a lock."""

    def __init__(
        self,
        image_dir: Path,
        bg_model: TimestampAwareBackgroundSubtractor,
        prior_annotations: dict,
        skip_no_motion: bool,
        paused: bool,
    ):
        self.lock = threading.Lock()

        self.image_dir = image_dir
        self.annot_file = image_dir / ANNOTATION_FILE
        self.bg_model = bg_model
        self.prior_annotations = prior_annotations

        self.annotations = load_annotations(self.annot_file)
        self.labels = self.annotations.setdefault("labels", {})

        all_files = list(get_all_timestamped_files_sorted(image_dir))
        self._total_files = len(all_files)
        self._file_iter = self._iter_files()
        self._recent_skipped = deque()

        self.history_left: deque = deque(maxlen=200)
        self.history_right: deque = deque(maxlen=200)

        self.current_key: str | None = None
        self.current_img_bytes: bytes | None = None
        self.current_bboxes: list[BoundingBox] = []

        self._loading: bool = False
        self._done: bool = False
        self._paused: bool = paused
        self._skipping = skip_no_motion
        self._images_processed: int = 0

    def _iter_files(self):
        """Yield (timestamp, filename, needs_annotation) for each file."""
        up_through = self.annotations.get("through", None)
        skipping = up_through is not None
        for time_and_file in get_all_timestamped_files_sorted(self.image_dir):
            timestamp, filename = time_and_file
            key = str(filename.relative_to(self.image_dir))
            if up_through is None or key >= up_through:
                skipping = False
            needs_annotation = key not in self.annotations or not isinstance(
                self.annotations[key], list
            )
            yield timestamp, filename, needs_annotation and not skipping

    def _encode_image(self, img) -> bytes:
        ok, buf = cv.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("Failed to encode image as JPEG")
        return buf.tobytes()

    def _process_image(self, timestamp, filename) -> tuple[bytes, list[BoundingBox]]:
        """Run the background model on one image and return (jpeg_bytes, bboxes)."""
        img = cv.imread(str(filename))
        _, blobs = self.bg_model.applyWithStats(img, timestamp)
        blobs = [b for b in blobs if np.mean(b.shadow_correlation()) < self.bg_model.shadow_correlation_threshold]
        bboxes = [b.bbox for b in blobs]
        img_bytes = self._encode_image(img)
        return img_bytes, bboxes

    def _load_frame(self, key, img_bytes, bboxes):
        """Set the current frame (called under lock)."""
        self.current_key = key
        self.current_img_bytes = img_bytes
        self.current_bboxes = bboxes
        self._loading = False

    def _do_load_next_from_iterator(self):
        """
        Advance through the file iterator until the next annotatable frame.
        Designed to run on a background thread when burn-through is needed.
        """
        try:
            while True:
                timestamp, filename, needs_annotation = next(self._file_iter)
                key = str(filename.relative_to(self.image_dir))
                with self.lock:
                    self.current_key = key

                if not needs_annotation:
                    self._recent_skipped.append((timestamp, filename))
                    continue

                # Burn through recently-skipped files to update the bg model
                while self._recent_skipped:
                    ts, fn = self._recent_skipped.popleft()
                    if ts >= timestamp - 2 * self.bg_model.history_seconds:
                        img = cv.imread(str(fn))
                        self.bg_model.applyWithStats(img, ts)

                self._images_processed += 1
                if self._images_processed % 100 == 0:
                    with self.lock:
                        save_annotations(
                            self.annot_file, self.annotations, through_key=key
                        )

                img_bytes, blobs_dicts = self._process_image(timestamp, filename)

                # Apply prior annotations via IoU matching
                prior = self.prior_annotations.get(key)
                if prior and key not in self.annotations:
                    _apply_prior_annotations(blobs_dicts, prior)

                if self._skipping and len(blobs_dicts) == 0:
                    # Skip this frame silently
                    with self.lock:
                        self.history_left.append((key, img_bytes, blobs_dicts))
                    continue

                # We have a frame to show
                with self.lock:
                    self.history_left.append(
                        (self.current_key, self.current_img_bytes, self.current_bboxes)
                    )
                    self._load_frame(key, img_bytes, blobs_dicts)
                return

        except StopIteration:
            with self.lock:
                self._done = True
                self._loading = False

    def _start_load_next(self):
        """
        Begin loading the next frame from the iterator. Sets _loading=True and
        spawns a background thread to do the work.
        """
        self._loading = True
        t = threading.Thread(target=self._do_load_next_from_iterator, daemon=True)
        t.start()

    def advance(self, action: str):
        """
        Navigate to a new frame based on action. Must be called under self.lock.
        'prev', 'next', 'resume', 'toggle_skip', are navigation actions; 'quit' exits.
        """
        # TODO - the ,/. logic in browser skips many frames, jumping only between those that had motion.
        if action == "prev":
            if self.history_left:
                self.history_right.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, blobs_dicts = self.history_left.pop()
                self._load_frame(key, img_bytes, blobs_dicts)

        elif action == "next":
            if self.history_right:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, blobs_dicts = self.history_right.pop()
                self._load_frame(key, img_bytes, blobs_dicts)
            else:
                self._start_load_next()

        elif action == "resume":
            while self.history_right:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, blobs_dicts = self.history_right.pop()
                self._load_frame(key, img_bytes, blobs_dicts)
            self._paused = False
            self._start_load_next()

        elif action == "toggle_skip":
            self._skipping = not self._skipping

        elif action == "quit":
            save_annotations(
                self.annot_file, self.annotations, through_key=self.current_key
            )
            self._done = True
            threading.Thread(
                target=lambda: os.kill(os.getpid(), signal.SIGINT), daemon=True
            ).start()

    def get_state_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the current state (call under lock)."""
        existing = self.annotations.get(self.current_key)
        if not isinstance(existing, list):
            existing = []
        return {
            "key": self.current_key,
            "processed": self._images_processed,
            "total": self._total_files,
            "blobs": [bbox.to_dict() for bbox in self.current_bboxes],
            "existing_annotations": existing,
            "labels": self.labels,
            "loading": self._loading,
            "skipping": self._skipping,
            "done": self._done,
        }

    def save_bboxes(self, key: str, bboxes: list[dict]):
        """Persist submitted bboxes for the given key (call under lock)."""
        labeled = [b for b in bboxes if b["label"] is not None]
        if labeled:
            self.annotations[key] = labeled
        elif key in self.annotations:
            del self.annotations[key]


def create_app(state: AnnotatorState) -> Flask:
    app = Flask(__name__, template_folder="templates")
    # Suppress Flask's default request logging for a cleaner terminal
    import logging

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return render_template("annotator.html")

    @app.route("/api/state")
    def api_state():
        with state.lock:
            return jsonify(state.get_state_dict())

    @app.route("/api/image")
    def api_image():
        img_bytes = state.current_img_bytes
        if img_bytes is None:
            return Response(status=204)
        return Response(img_bytes, mimetype="image/jpeg")

    @app.route("/api/submit", methods=["POST"])
    def api_submit():
        data = request.get_json()
        bboxes = data.get("bboxes", [])
        action = data.get("action", "next")
        with state.lock:
            if state.current_key is not None:
                state.save_bboxes(state.current_key, bboxes)
            state.advance(action)
            return jsonify(state.get_state_dict())

    @app.route("/api/labels", methods=["POST"])
    def api_labels():
        data = request.get_json()
        label_id = str(data["id"])
        name = str(data["name"]).strip()
        with state.lock:
            state.labels[label_id] = name
            save_annotations(
                state.annot_file, state.annotations, through_key=state.current_key
            )
            return jsonify({"labels": state.labels})

    @app.route("/api/quit", methods=["POST"])
    def api_quit():
        with state.lock:
            state.advance("quit")
        return jsonify({"status": "shutting_down"})

    return app


def main(
    image_dir: Path,
    bg_model: TimestampAwareBackgroundSubtractor,
    prior_annotations: dict,
    skip_no_motion: bool = False,
    paused: bool = False,
):
    state = AnnotatorState(
        image_dir, bg_model, prior_annotations, skip_no_motion, paused
    )

    pprint(state.labels)

    # Load the first frame before starting the server
    state._do_load_next_from_iterator()

    if state._done:
        print("No images to annotate.")
        return

    app = create_app(state)

    url = "http://127.0.0.1:5000"
    print(f"Starting annotation server at {url}")
    # Open browser slightly after server starts
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        with state.lock:
            save_annotations(
                state.annot_file, state.annotations, through_key=state.current_key
            )
        print(f"Annotations saved to {state.annot_file}")


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
        help="If set, start the script in 'paused' mode.",
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
