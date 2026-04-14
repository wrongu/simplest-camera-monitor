import argparse
import json
import os
import signal
import sys
import threading
import webbrowser
from collections import deque
from dataclasses import dataclass, field
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

# Sentinel used to mark a prefetch slot that has a thread running but hasn't finished yet.
_PREFETCH_PENDING = object()


# ─── Schemas ─────────────────────────────────────────────────────────────────


@dataclass
class LabeledBox:
    """A bounding box with an optional class label. Used in file storage and API."""

    bbox: list  # [x, y, width, height] — four ints
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "LabeledBox":
        return cls(bbox=list(d["bbox"]), label=d.get("label"))

    def to_dict(self) -> dict:
        return {"bbox": self.bbox, "label": self.label}

    @classmethod
    def from_bbox(cls, b: BoundingBox) -> "LabeledBox":
        return cls(
            bbox=[int(b.x), int(b.y), int(b.width), int(b.height)],
            label=b.class_id,
        )


@dataclass
class AnnotationFile:
    """Represents the full annotations.json on disk."""

    labels: dict = field(default_factory=dict)  # {label_id: name}
    through: Optional[str] = None               # last-processed image key
    images: dict = field(default_factory=dict)  # {image_key: list[LabeledBox]}

    @classmethod
    def load(cls, path: Path) -> "AnnotationFile":
        if not path.exists():
            return cls()
        with open(path, "r") as f:
            raw = json.load(f)
        images = {
            k: [LabeledBox.from_dict(b) for b in v]
            for k, v in raw.items()
            if k not in ("labels", "through") and isinstance(v, list)
        }
        return cls(
            labels=raw.get("labels", {}),
            through=raw.get("through"),
            images=images,
        )

    def save(self, path: Path, through_key: Optional[str] = None):
        if through_key is not None:
            self.through = through_key
        raw: dict = {"labels": self.labels}
        if self.through is not None:
            raw["through"] = self.through
        for k, v in self.images.items():
            raw[k] = [b.to_dict() for b in v]
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)


@dataclass
class FrameState:
    """API response payload for GET /api/state and POST /api/submit."""

    key: Optional[str]
    file_index: int
    total: int
    blobs: list           # list[LabeledBox]
    existing_annotations: list  # list[LabeledBox]
    labels: dict
    loading: bool
    skipping: bool
    needs_pause: bool     # False → frontend should auto-advance after its configured delay
    done: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "file_index": self.file_index,
            "total": self.total,
            "blobs": [b.to_dict() for b in self.blobs],
            "existing_annotations": [b.to_dict() for b in self.existing_annotations],
            "labels": self.labels,
            "loading": self.loading,
            "skipping": self.skipping,
            "needs_pause": self.needs_pause,
            "done": self.done,
        }


@dataclass
class SubmitRequest:
    """Parsed body of POST /api/submit."""

    bboxes: list   # list[LabeledBox]
    action: str

    @classmethod
    def from_dict(cls, d: dict) -> "SubmitRequest":
        return cls(
            bboxes=[LabeledBox.from_dict(b) for b in d.get("bboxes", [])],
            action=d.get("action", "next"),
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _iou(b1: LabeledBox, b2: LabeledBox) -> float:
    x1, y1, w1, h1 = b1.bbox
    x2, y2, w2, h2 = b2.bbox
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _apply_prior_annotations(
    blobs: list, prior_annots: list, iou_threshold: float = 0.5
):
    """Pre-populate blob labels from a prior annotation set via IoU matching."""
    for blob in blobs:
        best_iou, best_label = 0.0, None
        for prior in prior_annots:
            iou = _iou(blob, prior)
            if iou > best_iou:
                best_iou, best_label = iou, prior.label
        if best_label is not None and best_iou > iou_threshold:
            blob.label = best_label


# ─── Annotator state ──────────────────────────────────────────────────────────


class AnnotatorState:
    """All mutable state for the annotation session, protected by a lock."""

    # How many unannotated files ahead to prefetch from disk.
    PREFETCH_AHEAD = 3

    def __init__(
        self,
        image_dir: Path,
        bg_model: TimestampAwareBackgroundSubtractor,
        prior_annotations: dict,  # {image_key: list[LabeledBox]}
        skip_no_motion: bool,
        paused: bool,
    ):
        self.lock = threading.Lock()

        self.image_dir = image_dir
        self.annot_file = image_dir / ANNOTATION_FILE
        self.bg_model = bg_model
        self.prior_annotations = prior_annotations

        self.annot = AnnotationFile.load(self.annot_file)

        # Full sorted file list. Stored so we can index into it for prefetching.
        self._all_files: list = list(get_all_timestamped_files_sorted(image_dir))
        self._total_files: int = len(self._all_files)

        # Index into _all_files; the next file to process.
        self._iter_idx: int = self._find_resume_idx()

        # file_index counts how many files from _all_files have been visited.
        # Initialised to the resume offset so progress is correct on resume.
        self._file_index: int = self._iter_idx

        # Files just before the resume point that may be needed for bg model warm-up.
        self._recent_skipped: deque = deque()
        self._populate_recent_skipped_for_warmup()

        # Prefetch cache: str(filename) → bytes (raw JPEG) or _PREFETCH_PENDING.
        self._prefetch_cache: dict = {}

        self.history_left: deque = deque(maxlen=200)
        self.history_right: deque = deque(maxlen=200)

        self.current_key: Optional[str] = None
        self.current_img_bytes: Optional[bytes] = None
        self.current_bboxes: list = []          # list[LabeledBox]
        self.current_needs_pause: bool = True

        self._loading: bool = False
        self._done: bool = False
        # --paused overrides --skip-no-motion: start with skipping off so every frame pauses.
        self._skipping: bool = skip_no_motion and not paused
        self._images_processed: int = 0

    # ── Resume helpers ────────────────────────────────────────────────────────

    def _find_resume_idx(self) -> int:
        """Return the index of the first file at or after the `through` key."""
        through = self.annot.through
        if through is None:
            return 0
        for i, (ts, fn) in enumerate(self._all_files):
            if str(fn.relative_to(self.image_dir)) >= through:
                return i
        return len(self._all_files)  # through is past all files

    def _populate_recent_skipped_for_warmup(self):
        """
        Pre-populate _recent_skipped with files just before the resume point so
        the background model can be warmed up when processing the first new frame.
        """
        if self._iter_idx == 0 or self.annot.through is None:
            return
        if self._iter_idx >= len(self._all_files):
            resume_ts = self._all_files[-1][0]
        else:
            resume_ts = self._all_files[self._iter_idx][0]
        lookback_ts = resume_ts - 2 * self.bg_model.history_seconds
        # Scan backwards to find how far to look, then append in forward (chronological) order.
        start_i = 0
        for i in range(self._iter_idx - 1, -1, -1):
            if self._all_files[i][0] < lookback_ts:
                start_i = i + 1
                break
        for i in range(start_i, self._iter_idx):
            self._recent_skipped.append(self._all_files[i])

    # ── Prefetching ───────────────────────────────────────────────────────────

    def _do_prefetch(self, fn: Path):
        """Background thread: read a file from disk into the prefetch cache."""
        try:
            self._prefetch_cache[str(fn)] = fn.read_bytes()
        except Exception:
            self._prefetch_cache.pop(str(fn), None)

    def _kickoff_prefetch(self):
        """
        Launch background disk reads for the next PREFETCH_AHEAD unannotated files.
        Also evicts cache entries that are no longer needed (already processed).
        """
        # Evict stale entries: keep only files from the current iterator position forward.
        upcoming = {
            str(fn)
            for _, fn in self._all_files[self._iter_idx: self._iter_idx + self.PREFETCH_AHEAD * 2]
        }
        stale = [k for k in self._prefetch_cache if k not in upcoming]
        for k in stale:
            del self._prefetch_cache[k]

        # Launch prefetch threads for the next few unannotated files.
        idx = self._iter_idx
        launched = 0
        while idx < len(self._all_files) and launched < self.PREFETCH_AHEAD:
            ts, fn = self._all_files[idx]
            idx += 1
            key = str(fn.relative_to(self.image_dir))
            if key in self.annot.images:
                continue  # skip already-annotated files
            fn_str = str(fn)
            if fn_str not in self._prefetch_cache:
                self._prefetch_cache[fn_str] = _PREFETCH_PENDING
                threading.Thread(
                    target=self._do_prefetch, args=(fn,), daemon=True
                ).start()
            launched += 1

    # ── Image processing ──────────────────────────────────────────────────────

    def _process_image(self, timestamp, filename: Path) -> tuple:
        """
        Run the background model on one image.

        Returns (raw_jpeg_bytes, list[LabeledBox]).

        The raw JPEG bytes come directly from disk (no re-encoding), which avoids
        a costly cv.imencode call and serves the browser the original image quality.
        A prefetch cache is checked first so disk I/O overlaps with display time.
        """
        fn_str = str(filename)
        cached = self._prefetch_cache.pop(fn_str, None)
        if isinstance(cached, bytes):
            raw_bytes = cached
        else:
            # Either not prefetched yet, or prefetch thread still running — read directly.
            raw_bytes = filename.read_bytes()

        # Decode to BGR array for the background model.
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv.imdecode(buf, cv.IMREAD_COLOR)

        _, blobs = self.bg_model.applyWithStats(img, timestamp)
        blobs = [
            b for b in blobs
            if np.mean(b.shadow_correlation()) < self.bg_model.shadow_correlation_threshold
        ]
        bboxes = [LabeledBox.from_bbox(b.bbox) for b in blobs]
        return raw_bytes, bboxes

    def _load_frame(self, key: str, img_bytes: bytes, bboxes: list, needs_pause: bool):
        """Set the current frame. Must be called under self.lock."""
        self.current_key = key
        self.current_img_bytes = img_bytes
        self.current_bboxes = bboxes
        self.current_needs_pause = needs_pause
        self._loading = False

    # ── Forward iteration ─────────────────────────────────────────────────────

    def _do_load_next_from_iterator(self):
        """
        Advance through the file list to the next frame to display.

        May be called on a background thread (for subsequent frames) or the main
        thread (for the initial frame). All new (unannotated) frames are returned —
        no-motion frames in skip mode are marked needs_pause=False so the browser
        auto-advances them, but they are still visible so the user can spot false
        negatives from the background model.
        """
        # Push the current frame to history BEFORE iterating so that `,` returns
        # frames in strict chronological order.
        with self.lock:
            if self.current_key is not None:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )

        while self._iter_idx < len(self._all_files):
            timestamp, filename = self._all_files[self._iter_idx]
            self._iter_idx += 1
            self._file_index = self._iter_idx  # progress tracks all files visited

            key = str(filename.relative_to(self.image_dir))

            if key in self.annot.images:
                # Already annotated: add to burn-through queue and move on.
                self._recent_skipped.append((timestamp, filename))
                continue

            # Burn through recently-skipped files to keep the bg model current.
            while self._recent_skipped:
                ts, fn = self._recent_skipped.popleft()
                if ts >= timestamp - 2 * self.bg_model.history_seconds:
                    # Try cache first to avoid a redundant disk read.
                    cached = self._prefetch_cache.pop(str(fn), None)
                    if isinstance(cached, bytes):
                        buf = np.frombuffer(cached, dtype=np.uint8)
                        img = cv.imdecode(buf, cv.IMREAD_COLOR)
                    else:
                        img = cv.imread(str(fn))
                    self.bg_model.applyWithStats(img, ts)

            self._images_processed += 1
            if self._images_processed % 100 == 0:
                with self.lock:
                    self.annot.save(self.annot_file, through_key=key)

            img_bytes, bboxes = self._process_image(timestamp, filename)

            # Pre-populate labels from prior annotations via IoU matching.
            prior = self.prior_annotations.get(key, [])
            if prior and key not in self.annot.images:
                _apply_prior_annotations(bboxes, prior)

            needs_pause = bool(bboxes) or not self._skipping

            # Kick off prefetch for upcoming files while the bg thread is still running.
            self._kickoff_prefetch()

            with self.lock:
                self._load_frame(key, img_bytes, bboxes, needs_pause=needs_pause)
            return

        # Exhausted all files.
        with self.lock:
            self._done = True
            self._loading = False

    def _start_load_next(self):
        """Set loading=True and spawn a background thread to load the next frame."""
        self._loading = True
        threading.Thread(
            target=self._do_load_next_from_iterator, daemon=True
        ).start()

    # ── Navigation ────────────────────────────────────────────────────────────

    def advance(self, action: str):
        """
        Mutate navigation state. Must be called under self.lock.

        History navigation always sets needs_pause=True — the user is explicitly
        reviewing those frames. Forward iteration uses the frame's computed needs_pause.
        """
        if action == "prev":
            if self.history_left:
                self.history_right.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, bboxes = self.history_left.pop()
                self._load_frame(key, img_bytes, bboxes, needs_pause=True)

        elif action == "next":
            if self.history_right:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, bboxes = self.history_right.pop()
                self._load_frame(key, img_bytes, bboxes, needs_pause=True)
            else:
                self._start_load_next()

        elif action == "resume":
            # Drain history_right to reach the most recently-seen frame, then advance.
            while self.history_right:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, bboxes = self.history_right.pop()
                self._load_frame(key, img_bytes, bboxes, needs_pause=True)
            self._start_load_next()

        elif action == "toggle_skip":
            self._skipping = not self._skipping
            # Recompute needs_pause for the current frame immediately so the frontend
            # responds without waiting for the next navigation action.
            self.current_needs_pause = bool(self.current_bboxes) or not self._skipping

        elif action == "quit":
            self.annot.save(self.annot_file, through_key=self.current_key)
            self._done = True
            threading.Thread(
                target=lambda: os.kill(os.getpid(), signal.SIGINT), daemon=True
            ).start()

    # ── State snapshot ────────────────────────────────────────────────────────

    def get_state(self) -> FrameState:
        """Return a serialisable snapshot of the current state. Call under self.lock."""
        existing = self.annot.images.get(self.current_key, [])
        return FrameState(
            key=self.current_key,
            file_index=self._file_index,
            total=self._total_files,
            blobs=self.current_bboxes,
            existing_annotations=existing,
            labels=self.annot.labels,
            loading=self._loading,
            skipping=self._skipping,
            needs_pause=self.current_needs_pause,
            done=self._done,
        )

    def save_bboxes(self, key: str, bboxes: list):
        """Persist submitted bboxes for the given key. Call under self.lock."""
        labeled = [b for b in bboxes if b.label is not None]
        if labeled:
            self.annot.images[key] = labeled
        elif key in self.annot.images:
            del self.annot.images[key]


# ─── Flask app ────────────────────────────────────────────────────────────────


def create_app(state: AnnotatorState) -> Flask:
    app = Flask(__name__, template_folder="templates")

    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return render_template("annotator.html")

    @app.route("/api/state")
    def api_state():
        with state.lock:
            return jsonify(state.get_state().to_dict())

    @app.route("/api/image")
    def api_image():
        img_bytes = state.current_img_bytes
        if img_bytes is None:
            return Response(status=204)
        return Response(img_bytes, mimetype="image/jpeg")

    @app.route("/api/submit", methods=["POST"])
    def api_submit():
        req = SubmitRequest.from_dict(request.get_json())
        with state.lock:
            if state.current_key is not None:
                state.save_bboxes(state.current_key, req.bboxes)
            state.advance(req.action)
            return jsonify(state.get_state().to_dict())

    @app.route("/api/labels", methods=["POST"])
    def api_labels():
        data = request.get_json()
        label_id = str(data["id"])
        name = str(data["name"]).strip()
        with state.lock:
            state.annot.labels[label_id] = name
            state.annot.save(state.annot_file, through_key=state.current_key)
            return jsonify({"labels": state.annot.labels})

    @app.route("/api/quit", methods=["POST"])
    def api_quit():
        with state.lock:
            state.advance("quit")
        return jsonify({"status": "shutting_down"})

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────


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
    pprint(state.annot.labels)

    # Load the first frame synchronously before starting the server.
    state._do_load_next_from_iterator()

    if state._done:
        print("No images to annotate.")
        return

    app = create_app(state)
    url = "http://127.0.0.1:5000"
    print(f"Starting annotation server at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        with state.lock:
            state.annot.save(state.annot_file, through_key=state.current_key)
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
        help="Path to existing annotations to use as label defaults",
    )
    parser.add_argument(
        "--skip-no-motion",
        action="store_true",
        help="Auto-advance frames with no detected motion (still displayed; user can interrupt).",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start with skip mode off, pausing on every frame regardless of motion.",
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

    prior_annotations: dict = {}
    if args.prior_annotations is not None:
        if not args.prior_annotations.exists():
            print("Prior annotations file not found:", args.prior_annotations)
            sys.exit(1)
        with open(args.prior_annotations, "r") as f:
            raw = json.load(f)
        prior_annotations = {
            k: [LabeledBox.from_dict(b) for b in v]
            for k, v in raw.items()
            if k not in ("labels", "through") and isinstance(v, list)
        }

    main(image_dir, model, prior_annotations, args.skip_no_motion, args.paused)
