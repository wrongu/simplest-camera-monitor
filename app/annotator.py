import argparse
import bisect
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import portion
import yaml
from flask import Flask, Response, jsonify, render_template, request

from app.background_model import TimestampAwareBackgroundSubtractor
from app.image_loader import get_all_timestamped_files_sorted
from app.utils import BoundingBox

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
    start: Optional[str] = None  # first image key where this file starts
    through: Optional[str] = None  # last-processed image key
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
            start=raw.get("start"),
            through=raw.get("through"),
            images=images,
        )

    def save(self, path: Path, through_key: Optional[str] = None):
        if through_key is not None:
            self.through = through_key
        raw: dict = {"labels": self.labels}
        if self.start is not None:
            raw["start"] = self.start
        if self.through is not None:
            raw["through"] = self.through
        for k, v in self.images.items():
            raw[k] = [b.to_dict() for b in v]
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)

    def get_interval(self, fudge: bool = False) -> portion.Interval | None:
        if self.start is None or self.through is None:
            return None
        delta = timedelta(seconds=30) if fudge else timedelta(0)
        return portion.closed(
            lower=datetime.strptime(self.start.replace("\\", "/"), "%Y/%m/%d/%H%M%S.jpg") - delta,
            upper=datetime.strptime(self.through.replace("\\", "/"), "%Y/%m/%d/%H%M%S.jpg") + delta,
        )


@dataclass
class FrameState:
    """API response payload for GET /api/state and POST /api/submit."""

    key: Optional[str]
    file_index: int
    total: int
    blobs: list  # list[LabeledBox]
    existing_annotations: list  # list[LabeledBox]
    labels: dict
    loading: bool
    skipping: bool
    needs_pause: bool  # False → frontend should auto-advance after its configured delay
    done: bool
    timestamp: Optional[str]     # ISO datetime of current frame, for timeline cursor
    session_start: Optional[str]  # ISO datetime of active annotation file's start

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
            "timestamp": self.timestamp,
            "session_start": self.session_start,
        }


@dataclass
class SubmitRequest:
    """Parsed body of POST /api/submit."""

    bboxes: list  # list[LabeledBox]
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


def _key_to_dt(key: str) -> datetime:
    return datetime.strptime(key.replace("\\", "/"), "%Y/%m/%d/%H%M%S.jpg")


def _ts_to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts)


def load_all_annotation_files(image_dir: Path) -> list[tuple[AnnotationFile, Path]]:
    """Load all annotation JSON files in image_dir, sorted by filename."""
    result = []
    for p in sorted(image_dir.glob("annotations.*.json")):
        af = AnnotationFile.load(p)
        result.append((af, p))
    return result


def compute_covered_interval(annot_files: list) -> portion.Interval:
    """Union of all annotation file intervals (with fudge for ±30s boundary slop)."""
    iv = portion.empty()
    for af, _ in annot_files:
        seg = af.get_interval(fudge=True)
        if seg is not None:
            iv |= seg
    return iv


def _interval_to_json(iv: portion.Interval) -> list[dict]:
    result = []
    for atom in iv:
        if not atom.empty and isinstance(atom.lower, datetime) and isinstance(atom.upper, datetime):
            result.append({"start": atom.lower.isoformat(), "end": atom.upper.isoformat()})
    return result


# Large time gaps between images that indicate a recording break rather than idle time.
_SEGMENT_GAP_SECONDS = 3 * 3600  # 3 hours


def _compute_segments(
    all_files: list, gap_threshold: float = _SEGMENT_GAP_SECONDS
) -> list[tuple[float, float]]:
    """
    Split image files into contiguous time clusters separated by gaps >= gap_threshold.
    Returns a list of (start_epoch, end_epoch) tuples.
    """
    if not all_files:
        return []
    clusters = []
    seg_start = all_files[0][0]
    prev_ts = seg_start
    for ts, _ in all_files[1:]:
        if ts - prev_ts >= gap_threshold:
            clusters.append((seg_start, prev_ts))
            seg_start = ts
        prev_ts = ts
    clusters.append((seg_start, prev_ts))
    return clusters


# ─── Annotator state ──────────────────────────────────────────────────────────


class AnnotatorState:
    """All mutable state for the annotation session, protected by a lock."""

    # How many unannotated files ahead to prefetch from disk.
    PREFETCH_AHEAD = 3

    def __init__(
        self,
        image_dir: Path,
        bg_model: TimestampAwareBackgroundSubtractor,
        labels_dict: dict[str, str],
        skip_no_motion: bool,
        paused: bool,
        start_key: Optional[str],
        end_key: Optional[str],
        existing_annot_files: list,   # list[tuple[AnnotationFile, Path]]
        covered: portion.Interval,
    ):
        self.lock = threading.Lock()

        self.image_dir = image_dir
        self.bg_model = bg_model
        self._labels_dict: dict[str, str] = dict(labels_dict)
        self._end_key = end_key

        # Pre-existing annotation coverage (read-only during session).
        self._covered: portion.Interval = covered

        # Merged image annotations from all existing files, for display when browsing.
        self._all_annot_images: dict[str, list[LabeledBox]] = {}
        for af, _ in existing_annot_files:
            self._all_annot_images.update(af.images)

        # Full sorted file list.
        self._all_files: list = list(get_all_timestamped_files_sorted(image_dir))
        self._total_files: int = len(self._all_files)

        # Reverse lookup: image key → index in _all_files.
        self._key_to_idx: dict = {
            str(fn.relative_to(self.image_dir)): i for i, (_, fn) in enumerate(self._all_files)
        }

        # Index into _all_files; the next file to process.
        self._iter_idx: int = self._find_start_idx(start_key)
        self._file_index: int = self._iter_idx

        # Files just before the resume point that may be needed for bg model warm-up.
        self._recent_skipped: deque = deque()
        self._populate_recent_skipped_for_warmup()

        # Active annotation file for this session (created on first unannotated frame).
        self._active_annot: Optional[AnnotationFile] = None
        self._active_annot_path: Optional[Path] = None

        # Prefetch cache: str(filename) → tuple[bytes, cv.Mat] or _PREFETCH_PENDING.
        self._prefetch_cache: dict = {}

        self.history_left: deque = deque(maxlen=200)
        self.history_right: deque = deque(maxlen=200)

        self.current_key: Optional[str] = None
        self.current_img_bytes: Optional[bytes] = None
        self.current_bboxes: list = []
        self.current_needs_pause: bool = True

        self._loading: bool = False
        self._done: bool = False
        # --paused overrides --skip-no-motion: start with skipping off so every frame pauses.
        self._skipping: bool = skip_no_motion and not paused
        self._images_processed: int = 0

    # ── Resume helpers ────────────────────────────────────────────────────────

    def _find_start_idx(self, start_key: Optional[str] = None) -> int:
        """Find the index of the first unannotated file, optionally after start_key."""
        begin = 0
        if start_key is not None:
            for i, (ts, fn) in enumerate(self._all_files):
                if str(fn.relative_to(self.image_dir)) >= start_key:
                    begin = i
                    break
            else:
                return len(self._all_files)

        for i in range(begin, len(self._all_files)):
            ts, fn = self._all_files[i]
            if _ts_to_dt(ts) not in self._covered:
                return i
        return len(self._all_files)

    def _populate_recent_skipped_for_warmup(self):
        """
        Pre-populate _recent_skipped with files just before the resume point so
        the background model can be warmed up when processing the first new frame.
        """
        if self._iter_idx == 0:
            return
        if self._iter_idx >= len(self._all_files):
            resume_ts = self._all_files[-1][0]
        else:
            resume_ts = self._all_files[self._iter_idx][0]
        lookback_ts = resume_ts - 2 * self.bg_model.history_seconds
        start_i = 0
        for i in range(self._iter_idx - 1, -1, -1):
            if self._all_files[i][0] < lookback_ts:
                start_i = i + 1
                break
        for i in range(start_i, self._iter_idx):
            self._recent_skipped.append(self._all_files[i])

    # ── Annotation file lifecycle ─────────────────────────────────────────────

    def _ensure_active_annot(self, start_key: str):
        """Create a new active annotation file if none exists for this gap."""
        if self._active_annot is not None:
            return
        annot_uid = int(time.time())
        self._active_annot_path = self.image_dir / f"annotations.{annot_uid}.json"
        self._active_annot = AnnotationFile(labels=dict(self._labels_dict), start=start_key)

    def _save_active_annot(self, through_key: Optional[str] = None):
        """Persist the active annotation file. No-op if none is active."""
        if self._active_annot is not None and self._active_annot_path is not None:
            self._active_annot.save(self._active_annot_path, through_key=through_key)

    # ── Prefetching ───────────────────────────────────────────────────────────

    def _do_prefetch(self, fn: Path):
        """Background thread: read a file from disk into the prefetch cache."""
        try:
            byt = fn.read_bytes()
            img = cv.imdecode(np.frombuffer(byt, np.uint8), cv.IMREAD_COLOR)
            self._prefetch_cache[str(fn)] = byt, img
        except Exception:
            self._prefetch_cache.pop(str(fn), None)

    def _kickoff_prefetch(self):
        """
        Launch background disk reads for the next PREFETCH_AHEAD unannotated files.
        Also evicts cache entries that are no longer needed (already processed).
        """
        upcoming = {
            str(fn)
            for _, fn in self._all_files[self._iter_idx : self._iter_idx + self.PREFETCH_AHEAD * 2]
        }
        stale = [k for k in self._prefetch_cache if k not in upcoming]
        for k in stale:
            del self._prefetch_cache[k]

        idx = self._iter_idx
        launched = 0
        while idx < len(self._all_files) and launched < self.PREFETCH_AHEAD:
            ts, fn = self._all_files[idx]
            idx += 1
            key = str(fn.relative_to(self.image_dir))
            active_images = self._active_annot.images if self._active_annot else {}
            if _ts_to_dt(ts) in self._covered or key in active_images:
                continue
            fn_str = str(fn)
            if fn_str not in self._prefetch_cache:
                self._prefetch_cache[fn_str] = _PREFETCH_PENDING
                threading.Thread(target=self._do_prefetch, args=(fn,), daemon=True).start()
            launched += 1

    # ── Image processing ──────────────────────────────────────────────────────

    def _process_image(self, timestamp, filename: Path) -> tuple:
        """
        Run the background model on one image.

        Returns (raw_jpeg_bytes, list[LabeledBox]).
        """
        fn_str = str(filename)
        cached = self._prefetch_cache.pop(fn_str, None)
        if isinstance(cached, tuple):
            byt, img = cached
        else:
            byt = filename.read_bytes()
            img = cv.imdecode(np.frombuffer(byt, dtype=np.uint8), cv.IMREAD_COLOR)

        _, blobs = self.bg_model.applyWithStats(img, timestamp)
        blobs = [
            b
            for b in blobs
            if np.mean(b.shadow_correlation()) < self.bg_model.shadow_correlation_threshold
        ]
        bboxes = [LabeledBox.from_bbox(b.bbox) for b in blobs]
        return byt, bboxes

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
        thread (for the initial frame). Files already covered by existing annotation
        intervals are skipped transparently, keeping the bg model warmed up.
        """
        with self.lock:
            if self.current_key is not None:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )

        while self._iter_idx < len(self._all_files):
            timestamp, filename = self._all_files[self._iter_idx]
            self._iter_idx += 1
            self._file_index = self._iter_idx

            key = str(filename.relative_to(self.image_dir))

            if self._end_key is not None and key > self._end_key:
                break

            ts_dt = _ts_to_dt(timestamp)
            active_images = self._active_annot.images if self._active_annot else {}

            # Skip files covered by pre-existing annotations or already annotated this session.
            if ts_dt in self._covered or key in active_images:
                self._recent_skipped.append((timestamp, filename))
                continue

            # Burn through recently-skipped files to keep the bg model current.
            while self._recent_skipped:
                ts, fn = self._recent_skipped.popleft()
                if ts >= timestamp - 2 * self.bg_model.history_seconds:
                    cached = self._prefetch_cache.pop(str(fn), None)
                    if isinstance(cached, tuple):
                        _, img = cached
                    else:
                        img = cv.imread(str(fn))
                    self.bg_model.applyWithStats(img, ts)

            # Ensure we have an active annotation file for this unannotated gap.
            self._ensure_active_annot(key)

            self._images_processed += 1
            if self._images_processed % 100 == 0:
                with self.lock:
                    self._save_active_annot(through_key=key)

            img_bytes, bboxes = self._process_image(timestamp, filename)
            needs_pause = bool(bboxes) or not self._skipping
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
        threading.Thread(target=self._do_load_next_from_iterator, daemon=True).start()

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
            elif self.current_key is not None:
                cur_idx = self._key_to_idx.get(self.current_key, -1)
                if cur_idx > 0:
                    _, filename = self._all_files[cur_idx - 1]
                    key = str(filename.relative_to(self.image_dir))
                    try:
                        img_bytes = filename.read_bytes()
                    except OSError:
                        return
                    self.history_right.append(
                        (self.current_key, self.current_img_bytes, self.current_bboxes)
                    )
                    self._load_frame(key, img_bytes, [], needs_pause=True)

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
            while self.history_right:
                self.history_left.append(
                    (self.current_key, self.current_img_bytes, self.current_bboxes)
                )
                key, img_bytes, bboxes = self.history_right.pop()
                self._load_frame(key, img_bytes, bboxes, needs_pause=True)
            self._start_load_next()

        elif action == "toggle_skip":
            self._skipping = not self._skipping
            self.current_needs_pause = bool(self.current_bboxes) or not self._skipping

        elif action == "quit":
            self._save_active_annot(through_key=self.current_key)
            self._done = True
            threading.Thread(
                target=lambda: os.kill(os.getpid(), signal.CTRL_C_EVENT), daemon=True
            ).start()

    def jump_to(self, target_key: str):
        """
        Jump to a specific image key, starting a new annotation file for that gap.
        Must be called under self.lock.
        """
        # Save the current annotation file before leaving this gap.
        self._save_active_annot(through_key=self.current_key)

        # Find the index of the target key.
        target_idx = len(self._all_files)
        for i, (ts, fn) in enumerate(self._all_files):
            if str(fn.relative_to(self.image_dir)) >= target_key:
                target_idx = i
                break

        self._iter_idx = target_idx
        self._file_index = target_idx

        # Clear history and current frame.
        self.history_left.clear()
        self.history_right.clear()
        self.current_key = None
        self.current_img_bytes = None
        self.current_bboxes = []
        self._done = False

        # Start fresh annotation file for this gap (created lazily on first frame).
        self._active_annot = None
        self._active_annot_path = None

        # Reset bg model so old history doesn't bleed into the new time window,
        # then re-populate warmup frames from just before the new position.
        self.bg_model.reset()
        self._recent_skipped = deque()
        self._populate_recent_skipped_for_warmup()

    # ── State snapshot ────────────────────────────────────────────────────────

    def get_state(self) -> FrameState:
        """Return a serialisable snapshot of the current state. Call under self.lock."""
        active_images = self._active_annot.images if self._active_annot else {}
        # Prefer annotation from the active session; fall back to pre-existing.
        existing = active_images.get(self.current_key) or self._all_annot_images.get(self.current_key, [])
        labels = self._active_annot.labels if self._active_annot else self._labels_dict

        ts_str = None
        if self.current_key:
            try:
                ts_str = _key_to_dt(self.current_key).isoformat()
            except Exception:
                pass

        session_start_str = None
        if self._active_annot and self._active_annot.start:
            try:
                session_start_str = _key_to_dt(self._active_annot.start).isoformat()
            except Exception:
                pass

        return FrameState(
            key=self.current_key,
            file_index=self._file_index,
            total=self._total_files,
            blobs=self.current_bboxes,
            existing_annotations=existing,
            labels=labels,
            loading=self._loading,
            skipping=self._skipping,
            needs_pause=self.current_needs_pause,
            done=self._done,
            timestamp=ts_str,
            session_start=session_start_str,
        )

    def save_bboxes(self, key: str, bboxes: list):
        """Persist submitted bboxes for the given key. Call under self.lock."""
        if self._active_annot is None:
            return
        labeled = [b for b in bboxes if b.label is not None]
        if labeled:
            self._active_annot.images[key] = labeled
            self._all_annot_images[key] = labeled
        else:
            self._active_annot.images.pop(key, None)
            self._all_annot_images.pop(key, None)


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
            state._labels_dict[label_id] = name
            if state._active_annot is not None:
                state._active_annot.labels[label_id] = name
                state._save_active_annot(through_key=state.current_key)
            return jsonify({"labels": state._labels_dict})

    @app.route("/api/quit", methods=["POST"])
    def api_quit():
        with state.lock:
            state.advance("quit")
        return jsonify({"status": "shutting_down"})

    @app.route("/api/coverage")
    def api_coverage():
        with state.lock:
            if not state._all_files:
                return jsonify({"segments": []})

            all_ts = [ts for ts, _ in state._all_files]
            clusters = _compute_segments(state._all_files)

            result_segments = []
            for seg_start_ep, seg_end_ep in clusters:
                seg_start_dt = _ts_to_dt(seg_start_ep)
                seg_end_dt = _ts_to_dt(seg_end_ep)
                seg_range = portion.closed(seg_start_dt, seg_end_dt)

                covered_in_seg = state._covered & seg_range
                uncovered_in_seg = seg_range - state._covered

                gaps = []
                for atom in uncovered_in_seg:
                    if atom.empty or not isinstance(atom.lower, datetime):
                        continue
                    lo, hi = atom.lower, atom.upper
                    i_lo = bisect.bisect_left(all_ts, lo.timestamp())
                    i_hi = bisect.bisect_right(all_ts, hi.timestamp())
                    gaps.append({
                        "start": lo.isoformat(),
                        "end": hi.isoformat(),
                        "file_count": i_hi - i_lo,
                    })

                result_segments.append({
                    "start": seg_start_dt.isoformat(),
                    "end": seg_end_dt.isoformat(),
                    "covered": _interval_to_json(covered_in_seg),
                    "gaps": gaps,
                })

            return jsonify({"segments": result_segments})

    @app.route("/api/jump", methods=["POST"])
    def api_jump():
        data = request.get_json()
        target_key = data.get("target_key")
        target_ts_str = data.get("target_ts")  # ISO datetime string

        with state.lock:
            if not target_key and target_ts_str:
                target_dt = datetime.fromisoformat(target_ts_str)
                target_epoch = target_dt.timestamp()
                for ts, fn in state._all_files:
                    if ts >= target_epoch:
                        target_key = str(fn.relative_to(state.image_dir))
                        break

            if target_key:
                state.jump_to(target_key)
                state._start_load_next()

            return jsonify(state.get_state().to_dict())

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────
def get_parser():
    parser = argparse.ArgumentParser(description="Image annotation tool")
    parser.add_argument("image_dir", type=Path, help="Directory with images")
    parser.add_argument("bg_model_config", type=Path, help="Path to background model config")
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
    parser.add_argument(
        "--interval",
        nargs="*",
        default=None,
        help="two arguments 'start_key end_key' each in YYYY/MM/DD/HHMMSS format. "
        "If provided, this annotation session will run only for that interval.",
    )
    return parser


def run_app(
    image_dir: Path,
    bg_model: TimestampAwareBackgroundSubtractor,
    labels_dict: Optional[dict[str, str]] = None,
    skip_no_motion: bool = False,
    paused: bool = False,
    start_key: Optional[str] = None,
    end_key: Optional[str] = None,
):
    if labels_dict is None:
        labels_dict = {}

    existing_annot_files = load_all_annotation_files(image_dir)
    covered = compute_covered_interval(existing_annot_files)

    # Merge labels from all existing annotation files, with later files winning.
    merged_labels = {}
    for af, _ in existing_annot_files:
        merged_labels.update(af.labels)
    merged_labels.update(labels_dict)  # caller-supplied labels take precedence

    state = AnnotatorState(
        image_dir=image_dir,
        bg_model=bg_model,
        labels_dict=merged_labels,
        skip_no_motion=skip_no_motion,
        paused=paused,
        start_key=start_key,
        end_key=end_key,
        existing_annot_files=existing_annot_files,
        covered=covered,
    )

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
            state._save_active_annot(through_key=state.current_key)
        if state._active_annot_path:
            print(f"Annotations saved to {state._active_annot_path}")


def main(args):

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
    bg_model = TimestampAwareBackgroundSubtractor(**bg_config)

    start_key = None
    end_key = None
    if args.interval:
        assert len(args.interval) == 2
        start_key, end_key = args.interval

    run_app(
        image_dir=image_dir,
        bg_model=bg_model,
        skip_no_motion=args.skip_no_motion,
        paused=args.paused,
        start_key=start_key,
        end_key=end_key,
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
