import pickle
import time
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

import cv2 as cv
import numpy as np

from background_model import TimestampAwareBackgroundSubtractor, BoundingBox
from cameras import Camera
from classifier import featurize
from image_loader import (
    create_timestamped_filename,
    get_all_timestamped_files_sorted,
    ensure_files_timestamp_named,
)

ONE_DAY_SECONDS = 24 * 60 * 60


class State(Enum):
    INIT = -1
    RUNNING = 0
    CANT_CONNECT = 1
    CRASHED = 2
    REBOOT = 3


OnStateTransitionCallback = Callable[["CameraMonitor", State], None]
OnDetectionCallback = Callable[["CameraMonitor", State], None]


class CameraMonitor(object):
    def __init__(
        self,
        camera: Camera,
        name: str,
        brightness_threshold: int,
        history_seconds: float,
        bg_model: TimestampAwareBackgroundSubtractor,
        save_blobs: bool = False,
        model_file: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        log_lifespan: int = ONE_DAY_SECONDS,
        log=None,
        on_state_transition: Optional[OnStateTransitionCallback] = None,
        on_detection: Optional[OnDetectionCallback] = None,
    ):
        self.camera = camera
        self.name = name
        self.log = log or (lambda msg, level="INFO": print(f"[{level}] {msg}"))
        self.brightness_threshold = int(brightness_threshold)
        self.history_seconds = float(history_seconds)
        if model_file is not None:
            with open(model_file, "rb") as f:
                model_metadata = pickle.load(f)
            self.classifier = model_metadata["model"]
            self.label_lookup: dict[int, str] = model_metadata["label_lookup"]
            self.log(f"Loaded model with labels: {self.label_lookup}")
        else:
            self.classifier = None
            self.label_lookup = {}
            self.log("No model file provided, classifier disabled.", level="WARNING")

        self.save_blobs = save_blobs
        self.bg_model = bg_model
        self.last_timestamp = 0

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.reinitialize_bg_model_from_saved_images()
        self.log_lifespan = log_lifespan
        self.cleanup_files()

        self.on_state_transition = on_state_transition
        self.on_detection = on_detection

        self.state_machine = State.INIT
        self.state_meta = {}
        self.state_transition(State.RUNNING, since=time.time())

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def state_transition(self, new_state: State, **meta):
        old_state = self.state_machine
        if old_state != new_state:
            self.log(f"State transition {old_state} -> {new_state}")
            self.state_machine = new_state
            self.state_meta = meta

            if self.on_state_transition is not None:
                try:
                    self.on_state_transition(self, new_state)
                except Exception as e:
                    self.log(f"Error in on_state_transition: {e}", level="ERROR")

    def poll(self):
        if self.state_machine == State.RUNNING:
            try:
                timestamp, frame = self.camera.get_frame()
                if frame is not None and timestamp > self.last_timestamp:
                    foreground_objects = self.process_frame(
                        frame,
                        timestamp=timestamp,
                        detect_stuff=True,
                        save_blobs=self.save_blobs,
                    )
                    self.handle_detections(foreground_objects)
            except ConnectionError:
                self.state_transition(State.CANT_CONNECT, since=time.time())

        elif self.state_machine == State.CANT_CONNECT:
            # For the first minute of not being able to connect, just keep trying
            if time.time() - self.state_meta.get("since", 0) < 60:
                try:
                    timestamp, frame = self.camera.get_frame()
                    self.state_transition(State.RUNNING, since=time.time())
                    self.process_frame(frame, timestamp=timestamp, detect_stuff=False)
                except ConnectionError:
                    pass
            # After a minute, try sending a reboot signal to the camera
            else:
                self.log("Attempting to reboot camera via HTTP request")
                if self.camera.reboot():
                    self.log("Reboot request sent successfully")
                    self.state_transition(State.REBOOT, since=time.time())
                else:
                    self.log("Failed to send reboot request", level="WARNING")
                    self.state_transition(State.CRASHED, since=time.time())

        elif self.state_machine == State.REBOOT:
            # Wait 30 seconds after sending the reboot command, then try to connect again
            if time.time() - self.state_meta.get("since", 0) > 30:
                try:
                    timestamp, frame = self.camera.get_frame()
                    self.state_transition(State.RUNNING, since=time.time())
                    self.process_frame(frame, timestamp=timestamp, detect_stuff=False)
                except ConnectionError:
                    self.log(
                        "Still cannot connect after reboot attempt", level="WARNING"
                    )
                    self.state_transition(State.CRASHED, since=time.time())

        elif self.state_machine == State.CRASHED:
            # Retry every 5 minutes in case the camera eventually recovers
            if time.time() - self.state_meta.get("since", 0) > 300:
                self.log("Retrying crashed camera...")
                if self.camera.reboot():
                    self.state_transition(State.REBOOT, since=time.time())
                else:
                    self.log("Retry failed, staying in CRASHED", level="WARNING")
                    self.state_meta["since"] = time.time()

    def handle_detections(self, detected_things: list[BoundingBox]):
        if self.on_detection is not None:
            try:
                self.on_detection(self, detected_things)
            except Exception as e:
                self.log(f"Error in on_detection callback: {e}", level="ERROR")

    @staticmethod
    def _save_image(path: Path, image: cv.Mat):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(path), image)

    def process_frame(
        self, frame: cv.Mat, timestamp: float, detect_stuff=True, save_blobs=True
    ) -> list[BoundingBox]:
        if timestamp <= self.last_timestamp:
            return []

        self.last_timestamp = timestamp

        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return []

        # save the image
        if self.output_dir is not None:
            self._save_image(
                self.output_dir / create_timestamped_filename(timestamp, ".jpg"), frame
            )

        # update bg model and maybe detect stuff
        if not detect_stuff:
            self.bg_model.apply(frame, timestamp)
            return []
        else:
            _, blobs = self.bg_model.applyWithStats(frame, timestamp)

            # classify those buggers
            for i, blob in enumerate(blobs):
                if self.classifier is not None:
                    try:
                        pred_class_int = self.classifier.predict(
                            featurize(blob)[None, :]
                        ).item()
                        blob.bbox.class_id = self.label_lookup.get(
                            pred_class_int, "???"
                        )
                    except Exception as e:
                        self.log(f"Classifier error: {e}", level="ERROR")

                # Debugging: write out some images containing blob info
                if save_blobs and self.output_dir is not None:
                    if i == 0:
                        self._save_image(
                            self.output_dir
                            / "blobs"
                            / create_timestamped_filename(timestamp, ".jpg"),
                            frame,
                        )
                    blob_img = cv.normalize(blob.mask, None, 255, 0, cv.NORM_MINMAX)
                    self._save_image(
                        self.output_dir
                        / "blobs"
                        / create_timestamped_filename(
                            timestamp, f"_{i}_{blob.bbox.class_id}.jpg"
                        ),
                        blob_img,
                    )
            return [blob.bbox for blob in blobs]

    def reinitialize_bg_model_from_saved_images(self, now=None):
        self.log("Reinitializing background model from saved images...")
        if now is None:
            now = time.time()
        for t, f in get_all_timestamped_files_sorted(
            self.output_dir, glob="20*/**/*.jpg"
        ):
            if 0 < (now - t) < self.history_seconds:
                frame = cv.imread(str(f))
                if frame is not None:
                    self.log(f"\t{f}")
                    self.process_frame(frame, timestamp=t, detect_stuff=False)
        self.log("Reinitialization complete.")

    def cleanup_files(self):
        self.log("Starting cleanup")
        now = time.time()

        # Check that all logged filenames are appropriately timestamped
        ensure_files_timestamp_named(self.output_dir, dry_run=False, glob="**/*.jpg")

        # Delete images that are older than 24h and detected blobs that are older than 72h
        n_images_deleted = 0
        for t, f in get_all_timestamped_files_sorted(
            self.output_dir, glob="20*/**/*.jpg"
        ):
            if now - t > self.log_lifespan:
                f.unlink()
                n_images_deleted += 1
        self.log(f"Deleted {n_images_deleted} old images")

        n_blobs_deleted = 0
        for t, f in get_all_timestamped_files_sorted(
            self.output_dir, glob="blobs/**/*.jpg"
        ):
            if now - t > self.log_lifespan:
                f.unlink()
                n_blobs_deleted += 1
        self.log(f"Deleted {n_blobs_deleted} old blobs files")

        # Remove any remaining empty directories
        for path, subdirs, files in self.output_dir.walk(top_down=False):
            if not files and not subdirs and path != self.output_dir:
                path.rmdir()
        self.log("Cleanup complete.")
