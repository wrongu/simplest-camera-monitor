import time
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

import cv2 as cv

from app.cameras import Camera
from app.detection import DetectionModel
from app.image_loader import (
    create_timestamped_filename,
    get_all_timestamped_files_sorted,
    ensure_files_timestamp_named,
)
from app.utils import get_logger, BoundingBox

logger = get_logger("camera_monitor", batching=300)


ONE_DAY_SECONDS = 24 * 60 * 60


class State(Enum):
    INIT = -1
    RUNNING = 0
    CANT_CONNECT = 1
    CRASHED = 2
    REBOOT = 3


OnGetImageCallback = Callable[["CameraMonitor", cv.Mat, float], None]
OnStateTransitionCallback = Callable[["CameraMonitor", State], None]
OnDetectionCallback = Callable[["CameraMonitor", list[BoundingBox]], None]


class CameraMonitor(object):
    def __init__(
        self,
        camera: Camera,
        name: str,
        detection_model: Optional[DetectionModel] = None,
        output_dir: Optional[Path | str] = None,
        log_lifespan: int = ONE_DAY_SECONDS,
        on_get_image: Optional[OnGetImageCallback] = None,
        on_state_transition: Optional[OnStateTransitionCallback] = None,
        on_detection: Optional[OnDetectionCallback] = None,
        confidence_threshold: float = 0.5,
        roi: Optional[str | Path] = None,
    ):
        self.camera = camera
        self.name = name
        self.detector = detection_model
        self.confidence_threshold = confidence_threshold

        if roi is None:
            self._roi_image = None
        else:
            self._roi_image = cv.resize(
                cv.imread(str(roi), cv.IMREAD_GRAYSCALE),
                self.camera.resolution,
                interpolation=cv.INTER_AREA,
            )

        self.last_timestamp = 0

        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.detector is not None:
                self.detector.initialize_from_logs(self.output_dir)
        self.log_lifespan = log_lifespan
        self.cleanup_files()

        self.on_get_image = on_get_image
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
            logger.info(f"State transition {old_state} -> {new_state}")
            self.state_machine = new_state
            self.state_meta = meta

            if self.on_state_transition is not None:
                try:
                    self.on_state_transition(self, new_state)
                except Exception as e:
                    logger.error(f"Error in on_state_transition: {e}")

    def poll(self):
        if self.state_machine == State.RUNNING:
            try:
                timestamp, frame = self.camera.get_frame()
                if frame is not None and timestamp > self.last_timestamp:
                    self.log_frame(frame, timestamp)
                    if self.detector is not None:
                        detections = self.detector.process_frame(frame, timestamp)
                        self.handle_detections(detections)
            except ConnectionError:
                self.state_transition(State.CANT_CONNECT, since=time.time())

        elif self.state_machine == State.CANT_CONNECT:
            # For the first minute of not being able to connect, just keep trying
            if time.time() - self.state_meta.get("since", 0) < 60:
                try:
                    timestamp, frame = self.camera.get_frame()
                    self.state_transition(State.RUNNING, since=time.time())
                    self.log_frame(frame, timestamp=timestamp)
                except ConnectionError:
                    pass
            # After a minute, try sending a reboot signal to the camera
            else:
                logger.info("Attempting to reboot camera via HTTP request")
                if self.camera.reboot():
                    logger.info("Reboot request sent successfully")
                    self.state_transition(State.REBOOT, since=time.time())
                else:
                    logger.warning("Failed to send reboot request")
                    self.state_transition(State.CRASHED, since=time.time())

        elif self.state_machine == State.REBOOT:
            # Wait 30 seconds after sending the reboot command, then try to connect again
            if time.time() - self.state_meta.get("since", 0) > 30:
                try:
                    timestamp, frame = self.camera.get_frame()
                    self.state_transition(State.RUNNING, since=time.time())
                    self.log_frame(frame, timestamp=timestamp)
                except ConnectionError:
                    logger.warning("Still cannot connect after reboot attempt")
                    self.state_transition(State.CRASHED, since=time.time())

        elif self.state_machine == State.CRASHED:
            # Retry every 5 minutes in case the camera eventually recovers
            if time.time() - self.state_meta.get("since", 0) > 300:
                logger.info("Retrying crashed camera...")
                if self.camera.reboot():
                    self.state_transition(State.REBOOT, since=time.time())
                else:
                    logger.warning("Retry failed, staying in CRASHED")
                    self.state_meta["since"] = time.time()

    def handle_detections(self, detected_things: list[BoundingBox]):
        if self.on_detection is not None:
            # Filter by ROI and confidence
            detected_things = [
                box
                for box in detected_things
                if self._is_in_roi(box) and box.confidence >= self.confidence_threshold
            ]
            # Dispatch to callbacks
            try:
                self.on_detection(self, detected_things)
            except Exception as e:
                logger.error(f"Error in on_detection callback: {e}")

    @staticmethod
    def _save_image(path: Path, image: cv.Mat):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(path), image)

    def log_frame(self, frame: cv.Mat, timestamp: float) -> None:
        if timestamp <= self.last_timestamp:
            return

        if self.on_get_image is not None:
            self.on_get_image(self, frame, timestamp)

        self.last_timestamp = timestamp

        # save the image
        if self.output_dir is not None:
            self._save_image(
                self.output_dir / create_timestamped_filename(timestamp, ".jpg"), frame
            )

    def cleanup_files(self):
        if self.output_dir is None:
            return

        logger.info("Starting cleanup")
        now = time.time()

        # Check that all logged filenames are appropriately timestamped
        ensure_files_timestamp_named(self.output_dir, dry_run=False, glob="**/*.jpg")

        # Delete images that are older than 24h and detected blobs that are older than 72h
        n_images_deleted = 0
        for t, f in get_all_timestamped_files_sorted(self.output_dir, glob="20*/**/*.jpg"):
            if now - t > self.log_lifespan:
                f.unlink()
                n_images_deleted += 1
        logger.info(f"Deleted {n_images_deleted} old images")

        # Remove any remaining empty directories
        for path, subdirs, files in self.output_dir.walk(top_down=False):
            if not files and not subdirs and path != self.output_dir:
                path.rmdir()
        logger.info("Cleanup complete.")

    def _is_in_roi(self, box: BoundingBox):
        if self._roi_image is not None:
            x, y = int(round(box.cx)), int(round(box.cy))
            if x < 0 or y < 0 or x >= self.camera.resolution[0] or y >= self.camera.resolution[1]:
                return False
            return self._roi_image[y, x] > 127
        return True
