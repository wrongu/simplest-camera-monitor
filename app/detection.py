import pickle
import time
from pathlib import Path
from typing import Protocol, Optional, assert_never

import cv2 as cv
import numpy as np
import yaml
from ultralytics import YOLO

from app.background_model import BoundingBox, TimestampAwareBackgroundSubtractor
from app.classifier import featurize
from app.image_loader import get_all_timestamped_files_sorted
from app.utils import get_logger

logger = get_logger("detection", batching=300)


class DetectionModel(Protocol):
    def initialize_from_logs(self, log_dir: Path): ...

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]: ...


class BackgroundModelWithMorphologyClassifier(DetectionModel):
    def __init__(
        self,
        bg_model: TimestampAwareBackgroundSubtractor,
        model_file: str | Path,
        brightness_threshold: float = 0.0,
    ):
        self.brightness_threshold = brightness_threshold
        self.bg_model = bg_model
        with open(model_file, "rb") as f:
            model_metadata = pickle.load(f)
        self.classifier = model_metadata["model"]
        self.label_lookup: dict[int, str] = model_metadata["label_lookup"]
        logger.info(f"Loaded model with labels: {self.label_lookup}")

    def initialize_from_logs(self, log_dir: Path, now: Optional[float] = None):
        if log_dir is None:
            logger.warning("No output directory set, cannot reinitialize bg model")
            return
        logger.info("Reinitializing background model from saved images...")
        if now is None:
            now = time.time()
        for t, f in get_all_timestamped_files_sorted(log_dir, glob="20*/**/*.jpg"):
            if 0 < (now - t) < self.bg_model.history_seconds:
                frame = cv.imread(str(f))
                if frame is not None:
                    logger.info(f"\t{f}")
        logger.info("Reinitialization complete.")

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]:
        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return []

        _, blobs = self.bg_model.applyWithStats(frame, timestamp)

        # classify those buggers
        for i, blob in enumerate(blobs):
            if self.classifier is not None:
                try:
                    pred_class_int = self.classifier.predict(featurize(blob)[None, :]).item()
                    blob.bbox.class_id = self.label_lookup.get(pred_class_int, "???")
                except Exception as e:
                    logger.error(f"Classifier error: {e}")

        return [blob.bbox for blob in blobs]


class YoloDetectionModel(DetectionModel):
    def __init__(
        self,
        weights: Path,
        roi: Optional[str | Path] = None,
        brightness_threshold: float = 0.0,
    ):
        self.yolo = YOLO(weights, task="detect")
        self.roi_img = cv.imread(str(roi), cv.IMREAD_GRAYSCALE) if roi is not None else None
        self.brightness_threshold = brightness_threshold

    def is_in_roi(self, norm_x: float, norm_y: float):
        if self.roi_img is not None:
            h, w = self.roi_img.shape
            x, y = int(w * norm_x), int(h * norm_y)
            return self.roi_img[y, x] > 127
        return True

    def initialize_from_logs(self, log_dir: Path, now: Optional[float] = None):
        pass

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]:
        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return []

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detections = self.yolo.predict(frame_rgb)
        detections_out = []

        for det in detections:
            boxes = BoundingBox.from_yolo(det.boxes)
            for box in boxes:
                if self.is_in_roi(box.cx, box.cy):
                    detections_out.append(box)
        return detections_out


def create_detector(config_file: str | Path) -> DetectionModel:
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)

    match config.pop("class"):
        case "BackgroundModelWithMorphologyClassifier":
            return BackgroundModelWithMorphologyClassifier(
                bg_model=TimestampAwareBackgroundSubtractor(**config),
                model_file=config["model_file"],
                brightness_threshold=config.get("brightness_threshold", 10.0),
            )
        case "YoloDetectionModel":
            return YoloDetectionModel(**config)
        case _:
            assert_never(config["class"])
