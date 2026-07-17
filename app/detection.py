import json
import pickle
import time
from pathlib import Path
from typing import Protocol, Optional, assert_never, NamedTuple
from ast import literal_eval

import cv2 as cv
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from onnxruntime import InferenceSession as OnnxSession
except ImportError:
    OnnxSession = None

from app.background_model import TimestampAwareBackgroundSubtractor
from app.classifier import featurize
from app.image_loader import get_all_timestamped_files_sorted
from app.utils import get_logger, YoloBoundingBox, BoundingBox

logger = get_logger("detection", batching=300)


class DetectionModel(Protocol):
    @property
    def classes(self) -> list[str]: ...

    def initialize_from_logs(self, log_dir: Path): ...

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]: ...


# Note on batching: one wonders if it might be faster to handle images from multiple cameras with a single 'batched'
# inference pass rather than calling process_frame() serially for each camera. Batching might help for the YOLO
# detection subclass, but for ONNX it requires a special model export flag. How to handle cases where the monitor calls
# for batched inference when the model might or might not support it seems complicated and not worth it (for now).


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

    @property
    def classes(self) -> list[str]:
        return list(self.label_lookup.values())

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
    def __init__(self, weights: Path, brightness_threshold: float = 0.0):
        self.yolo = YOLO(weights, task="detect")
        self.brightness_threshold = brightness_threshold

    @property
    def classes(self) -> list[str]:
        return list(self.yolo.names.values())

    def initialize_from_logs(self, log_dir: Path, now: Optional[float] = None):
        pass

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]:
        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return []

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detections = self.yolo.predict(frame_rgb)
        yolo_boxes = [
            YoloBoundingBox(
                *yolo_box.xywh.flatten().cpu().numpy(),
                yolo_box.conf.item(),
                yolo_box.cls.cpu().numpy().astype(int).item(),
            )
            for det in detections
            for yolo_box in det.boxes
        ]

        return [BoundingBox.from_yolo(yolo_box, self.yolo.names) for yolo_box in yolo_boxes]


class OnnxYoloDetectionModel(DetectionModel):
    def __init__(self, weights: Path, brightness_threshold: float = 0.0):
        self.model = OnnxSession(weights)
        meta = self.model.get_modelmeta()
        self.img_size = tuple(literal_eval(meta.custom_metadata_map["imgsz"]))
        self.class_lookup = literal_eval(meta.custom_metadata_map["names"])
        self.brightness_threshold = brightness_threshold

    @property
    def classes(self) -> list[str]:
        return list(self.class_lookup.values())

    def prep_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if (w, h) != self.img_size:
            pad_w, pad_h = self.img_size[0] - w, self.img_size[1] - h
            if pad_w < 0 or pad_h < 0:
                image = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
            else:
                image = np.pad(
                    image, [(0, pad_h), (0, pad_w), (0, 0)], mode="constant", constant_values=127
                )
        return (image.transpose(2, 0, 1)[None, ::-1, :, :] / 255).astype(np.float32)

    def initialize_from_logs(self, log_dir: Path, now: Optional[float] = None):
        pass

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]:
        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return []

        aspect_w = self.img_size[0] / frame.shape[1]
        aspect_h = self.img_size[1] / frame.shape[0]

        raw_output = self.model.run(None, {"images": self.prep_image(frame)})[0][0]
        onnx_bboxes = [YoloBoundingBox(*row) for row in raw_output]
        return [
            BoundingBox.from_yolo(box, self.class_lookup).unscale((aspect_w, aspect_h))
            for box in onnx_bboxes
        ]


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
            if YOLO is None:
                raise ValueError("YOLO is not available; try `pip install ultralytics`")
            return YoloDetectionModel(**config)
        case "OnnxYoloDetectionModel":
            if OnnxSession is None:
                raise ValueError("ONNX Runtime is not available; try `pip install onnxruntime`")
            return OnnxYoloDetectionModel(**config)
        case _:
            assert_never(config["class"])
