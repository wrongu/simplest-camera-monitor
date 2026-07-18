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

        self._uid = bg_model._uid + (str(model_file), brightness_threshold)

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

        self._uid = (str(weights), brightness_threshold)

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
        # Docs say that exported imgsz value is (height, width). It's plausible that this is (width, height), but in
        # our case they are expected to be equal. If we get it wrong then the inference pass should throw an error.
        self.img_size = tuple(literal_eval(meta.custom_metadata_map["imgsz"]))
        self.class_lookup = literal_eval(meta.custom_metadata_map["names"])
        self.brightness_threshold = brightness_threshold

        self._uid = (str(weights), brightness_threshold)

    @property
    def classes(self) -> list[str]:
        return list(self.class_lookup.values())

    def letterbox(
        self, img: cv.Mat, gray=(114, 114, 114), scaleup=True
    ) -> tuple[cv.Mat, float, tuple[int, int]]:
        """Aspect-preserving resize + gray padding.

        Returns the padded image, the scale ratio r, and (dw, dh) one-sided padding.
        """
        shape = img.shape[:2]  # (h, w)

        r: float = min(self.img_size[0] / shape[0], self.img_size[1] / shape[1])
        if not scaleup:  # only downscale (better val mAP); predict uses True
            r = min(r, 1.0)

        new_unpad = (round(shape[1] * r), round(shape[0] * r))  # (w, h)
        # Floating point half-padding
        dw: float = (self.img_size[1] - new_unpad[0]) / 2
        dh: float = (self.img_size[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)

        # Even/odd-aware int padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=gray)
        return img, r, (left, top)

    def prep_image(self, image: cv.Mat) -> tuple[cv.Mat, float, tuple[int, int]]:
        image, r, (left, top) = self.letterbox(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        image_bchw = (image.transpose(2, 0, 1)[None] / 255).astype(np.float32)
        return image_bchw, r, (left, top)

    def initialize_from_logs(self, log_dir: Path, now: Optional[float] = None):
        pass

    def process_frame(self, frame: cv.Mat, timestamp: float) -> list[BoundingBox]:
        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return []

        frame_bchw, scaled_by, (left, top) = self.prep_image(frame)
        raw_output = self.model.run(None, {"images": frame_bchw})[0][0]
        x1, y1, x2, y2, conf, cls = raw_output.T
        names = [self.class_lookup[c] for c in cls]
        x1 = ((x1 - left) / scaled_by).round().astype(int).tolist()
        x2 = ((x2 - left) / scaled_by).round().astype(int).tolist()
        y1 = ((y1 - top) / scaled_by).round().astype(int).tolist()
        y2 = ((y2 - top) / scaled_by).round().astype(int).tolist()
        conf = conf.tolist()
        return [
            BoundingBox(
                x=min(_x1, _x2),
                y=min(_y1, _y2),
                width=abs(_x2 - _x1),
                height=abs(_y2 - _y1),
                class_id=_n,
                confidence=_c,
            )
            for _x1, _y1, _x2, _y2, _n, _c in zip(x1, y1, x2, y2, names, conf)
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
