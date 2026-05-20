from collections import defaultdict
import logging
import time
from dataclasses import dataclass
from typing import Optional, Self

import cv2 as cv
import numpy as np

ONE_HOUR_SECONDS = 3600
LogLevel = int


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class LogHandler(logging.Handler):
    def __init__(self, batch_every: float = ONE_HOUR_SECONDS):
        super().__init__()
        self.batch_every = batch_every
        self.last_emit_time = defaultdict(float)
        self.message_count = defaultdict(int)

    def emit(self, record: logging.LogRecord) -> None:
        key = repr(record.msg)
        self.message_count[key] += 1
        now = time.time()
        if now - self.last_emit_time[key] >= self.batch_every:
            count = self.message_count[key]
            if count > 1:
                record.msg = f"{record.msg} (repeated {count} times)"
            print(record.getMessage())
            self.last_emit_time[key] = now
            self.message_count[key] = 0


def get_logger(name: str, level=logging.INFO, batching: int = ONE_HOUR_SECONDS) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(LogHandler(batching))
    return logger


@dataclass
class YoloBoundingBox:
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    class_id: int


@dataclass
class BoundingBox:
    x: int  # left
    y: int  # top
    width: int
    height: int
    confidence: float = 1.0
    class_id: Optional[str] = None

    def unscale(self, img_scale: float | tuple[float, float]) -> Self:
        if isinstance(img_scale, tuple):
            scale_x, scale_y = img_scale
        else:
            scale_x, scale_y = img_scale, img_scale

        self.x = int(round(self.x / scale_x))
        self.y = int(round(self.y / scale_y))
        self.width = int(round(self.width / scale_x))
        self.height = int(round(self.height / scale_y))
        return self

    def to_dict(self) -> dict:
        return {
            "bbox": [int(self.x), int(self.y), int(self.width), int(self.height)],
            "conf": self.confidence,
            "label": self.class_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(*d["bbox"], confidence=d["conf"], class_id=d["label"])

    @classmethod
    def from_yolo(cls, yolo_box: YoloBoundingBox, class_lookup: dict[int, str]) -> "BoundingBox":
        return cls(
            x=int(round(yolo_box.cx - yolo_box.w / 2)),
            y=int(round(yolo_box.cy - yolo_box.h / 2)),
            width=int(round(yolo_box.w)),
            height=int(round(yolo_box.h)),
            confidence=yolo_box.confidence,
            class_id=class_lookup[int(yolo_box.class_id)],
        )

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return self.x + self.width // 2

    @property
    def cy(self) -> float:
        return self.y + self.height // 2

    def set_bounds(self, x0, y0, x1, y1):
        left, top = min(x0, x1), min(y0, y1)
        right, bottom = max(x0, x1), max(y0, y1)
        self.x, self.y = left, top
        self.width, self.height = right - left, bottom - top

    def distance(self, x, y):
        # Calculate rectangle bounds
        left, top = self.x, self.y
        right, bottom = left + self.width, top + self.height

        if left <= x <= right and top <= y <= bottom:
            # If point is inside the rectangle, compute distance to nearest edge
            dist_left = abs(x - left)
            dist_right = abs(x - right)
            dist_top = abs(y - top)
            dist_bottom = abs(y - bottom)
            return min(dist_left, dist_right, dist_top, dist_bottom)
        else:
            # If point is outside the rectangle, find the nearest point on the rectangle and
            # calculate distance to that point
            clamped_x = max(left, min(x, right))
            clamped_y = max(top, min(y, bottom))
            return np.sqrt((x - clamped_x) ** 2 + (y - clamped_y) ** 2)

    @staticmethod
    def iou(bbox1: "BoundingBox", bbox2: "BoundingBox"):
        x1, y1, w1, h1 = bbox1.x, bbox1.y, bbox1.width, bbox1.height
        x2, y2, w2, h2 = bbox2.x, bbox2.y, bbox2.width, bbox2.height

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        intersection = inter_width * inter_height

        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def draw(self, img: cv.Mat, color: tuple[int, int, int]) -> cv.Mat:
        cv.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), color, 2)
        txt = str(self.class_id)
        if self.confidence is not None and self.confidence < 1.0:
            txt += f" [{self.confidence:.2f}]"
        cv.putText(img, txt, (self.x + 10, self.y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color)
        return img



__all__ = [
    "BoundingBox",
    "YoloBoundingBox",
    "LogHandler",
    "LogLevel",
    "get_logger",
]