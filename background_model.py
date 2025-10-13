import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class ForegroundBlob:
    background: cv.Mat
    image: cv.Mat
    area: int
    bbox: tuple[int, int, int, int]  # x, y, width, height
    mask: cv.Mat
    class_id: Optional[str] = None

    def shadow_correlation(self) -> tuple[float, float, float]:
        bg_bgr = self.background[self.mask > 0]
        im_bgr = self.image[self.mask > 0]
        return tuple(
            np.corrcoef(im, bg)[0, 1].item() for im, bg in zip(im_bgr.T, bg_bgr.T)
        )

    @staticmethod
    def iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

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


def _is_night_mode_image(img: cv.Mat, grayness_threshold: float = 0.01) -> bool:
    grayified = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    diff = np.abs(grayified / 255 - img / 255)
    return np.mean(diff) < grayness_threshold


class MorphologyFn:
    kernel: cv.Mat
    frac_threshold: float
    iterations: int

    def __init__(
        self, kern_radius: int = 5, frac_threshold: float = 0.5, iterations: int = 1
    ):
        kernel_size = 2 * kern_radius + 1
        self.kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        self.frac_threshold = frac_threshold
        self.iterations = iterations

    @property
    def radius(self) -> int:
        return (self.kernel.shape[0] - 1) // 2

    @radius.setter
    def radius(self, r: int):
        kernel_size = 2 * r + 1
        self.kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    def __call__(self, img: cv.Mat) -> cv.Mat:
        cleaned = img.copy() / 255
        for _ in range(self.iterations):
            cleaned = (
                cv.filter2D(cleaned, -1, self.kernel)
                >= int(np.sum(self.kernel) * self.frac_threshold)
            ).astype(np.float32)
        return np.clip(np.round(cleaned * 255), 0, 255).astype(np.uint8)


class TimestampAwareBackgroundSubtractor(object):
    def __init__(
        self,
        history_seconds: float = 180.0,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        area_threshold: int = 500,
        shadow_correlation_threshold: float = 0.6,
        morph_radius: int = 5,
        morph_thresh: float = 0.3,
        morph_iters: int = 3,
        default_fps: float = 0.1,
        region_of_interest: Optional[cv.Mat] = None,
        night_mode_kwargs: Optional[dict] = None,
        debug_dir: Optional[Path] = None,
    ):
        # Parameters for the OpenCV background model
        self.history_seconds = history_seconds
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.default_fps = default_fps
        self.fps = default_fps
        self.last_timestamp = 0.0
        self.update_fps(delta_t=1.0 / default_fps)

        # Parameters for post-processing cleanup
        self.area_threshold = area_threshold
        self.shadow_correlation_threshold = shadow_correlation_threshold
        self.morphology_fn = MorphologyFn(
            kern_radius=morph_radius,
            frac_threshold=morph_thresh,
            iterations=morph_iters,
        )

        self.night_mode = False
        self._night_mode_kwargs = night_mode_kwargs or {}
        self._day_mode_kwargs = {
            k: self.__dict__[k] for k in self._night_mode_kwargs.keys()
        }

        self.model = cv.createBackgroundSubtractorMOG2(
            history=int(history_seconds * default_fps),
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

        self.roi = region_of_interest
        if self.roi is not None:
            self.roi = self.roi / max(1.0, np.max(self.roi))

        self.debug_dir = debug_dir

    def setArgs(self, **kwargs):
        for k, v in kwargs.items():
            match k:
                case "var_threshold" | "detect_shadows" | "history_seconds":
                    if self.__dict__[k] != v:
                        setattr(self, k, v)
                    # If model params are updated, we need to re-instantiate the model itself
                    self.model = cv.createBackgroundSubtractorMOG2(
                        history=int(self.history_seconds * self.fps),
                        varThreshold=self.var_threshold,
                        detectShadows=self.detect_shadows,
                    )
                case "area_threshold":
                    self.area_threshold = v
                case "shadow_correlation_threshold":
                    self.shadow_correlation_threshold = v
                case "morph_radius":
                    self.morphology_fn.radius = v
                case "morph_thresh":
                    self.morphology_fn.thresh = v
                case "morph_iters":
                    self.morphology_fn.iterations = v
                case "default_fps":
                    self.default_fps = float(v)
                    self.update_fps(delta_t=1.0 / self.default_fps, ema_alpha=1.0)

    def update_fps(self, delta_t: float, ema_alpha: float = 0.1):
        # Exponential moving average, or reset to default if delta_t is very large
        if self.fps is None or delta_t > self.history_seconds / 2:
            self.fps = self.default_fps
        else:
            self.fps += (1.0 / delta_t - self.fps) * ema_alpha

    def apply(self, img: cv.Mat, t: Optional[float] = None) -> cv.Mat:
        if t is None:
            t = time.time()

        if t <= self.last_timestamp:
            raise ValueError(
                f"Images must be given in chronological order but "
                f"last timestamp is {self.last_timestamp} and new timestamp is {t}."
            )

        delta_t = t - self.last_timestamp
        learning_rate = min(delta_t / self.history_seconds, 1.0)
        self.update_fps(delta_t)

        # Check for night mode changes
        is_night_mode = _is_night_mode_image(img)
        if is_night_mode != self.night_mode:
            # Updates with a 'learning_rate' of 1.0 will discard old information and treat the
            # current image as the entire background
            learning_rate = 1.0
            self.night_mode = is_night_mode
            new_kwargs = (
                self._night_mode_kwargs if is_night_mode else self._day_mode_kwargs
            )
            self.setArgs(**new_kwargs)

        # Update the bg model with a time-adaptive learning rate and get the fg mask
        self.last_timestamp = t
        return self.model.apply(img, learning_rate)

    def applyWithStats(
        self, img: cv.Mat, t: Optional[float] = None
    ) -> tuple[cv.Mat, list[ForegroundBlob]]:
        mask = self.apply(img, t)

        # Get a copy of the avg bg image
        background = self.model.getBackgroundImage()

        # Clean up noise with opening, then reconnect components and group blobs together with closing
        clean_mask = np.where(self.morphology_fn(mask) > 0, mask, 0).astype(np.uint8)

        if self.debug_dir is not None:
            cv.imwrite(str(self.debug_dir / "background.png"), background)
            cv.imwrite(str(self.debug_dir / "current.png"), img)
            cv.imwrite(str(self.debug_dir / "raw_mask.png"), mask)
            cv.imwrite(str(self.debug_dir / "clean_mask.png"), clean_mask)
            shadow_info = cv.cvtColor(clean_mask, cv.COLOR_GRAY2BGR)

        # Check each connected component
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(clean_mask)

        blobs = []
        for i in range(1, n_labels):
            # Drop any tiny blobs
            if stats[i, cv.CC_STAT_AREA] < self.area_threshold:
                continue

            if self.roi is not None:
                pixels_in_roi = np.sum(self.roi[labels == i])
                if pixels_in_roi / stats[i, cv.CC_STAT_AREA] < 0.5:
                    continue

            # Append this foreground blob info to the list; note that each blob shares the same
            # background and image object references but has a different mask
            blobs.append(
                ForegroundBlob(
                    background=background,
                    image=img,
                    area=stats[i, cv.CC_STAT_AREA],
                    bbox=stats[i, :4],  # x, y, width, height
                    mask=np.array(labels == i).astype(np.uint8) * clean_mask,
                )
            )

            if self.debug_dir is not None:
                cv.putText(
                    shadow_info,
                    f"{np.mean(blobs[-1].shadow_correlation()):.3f}",
                    (stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP] + 15),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 200),
                    2,
                )

        if self.debug_dir is not None:
            cv.imwrite(str(self.debug_dir / "shadow.png"), shadow_info)

        return mask, blobs
