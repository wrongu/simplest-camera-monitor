import time
from collections import deque
from dataclasses import dataclass
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


def _is_night_mode_image(img: cv.Mat, grayness_threshold: float = 0.01) -> bool:
    grayified = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    diff = np.abs(grayified / 255 - img / 255)
    return np.mean(diff) < grayness_threshold


class TimestampAwareBackgroundSubtractor(object):
    def __init__(
        self,
        history_seconds: float = 180.0,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        area_threshold: int = 500,
        shadow_correlation_threshold: float = 0.6,
        kernel_cleanup: int = 5,
        kernel_close: int = 11,
        default_fps: float = 0.1,
        reset_on_night_mode: bool = True,
    ):
        # Parameters for the OpenCV background model
        self._history_seconds = history_seconds
        self._var_threshold = var_threshold
        self._detect_shadows = detect_shadows
        self._recents: deque[tuple[float, cv.Mat]] = deque()
        self._last_calculated_fps = default_fps  # Default FPS, will be updated as we go

        # Parameters for post-processing cleanup
        self.area_threshold = area_threshold
        self.shadow_correlation_threshold = shadow_correlation_threshold
        self.kernel_cleanup = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (kernel_cleanup, kernel_cleanup)
        )
        self.kernel_close = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (kernel_close, kernel_close)
        )

        self.night_mode = False
        self.reset_on_night_mode = reset_on_night_mode

        self.model = None
        self.instantiate_model()

    @property
    def fps(self):
        if len(self._recents) < 2:
            # Not enough data to calculate FPS, return the last calculated FPS
            return self._last_calculated_fps
        time_deltas = [
            self._recents[i][0] - self._recents[i - 1][0]
            for i in range(1, len(self._recents))
        ]
        median_delta = sorted(time_deltas)[len(time_deltas) // 2]
        self._last_calculated_fps = 1.0 / max(median_delta, 1e-2)
        return self._last_calculated_fps

    @property
    def last_timestamp(self) -> float:
        return self._recents[-1][0] if self._recents else 0.0

    def reset(self):
        self._recents.clear()
        self.instantiate_model()

    def instantiate_model(self):
        fps_adaptive_history = int(self._history_seconds * self.fps)
        self.model = cv.createBackgroundSubtractorMOG2(
            history=fps_adaptive_history,
            varThreshold=self._var_threshold,
            detectShadows=self._detect_shadows,
        )

        for _, img in self._recents:
            self.model.apply(img)

    def apply(self, img: cv.Mat, t: Optional[float] = None) -> cv.Mat:
        if t is None:
            t = time.time()

        if t < self.last_timestamp:
            raise ValueError(
                f"Images must be given in chronological order but "
                f"last timestamp is {self.last_timestamp} and new timestamp is {t}."
            )

        # Check for night mode changes
        is_night_mode = _is_night_mode_image(img)
        if is_night_mode != self.night_mode:
            self.night_mode = is_night_mode
            if self.reset_on_night_mode:
                self.reset()

        # Remove all old images from the history
        while self._recents and self._recents[0][0] < t - self._history_seconds:
            self._recents.popleft()

        # Rule-of-thumb: if we are down to fewer than 10 images, re-instantiate the model
        if len(self._recents) < 10:
            self.instantiate_model()

        # Add the new image to the history and 'apply' it
        self._recents.append((t, img))
        return self.model.apply(img)

    def applyWithStats(
        self, img: cv.Mat, t: Optional[float] = None
    ) -> tuple[cv.Mat, list[ForegroundBlob]]:
        mask = self.apply(img, t)

        # No stats if we don't have enough images
        if len(self._recents) < 10:
            return mask, []

        # Get a copy of the avg bg image
        background = self.model.getBackgroundImage()

        # Clean up noise with opening, then reconnect components and group blobs together with closing
        clean_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, self.kernel_cleanup)
        clean_mask = cv.morphologyEx(clean_mask, cv.MORPH_CLOSE, self.kernel_close)

        # Check each connected component
        n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(clean_mask)

        blobs = []
        for i in range(1, n_labels):
            # Drop any tiny blobs
            if stats[i, cv.CC_STAT_AREA] < self.area_threshold:
                continue

            # Do a quick brightness scaling check to see if this is likely just a sunlight/shadow change
            background_blob = background[labels == i, :]
            current_blob = img[labels == i, :]

            brightness_correlation = np.corrcoef(
                current_blob.ravel(),
                background_blob.ravel(),
            )[0, 1]

            if brightness_correlation > self.shadow_correlation_threshold:
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

        return mask, blobs
