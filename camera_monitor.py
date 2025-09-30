import pickle
import time
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from background_model import TimestampAwareBackgroundSubtractor, ForegroundBlob
from classifier import featurize
from image_loader import (
    create_timestamped_filename,
    get_all_timestamped_files_sorted,
)


class CameraMonitor(object):
    def __init__(
        self,
        brightness_threshold: int,
        history_seconds: float,
        bg_model: TimestampAwareBackgroundSubtractor,
        model_file: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        log=None,
    ):
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

        self.bg_model = bg_model

        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.reinitialize_bg_model_from_saved_images()

    @staticmethod
    def _save_image(path: Path, image: cv.Mat):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(path), image)

    def process_frame(
        self, frame: cv.Mat, timestamp: float, detect_stuff=True
    ) -> list[ForegroundBlob]:
        if timestamp is None:
            timestamp = time.time()

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
                    pred_class_int = self.classifier.predict(featurize(blob)).item()
                    blob.class_id = self.label_lookup.get(pred_class_int, "unknown")

                # Debugging: write out some images containing blob info
                if self.output_dir is not None:
                    if i == 0:
                        self._save_image(
                            self.output_dir
                            / "blobs"
                            / create_timestamped_filename(timestamp, ".jpg"),
                            frame,
                        )
                    blob_img = cv.normalize(blob, None, 255, 0, cv.NORM_MINMAX)
                    self._save_image(
                        self.output_dir
                        / "blobs"
                        / create_timestamped_filename(
                            timestamp, f"_{i}_{blob.class_id}.jpg"
                        ),
                        blob_img,
                    )
            return blobs

    def reinitialize_bg_model_from_saved_images(self, now=None):
        self.log("Reinitializing background model from saved images...")
        if now is None:
            now = time.time()
        for t, f in get_all_timestamped_files_sorted(self.output_dir, glob="20*/**/*.jpg"):
            if 0 < (now - t) < self.history_seconds:
                frame = cv.imread(str(f))
                if frame is not None:
                    self.log(f"\t{f}")
                    self.process_frame(frame, timestamp=t, detect_stuff=False)
        self.log("Reinitialization complete.")
