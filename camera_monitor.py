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

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.reinitialize_bg_model_from_saved_images()

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
            out_file = self.output_dir / create_timestamped_filename(timestamp, ".jpg")
            if not out_file.exists():
                self.log(f"Writing image to {out_file}")
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv.imwrite(str(out_file), frame)

        # update bg model and maybe detect stuff
        if not detect_stuff:
            self.bg_model.apply(frame, timestamp)
            return []
        else:
            _, blobs = self.bg_model.applyWithStats(frame, timestamp)

            self.log(f"Detected {len(blobs)} blobs", level="DEBUG")

            if self.classifier is None:
                return blobs

            # classify those buggers
            for i, blob in enumerate(blobs):
                pred_class_int = self.classifier.predict(featurize(blob)).item()
                blob.class_id = self.label_lookup.get(pred_class_int, "unknown")
                self.log(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))} CLASS {blob.class_id}",
                    level="INFO",
                )

                # Debugging: write out some images containing blob info
                if self.output_dir is not None:
                    blob_file = (
                        self.output_dir
                        / "blobs"
                        / create_timestamped_filename(
                            timestamp, f"_{i}_{blob.class_id}.jpg"
                        )
                    )
                    if not blob_file.exists():
                        blob_file.parent.mkdir(parents=True, exist_ok=True)
                        cv.imwrite(
                            str(blob_file),
                            cv.normalize(blob.mask, None, 0, 255, cv.NORM_MINMAX),
                        )
            return blobs

    def reinitialize_bg_model_from_saved_images(self):
        self.log("Reinitializing background model from saved images...")
        now = time.time()
        for t, f in get_all_timestamped_files_sorted(self.output_dir, glob="**/*.jpg"):
            if 0 < (now - t) < self.history_seconds:
                frame = cv.imread(str(f))
                if frame is not None:
                    self.log(f"\t{f}")
                    self.process_frame(frame, timestamp=t, detect_stuff=False)
        self.log("Reinitialization complete.")
