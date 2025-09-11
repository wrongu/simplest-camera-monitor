import appdaemon.plugins.hass.hassapi as hass
import cv2 as cv
import numpy as np
from pathlib import Path
import requests

from sklearn.naive_bayes import GaussianNB
from background_model import TimestampAwareBackgroundSubtractor
from image_loader import (
    create_timestamped_filename,
    ensure_files_timestamp_named,
    get_all_timestamped_files_sorted,
)
from classifier import featurize
import pickle
import time
from collections import deque


ONE_DAY_SECONDS = 24 * 60 * 60


class CameraMonitor(hass.Hass):
    def initialize(self):
        self.source = self.args["url"]
        self.brightness_threshold = int(self.args["brightness_threshold"])
        self.poll_frequency = float(self.args["poll_frequency"])
        with open(self.args["model_file"], "rb") as f:
            data = pickle.load(f)
        self.classifier: GaussianNB = data["model"]
        self.label_lookup: dict[int, str] = data["label_lookup"]

        self.log(f"Loaded model with labels: {self.label_lookup}")

        self.bg_model = TimestampAwareBackgroundSubtractor(
            history_seconds=self.args["history_seconds"],
            var_threshold=self.args["var_threshold"],
            detect_shadows=self.args["detect_shadows"],
            area_threshold=self.args["area_threshold"],
            shadow_correlation_threshold=self.args["shadow_correlation_threshold"],
            kernel_cleanup=self.args["kernel_cleanup"],
            kernel_close=self.args["kernel_close"],
            default_fps=self.args["default_fps"],
        )

        self.detections = deque([], maxlen=self.args["debounce"])

        self.output_dir = Path("/media/front camera/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reinitialize_bg_model_from_saved_images()

        self._last_cleanup_time = float("-inf")

        self.trigger_class = self.args["watch_for_class"]

        self.sensor = self.get_entity(self.args["hass_sensor_entity"])
        self.sensor.set_state("off")

        self.run_every(self.poll, interval=self.poll_frequency)

    def poll(self, **kwargs):
        frame = self.get_frame()
        if frame is not None:
            self.process_frame(frame)
        self.maybe_cleanup()

    def maybe_cleanup(self):
        now = time.time()
        if now - self._last_cleanup_time > ONE_DAY_SECONDS:
            ensure_files_timestamp_named(
                self.output_dir, dry_run=False, glob="**/*.jpg"
            )

            for t, f in get_all_timestamped_files_sorted(
                self.output_dir, glob="**/*.jpg"
            ):
                if now - t > 3 * ONE_DAY_SECONDS:
                    f.unlink()
            self._last_cleanup_time = now

    def get_frame(self):
        resp = requests.get(self.source, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv.imdecode(image_array, cv.IMREAD_COLOR)
            return frame
        else:
            self.log(
                f"Failed to get frame from source: {resp.status_code}", level="WARNING"
            )
            return None

    def process_frame(self, frame, timestamp=None, detect_stuff=True):
        if timestamp is None:
            timestamp = time.time()

        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return

        # save the image
        out_file = self.output_dir / create_timestamped_filename(timestamp, "jpg")
        if not out_file.exists():
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(out_file), frame)

        # update bg model
        if detect_stuff:
            _, blobs = self.bg_model.applyWithStats(frame, timestamp)

            # classify those buggers
            for blob in blobs:
                pred_class_int = self.classifier.predict(featurize(blob)).item()
                pred_label = self.label_lookup[pred_class_int]
                self.log(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))} CLASS {pred_label}",
                    level="INFO",
                )
                self.handle_detection(pred_label)
        else:
            self.bg_model.apply(frame, timestamp)

    def handle_detection(self, label: str):
        self.detections.append(label)
        if (
            all(d == self.trigger_class for d in self.detections)
            and len(self.detections) == self.detections.maxlen
        ):
            if self.sensor.state != "on":
                self.log(f"DETECTED {self.trigger_class}", level="WARNING")
                self.sensor.set_state("on")
        elif self.sensor.state != "off":
            self.log(f"DEACTIVATED {self.trigger_class}", level="INFO")
            self.sensor.set_state("off")

    def reinitialize_bg_model_from_saved_images(self):
        self.log("Reinitializing background model from saved images...")
        now = time.time()
        for t, f in get_all_timestamped_files_sorted(self.output_dir, glob="**/*.jpg"):
            if 0 < (now - t) < self.args["history_seconds"]:
                frame = cv.imread(str(f))
                if frame is not None:
                    self.log(f"\t{f}")
                    self.process_frame(frame, timestamp=None, detect_stuff=False)
        self.log("Reinitialization complete.")
