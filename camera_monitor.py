import appdaemon.plugins.hass.hassapi as hass
import cv2 as cv
import numpy as np
from pathlib import Path
import requests

from sklearn.naive_bayes import GaussianNB
from background_model import TimestampAwareBackgroundSubtractor
from image_loader import get_all_timestamped_files_sorted
import pickle
import time
from collections import deque


class CameraMonitor(hass.Hass):
    def initialize(self):
        self.source = self.args["url"]
        self.min_area = int(self.args["min_area"])
        self.brightness_threshold = int(self.args["brightness_threshold"])
        self.poll_frequency = float(self.args["poll_frequency"])
        with open(self.args["model_file"], "rb") as f:
            self.classifier : GaussianNB = pickle.load(f)

        self.bg_model = TimestampAwareBackgroundSubtractor(
            history_seconds=self.args["history_seconds"],
            var_threshold=self.args["var_threshold"],
            detect_shadows=self.args["detect_shadows,"],
            area_threshold=self.args["area_threshold"],
            shadow_correlation_threshold=self.args["shadow_correlation_threshold"],
            kernel_cleanup=self.args["kernel_cleanup"],
            kernel_close=self.args["kernel_close"],
            default_fps=self.args["default_fps"]
        )

        self.detections = deque([], maxlen=self.args["debounce"])

        self.last_frame = None
        self.current_frame = None

        self.output_dir = Path("/media/front camera/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._last_cleanup_time = float("-inf")

        self.sensor = self.get_entity(self.args["hass_sensor_entity"])
        self.sensor.set_state("off")

        self.run_every(self.poll, interval = self.poll_frequency)


    def poll(self, **kwargs):
        frame = self.get_frame()
        if frame is not None:
            self.process_frame(frame)
        self.maybe_cleanup()
    

    def maybe_cleanup():
        now = time.time()
        if now - self._last_cleanup_time > 24 * 60 * 60:
            for t, f in get_all_timestamped_files_sorted(self.output_dir, glob="*.png"):
                if now - t > 5 * 24 * 60 * 60:
                    f.unlink()
            self._last_cleanup_time = now


    def get_frame(self):
        resp = requests.get(self.source, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv.imdecode(image_array, cv.IMREAD_COLOR)
            return frame
        else:
            print("Failed to get frame from source:", resp.status_code)
            return None


    def process_frame(self, frame):
        now = time.time()

        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return
        
        # save the image
        yyyymmdd_hhmmss = time.strftime("%Y%m%d_%H%M%S", now)
        out_file = self.output_dir / f"{yyyymmdd_hhmmss}.png"
        cv.imwrite(str(out_file), frame)

        # update bg model and check for blobs
        _, blobs = self.bg_model.applyWithStats(frame, now)

        # classify those buggers
        for blob in blobs:
            pred = self.classifier.predict(featurize(blob))
            print(yyyymmdd_hhmmss, "CLASS", pred)
            self.handle_detection(pred)

    def handle_detection(self, y):
        self.detections.append(y)
        if np.all(d == 1 for d in self.detections):
            self.sensor.set_state("on")
        else:
            self.sensor.set_state("off")
