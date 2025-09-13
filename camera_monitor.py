from contextlib import contextmanager
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
from enum import Enum


ONE_DAY_SECONDS = 24 * 60 * 60


class State(Enum):
    INIT = -1
    RUNNING = 0
    CANT_CONNECT = 1
    CRASHED = 2
    REBOOT = 3


class ESPHomeCameraWrapper(object):
    def __init__(self, url: str):
        self.url = url
        self.last_frame = None
        self.last_frame_time = 0
        # TODO - consider using the event stream API and keeping a persistent app:camera http connection open
        self.components = {"switch/reboot": None, "switch/flash": None}
        self.poll_components()

    def get_frame(self):
        resp = requests.get(self.url, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv.imdecode(image_array, cv.IMREAD_COLOR)
            self.last_frame = frame
            self.last_frame_time = time.time()
            return frame
        else:
            raise ConnectionError(f"Failed to get frame: {resp.status_code}")

    def get_last_frame(self):
        if self.last_frame is not None and (time.time() - self.last_frame_time) < 5:
            return self.last_frame
        else:
            return self.get_frame()
        
    def poll_components(self):
        for comp in self.components.keys():
            try:
                resp = requests.get(self.url + f"/{comp}/", timeout=2)
                if resp.status_code == 200:
                    self.components[comp] = resp.json()
                else:
                    self.components[comp] = None
            except Exception as e:
                self.components[comp] = None

    def reboot(self) -> bool:
        if self.components.get("switch/reboot") is not None:
            try:
                resp = requests.post(self.url + "/switch/reboot/turn_on", timeout=5)
                return resp.status_code == 200
            except Exception as e:
                print(f"Failed to send reboot request: {e}")
                return False


class CameraMonitor(hass.Hass):
    def initialize(self):
        self.camera = ESPHomeCameraWrapper(self.args["url"])
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

        self.state_machine = State.INIT
        self.state_meta = None
        self.state_transition(State.RUNNING, since=time.time())

        self.reinitialize_bg_model_from_saved_images()

        self._last_cleanup_time = float("-inf")

        self.trigger_class = self.args["watch_for_class"]

        self.sensor = self.get_entity(self.args["hass_sensor_entity"])
        self.sensor.set_state("off")

        self.run_every(self.poll, interval=self.poll_frequency)

    def state_transition(self, new_state: State, **meta):
        old_state = self.state_machine
        if old_state != new_state:
            self.log(f"State transition {old_state} -> {new_state}")
            self.state_machine = new_state
            self.state_meta = meta

            if new_state in (State.CANT_CONNECT, State.CRASHED, State.REBOOT):
                self.sensor.set_state("unavailable")

    def poll(self, **kwargs):
        if self.state_machine == State.RUNNING:
            try:
                frame = self.camera.get_frame()
            except ConnectionError:
                self.state_transition(State.CANT_CONNECT, since=time.time())
            if frame is not None:
                self.process_frame(frame)
        
        elif self.state_machine == State.CANT_CONNECT:
            # For the first minute of not being able to connect, just keep trying
            if time.time() - self.state_meta.get("since", 0) < 60:
                try:
                    frame = self.camera.get_frame()
                    self.state_transition(State.RUNNING, since=time.time())
                    self.process_frame(frame)
                except ConnectionError:
                    pass
            # After a minute, try sending a reboot signal to the camera
            else:
                self.log("Attempting to reboot camera via HTTP request")
                if self.camera.reboot():
                    self.log("Reboot request sent successfully")
                    self.state_transition(State.REBOOT, since=time.time())
                else:
                    self.log("Failed to send reboot request", level="WARNING")
                    self.state_transition(State.CRASHED, since=time.time())
        
        elif self.state_machine == State.REBOOT:
            # Wait 30 seconds after sending the reboot command, then try to connect again
            if time.time() - self.state_meta.get("since", 0) > 30:
                try:
                    frame = self.camera.get_frame()
                    self.state_transition(State.RUNNING, since=time.time())
                    self.process_frame(frame)
                except ConnectionError:
                    self.log("Still cannot connect after reboot attempt", level="WARNING")
                    self.state_transition(State.CRASHED, since=time.time())

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

    def process_frame(self, frame, timestamp=None, detect_stuff=True):
        if timestamp is None:
            timestamp = time.time()

        # check if too dark
        if np.median(frame.ravel()) < self.brightness_threshold:
            return

        # save the image
        out_file = self.output_dir / create_timestamped_filename(timestamp, ".jpg")
        if not out_file.exists():
            self.log(f"Writing image to {out_file}")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(out_file), frame)

        # update bg model
        if detect_stuff:
            _, blobs = self.bg_model.applyWithStats(frame, timestamp)

            self.log(f"Detected {len(blobs)} blobs", level="DEBUG")

            # classify those buggers
            for i, blob in enumerate(blobs):
                pred_class_int = self.classifier.predict(featurize(blob).reshape(1, -1)).item()
                pred_label = self.label_lookup[pred_class_int]
                self.log(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))} CLASS {pred_label}",
                    level="INFO",
                )

                # Debugging: write out some images containing blob info
                blob_file = self.output_dir / "blobs" / create_timestamped_filename(timestamp, f"_{i}_{pred_label}.jpg")
                if not blob_file.exists():
                    blob_file.parent.mkdir(parents=True, exist_ok=True)
                    cv.imwrite(str(blob_file), (blob["mask"] * 255).astype(np.uint8))

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
