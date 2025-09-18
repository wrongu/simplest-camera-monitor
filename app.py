import time
from enum import Enum

import appdaemon.plugins.hass.hassapi as hass
import cv2 as cv
import numpy as np
import requests

from background_model import ForegroundBlob
from camera_monitor import CameraMonitor
from image_loader import ensure_files_timestamp_named, get_all_timestamped_files_sorted

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

    def get_frame(self) -> cv.Mat:
        resp = requests.get(self.url, timeout=10)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            frame = cv.imdecode(image_array, cv.IMREAD_COLOR)
            self.last_frame = frame
            self.last_frame_time = time.time()
            return frame
        else:
            raise ConnectionError(f"Failed to get frame: {resp.status_code}")

    def get_last_frame(self) -> cv.Mat:
        if self.last_frame is not None and (time.time() - self.last_frame_time) < 5:
            return self.last_frame
        else:
            return self.get_frame()

    def poll_components(self) -> None:
        for comp in self.components.keys():
            try:
                resp = requests.get(self.url + f"/{comp}/", timeout=2)
                if resp.status_code == 200:
                    self.components[comp] = resp.json()
                else:
                    self.components[comp] = None
            except requests.exceptions.RequestException:
                self.components[comp] = None

    def reboot(self) -> bool:
        if self.components.get("switch/reboot") is not None:
            try:
                resp = requests.post(self.url + "/switch/reboot/turn_on", timeout=5)
                return resp.status_code == 200
            except Exception as e:
                print(f"Failed to send reboot request: {e}")
                return False
        else:
            return False


class CameraMonitorApp(hass.Hass):
    def initialize(self):
        self.monitor = CameraMonitor(
            **self.args, log=self.log, output_dir=self.args["output_dir"]
        )

        self.camera = ESPHomeCameraWrapper(self.args["url"])

        self.log(f"Loaded model with labels: {self.monitor.label_lookup}")

        self.sensor = self.get_entity(self.args["hass_sensor_entity"])
        self.sensor.set_state("off")

        self.trigger_class = self.args["watch_for_class"]

        self.state_machine = State.INIT
        self.state_meta = None
        self.state_transition(State.RUNNING, since=time.time())

        self.run_every(self.poll, interval=self.args["poll_frequency"])

        self._last_cleanup_time = float("-inf")

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
                if frame is not None:
                    foreground_objects = self.monitor.process_frame(
                        frame, timestamp=time.time(), detect_stuff=True
                    )
                    self.handle_detections(foreground_objects)
            except ConnectionError:
                self.state_transition(State.CANT_CONNECT, since=time.time())

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
                    self.log(
                        "Still cannot connect after reboot attempt", level="WARNING"
                    )
                    self.state_transition(State.CRASHED, since=time.time())

        self.maybe_cleanup()

    def handle_detections(self, detected_things: list[ForegroundBlob]):
        if detected_things:
            if any(obj.class_id == self.trigger_class for obj in detected_things):
                if self.sensor.state != "on":
                    self.log(f"DETECTED {self.trigger_class}", level="WARNING")
                    self.sensor.set_state("on")
        else:
            if self.sensor.state != "off":
                self.log(f"DEACTIVATED {self.trigger_class}", level="INFO")
                self.sensor.set_state("off")

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
