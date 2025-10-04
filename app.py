import time
from enum import Enum

import appdaemon.plugins.hass.hassapi as hass

from background_model import ForegroundBlob, TimestampAwareBackgroundSubtractor
from camera_monitor import CameraMonitor
from cameras import ESPHomeCameraWrapper, ONVIFCameraWrapper
from image_loader import ensure_files_timestamp_named, get_all_timestamped_files_sorted

ONE_DAY_SECONDS = 24 * 60 * 60


class State(Enum):
    INIT = -1
    RUNNING = 0
    CANT_CONNECT = 1
    CRASHED = 2
    REBOOT = 3


class CameraMonitorApp(hass.Hass):
    def initialize(self):
        self.save_blobs = self.args.get("save_blobs", True)

        self.monitor = CameraMonitor(
            brightness_threshold=self.args["brightness_threshold"],
            history_seconds=self.args["history_seconds"],
            bg_model=TimestampAwareBackgroundSubtractor(
                history_seconds=self.args["history_seconds"],
                var_threshold=self.args["var_threshold"],
                detect_shadows=self.args["detect_shadows"],
                area_threshold=self.args["area_threshold"],
                shadow_correlation_threshold=self.args["shadow_correlation_threshold"],
                kernel_cleanup=self.args["kernel_cleanup"],
                kernel_close=self.args["kernel_close"],
                default_fps=1 / self.args["poll_frequency"],
            ),
            model_file=self.args["model_file"],
            output_dir=self.args["output_dir"],
            log=self.log,
        )

        self.camera = ONVIFCameraWrapper(
            self.args["url"],
            self.args["port"],
            self.args["username"],
            self.args["password"],
            resolution=tuple(self.args["resolution"]),
        )

        self.log(f"Loaded model with labels: {self.monitor.label_lookup}")

        self.sensor = self.get_entity(self.args["hass_sensor_entity"])
        self.sensor.set_state("off")

        self.trigger_class = self.args["watch_for_class"]

        self.state_machine = State.INIT
        self.state_meta = None
        self.state_transition(State.RUNNING, since=time.time())

        self.cleanup_files()
        self.run_every(self.poll, interval=self.args["poll_frequency"])
        self.run_every(self.cleanup_files, interval=ONE_DAY_SECONDS / 6)

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
                        frame,
                        timestamp=time.time(),
                        detect_stuff=True,
                        save_blobs=self.save_blobs,
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
                    self.monitor.process_frame(frame)
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
                    self.monitor.process_frame(frame)
                except ConnectionError:
                    self.log(
                        "Still cannot connect after reboot attempt", level="WARNING"
                    )
                    self.state_transition(State.CRASHED, since=time.time())

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

    def cleanup_files(self):
        self.log("Starting cleanup")
        now = time.time()

        # Check that all logged filenames are appropriately timestamped
        ensure_files_timestamp_named(
            self.monitor.output_dir, dry_run=False, glob="**/*.jpg"
        )

        # Delete images that are older than 24h and detected blobs that are older than 72h
        n_images_deleted = 0
        for t, f in get_all_timestamped_files_sorted(
            self.monitor.output_dir, glob="20*/**/*.jpg"
        ):
            if now - t > ONE_DAY_SECONDS:
                f.unlink()
                n_images_deleted += 1
        self.log(f"Deleted {n_images_deleted} old images")

        n_blobs_deleted = 0
        for t, f in get_all_timestamped_files_sorted(
            self.monitor.output_dir, glob="blobs/**/*.jpg"
        ):
            if now - t > 3 * ONE_DAY_SECONDS:
                f.unlink()
                n_blobs_deleted += 1
        self.log(f"Deleted {n_blobs_deleted} old blobs files")
