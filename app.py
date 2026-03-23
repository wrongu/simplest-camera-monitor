import appdaemon.plugins.hass.hassapi as hass
import cv2 as cv

from background_model import TimestampAwareBackgroundSubtractor
from camera_monitor import CameraMonitor, State
from cameras import ONVIFCameraWrapper

ONE_DAY_SECONDS = 24 * 60 * 60


class CameraMonitorApp(hass.Hass):
    def initialize(self):
        self.monitors: list[CameraMonitor] = []
        for config in self.args["cameras"]:
            mon = CameraMonitor(
                camera=ONVIFCameraWrapper(
                    config["url"],
                    config["port"],
                    config["username"],
                    config["password"],
                    resolution=tuple(config["resolution"]),
                ),
                name=config["name"],
                brightness_threshold=config["brightness_threshold"],
                history_seconds=config["history_seconds"],
                bg_model=TimestampAwareBackgroundSubtractor(
                    history_seconds=config["history_seconds"],
                    var_threshold=config["var_threshold"],
                    detect_shadows=config["detect_shadows"],
                    area_threshold=config["area_threshold"],
                    shadow_correlation_threshold=config["shadow_correlation_threshold"],
                    morph_radius=config["morph_radius"],
                    morph_thresh=config["morph_thresh"],
                    morph_iters=config["morph_iters"],
                    default_fps=1 / config["poll_frequency"],
                    region_of_interest=cv.imread(
                        config["region_of_interest"], cv.IMREAD_GRAYSCALE
                    ),
                    night_mode_kwargs={
                        k[6:]: v for k, v in config.items() if k.startswith("night_")
                    },
                ),
                save_blobs=config.get("save_blobs", True),
                model_file=config.get("model_file", None),
                output_dir=config["output_dir"],
                log=self.log,
                on_state_transition=self.handle_state_transition,
                on_detection=self.handle_detections,
            )

            self.monitors.append(mon)

        self.trigger_classes = self.args.get("watch_for_class", [])

        self.cleanup_files()
        self.run_every(self.poll, interval=self.args["poll_frequency"])
        self.run_every(self.cleanup_files, interval=ONE_DAY_SECONDS / 6)

    def poll(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.poll()

    def cleanup_files(self, *args, **kwargs):
        for monitor in self.monitors:
            monitor.cleanup_files()

    def handle_state_transition(self, monitor: CameraMonitor, new_state: State):
        if new_state in (State.CANT_CONNECT, State.CRASHED, State.REBOOT):
            for cls in self.trigger_classes:
                self.get_entity(f"binary_sensor.{monitor.name}_{cls}").set_state(
                    "unavailable"
                )
        elif new_state == State.RUNNING:
            for cls in self.trigger_classes:
                self.get_entity(f"binary_sensor.{monitor.name}_{cls}").set_state(
                    "available"
                )

    def handle_detections(self, monitor: CameraMonitor, detections: list[str]):
        for cls in self.trigger_classes:
            sensor = self.get_entity(f"binary_sensor.{monitor.name}_{cls}")
            if cls in detections:
                if sensor.state != "on":
                    self.log(f"DETECTED {cls}", level="WARNING")
                    sensor.set_state("on")
            else:
                if sensor.state != "off":
                    self.log(f"DEACTIVATED {cls}", level="INFO")
                    sensor.set_state("off")
