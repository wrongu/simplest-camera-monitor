import logging
import os
import threading
import time
from pathlib import Path

import cv2 as cv
import requests
import yaml
from background_model import TimestampAwareBackgroundSubtractor, BoundingBox
from camera_monitor import (
    CameraMonitor,
    State,
    OnDetectionCallback,
    OnStateTransitionCallback,
)
from cameras import ONVIFCameraWrapper
from typing import Callable, Optional
from utils import LogHandler

ONE_DAY_SECONDS = 24 * 60 * 60

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/config/config.yaml"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("camera_monitor")
logger.addHandler(LogHandler(batch_every=30 * 60))


# ---------------------------------------------------------------------------
# HA REST API helpers
# ---------------------------------------------------------------------------


class HomeAssistantClient:
    def __init__(
        self,
        binary_sensors: list[str] = [],
        token: Optional[str] = None,
        api_url: str = "http://supervisor/core/api",
    ):
        if token is None:
            self.token = os.environ.get("SUPERVISOR_TOKEN", None)
        else:
            self.token = token

        if not self.token:
            logger.warning(
                "SUPERVISOR_TOKEN not set — HA state updates will fail. "
                "Ensure homeassistant_api: true in the add-on config.yaml."
            )

        self.api_url = api_url
        self._binary_sensors = {
            name.lower().replace(" ", "_"): {
                "state": "off",
                "attributes": {"friendly_name": name},
            }
            for name in binary_sensors
        }
        self.sync_states()

    def add_binary_sensor(self, name: str) -> None:
        entity_id = name.lower().replace(" ", "_")
        if entity_id in self._binary_sensors:
            return
        self._binary_sensors[entity_id] = {
            "state": "off",
            "attributes": {"friendly_name": name},
        }

    def sync_states(self) -> None:
        if self.token is None:
            return
        try:
            for entity_id, data in self._binary_sensors.items():
                resp = requests.get(
                    f"{self.api_url}/states/binary_sensor.{entity_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=5,
                )
                resp.raise_for_status()
                self._binary_sensors[entity_id]["state"] = data["state"]
        except Exception as e:
            logger.error(f"Failed to sync HA states: {e}")

    def set_state(self, name: str, state: str) -> None:
        entity_id = name.lower().replace(" ", "_")
        if self.token is None:
            return
        if self._binary_sensors.get(entity_id, {}).get("state") == state:
            return
        try:
            resp = requests.post(
                f"{self.api_url}/states/{entity_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"state": state},
                timeout=5,
            )
            resp.raise_for_status()
            self._binary_sensors[entity_id]["state"] = state
        except Exception as e:
            logger.error(f"Failed to set HA state {entity_id}={state}: {e}")


# ---------------------------------------------------------------------------
# AppDaemon self.log replacement
# ---------------------------------------------------------------------------


def make_log(name: str):
    """Return a log callable with the same signature as AppDaemon's self.log."""
    # TODO - just change the CameraMonitor log signature to match logging.Logger and skip this adapter step
    log = logging.getLogger(name)
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def _log(msg: str, level: str = "INFO"):
        log.log(level_map.get(level.upper(), logging.INFO), msg)

    return _log


# ---------------------------------------------------------------------------
# Callbacks (mirrors app.py handle_state_transition / handle_detections)
# ---------------------------------------------------------------------------


def make_handle_state_transition(
    client: HomeAssistantClient, sensor_names: list[str]
) -> OnStateTransitionCallback:
    def handle_state_transition(monitor: CameraMonitor, new_state: State):
        for name in sensor_names:
            friendly_name = f"{monitor.name} {name} detector"
            client.add_binary_sensor(friendly_name)
            if new_state in (State.CANT_CONNECT, State.CRASHED, State.REBOOT):
                client.set_state(friendly_name, "unavailable")
            elif new_state == State.RUNNING:
                client.set_state(friendly_name, "available")

    return handle_state_transition


def make_handle_detections(
    client: HomeAssistantClient, trigger_classes: list[str], log
) -> OnDetectionCallback:
    def handle_detections(monitor: CameraMonitor, detections: list[BoundingBox]):
        detected_classes = {d.class_id for d in detections}
        for name in trigger_classes:
            friendly_name = f"{monitor.name} {name} detector"
            client.add_binary_sensor(friendly_name)
            if name in detected_classes:
                client.set_state(friendly_name, "on")
            else:
                client.set_state(friendly_name, "off")

    return handle_detections


# ---------------------------------------------------------------------------
# Camera initialisation (mirrors app.py initialize())
# ---------------------------------------------------------------------------


def init_monitors(config: dict, client: HomeAssistantClient) -> list[CameraMonitor]:
    trigger_classes = config.get("watch_for_class", [])
    poll_frequency = config["poll_frequency"]
    monitor_slots: list[CameraMonitor | None] = [None] * len(config["cameras"])

    def init_camera(i, cam_config):
        cam_name = cam_config.get("name", str(i))
        media_dir = Path("/media") / cam_name.lower().replace(" ", "_")
        roi_path = media_dir / "roi.png"
        roi = (
            cv.imread(str(roi_path), cv.IMREAD_GRAYSCALE) if roi_path.exists() else None
        )
        log = make_log(f"camera_monitor.{cam_name}")
        try:
            mon = CameraMonitor(
                camera=ONVIFCameraWrapper(
                    cam_config["url"],
                    cam_config["port"],
                    cam_config["username"],
                    cam_config["password"],
                    resolution=tuple(cam_config["resolution"]),
                ),
                name=cam_name,
                brightness_threshold=cam_config["brightness_threshold"],
                history_seconds=cam_config["history_seconds"],
                bg_model=TimestampAwareBackgroundSubtractor(
                    history_seconds=cam_config["history_seconds"],
                    var_threshold=cam_config["var_threshold"],
                    detect_shadows=cam_config["detect_shadows"],
                    area_threshold=cam_config["area_threshold"],
                    shadow_correlation_threshold=cam_config[
                        "shadow_correlation_threshold"
                    ],
                    morph_radius=cam_config["morph_radius"],
                    morph_thresh=cam_config["morph_thresh"],
                    morph_iters=cam_config["morph_iters"],
                    default_fps=1 / poll_frequency,
                    region_of_interest=roi,
                    night_mode_kwargs={
                        k[6:]: v
                        for k, v in cam_config.items()
                        if k.startswith("night_")
                    },
                ),
                save_blobs=cam_config.get("save_blobs", True),
                model_file=cam_config.get("model_file", None),
                output_dir=media_dir,
                log_lifespan=cam_config.get("log_lifespan", ONE_DAY_SECONDS / 2),
                log=log,
                on_state_transition=make_handle_state_transition(
                    client, trigger_classes
                ),
                on_detection=make_handle_detections(
                    client, trigger_classes, make_log(f"camera_monitor.{cam_name}")
                ),
            )
            monitor_slots[i] = mon
        except Exception as e:
            logger.error(f"Failed to initialize camera {cam_name}: {e}")

    threads = [
        threading.Thread(target=init_camera, args=(i, c), daemon=True)
        for i, c in enumerate(config["cameras"])
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    return [m for m in monitor_slots if m is not None]


# ---------------------------------------------------------------------------
# Poll and cleanup loops (replaces self.run_every)
# ---------------------------------------------------------------------------


def poll_loop(monitors: list[CameraMonitor], interval: float) -> None:
    while True:
        for monitor in monitors:
            monitor.poll()
        time.sleep(interval)


def cleanup_loop(monitors: list[CameraMonitor]) -> None:
    while True:
        time.sleep(ONE_DAY_SECONDS / 24)  # Run cleanup every hour
        for monitor in monitors:
            monitor.cleanup_files()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Connect a client
    client = HomeAssistantClient(
        binary_sensors=[
            f"{cam['name']} {c} detector"
            for cam in config["cameras"]
            for c in config.get("watch_for_class", [])
        ]
    )

    monitors = init_monitors(config, client)
    if not monitors:
        logger.error("No cameras initialized. Exiting.")
        return

    # Initial file cleanup
    for monitor in monitors:
        monitor.cleanup_files()

    poll_interval = config["poll_frequency"]
    threading.Thread(
        target=poll_loop, args=(monitors, poll_interval), daemon=True, name="poll"
    ).start()
    threading.Thread(
        target=cleanup_loop, args=(monitors,), daemon=True, name="cleanup"
    ).start()

    logger.info(
        f"Camera monitor running: {len(monitors)} camera(s), "
        f"poll_frequency={poll_interval}s"
    )
    # Block the main thread indefinitely
    threading.Event().wait()


if __name__ == "__main__":
    main()
