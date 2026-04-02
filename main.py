import logging
import os
import threading
import time
from pathlib import Path

import cv2 as cv
import requests
import yaml
from background_model import TimestampAwareBackgroundSubtractor
from camera_monitor import CameraMonitor, State
from cameras import ONVIFCameraWrapper

ONE_DAY_SECONDS = 24 * 60 * 60

CONFIG_PATH = Path("/config/camera_monitor.yaml")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("camera_monitor")


# ---------------------------------------------------------------------------
# HA REST API helpers
# ---------------------------------------------------------------------------

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
HA_API = "http://supervisor/core/api"

# Local HA entity state cache for deduplication
_sensor_states: dict[str, str] = {}


def set_ha_state(entity_id: str, state: str) -> None:
    if _sensor_states.get(entity_id) == state:
        return
    try:
        resp = requests.post(
            f"{HA_API}/states/{entity_id}",
            headers={"Authorization": f"Bearer {SUPERVISOR_TOKEN}"},
            json={"state": state},
            timeout=5,
        )
        resp.raise_for_status()
        _sensor_states[entity_id] = state
    except Exception as e:
        logger.error(f"Failed to set HA state {entity_id}={state}: {e}")


def get_ha_state(entity_id: str) -> str | None:
    return _sensor_states.get(entity_id)


# ---------------------------------------------------------------------------
# AppDaemon self.log replacement
# ---------------------------------------------------------------------------


def make_log(name: str):
    """Return a log callable with the same signature as AppDaemon's self.log."""
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


def make_handle_state_transition(trigger_classes: list[str]):
    def handle_state_transition(monitor: CameraMonitor, new_state: State):
        if new_state in (State.CANT_CONNECT, State.CRASHED, State.REBOOT):
            for cls in trigger_classes:
                set_ha_state(f"binary_sensor.{monitor.name}_{cls}", "unavailable")
        elif new_state == State.RUNNING:
            for cls in trigger_classes:
                set_ha_state(f"binary_sensor.{monitor.name}_{cls}", "available")

    return handle_state_transition


def make_handle_detections(trigger_classes: list[str], log):
    def handle_detections(monitor: CameraMonitor, detections: list[str]):
        for cls in trigger_classes:
            entity_id = f"binary_sensor.{monitor.name}_{cls}"
            if cls in detections:
                if get_ha_state(entity_id) != "on":
                    log(f"DETECTED {cls}", level="WARNING")
                    set_ha_state(entity_id, "on")
            else:
                if get_ha_state(entity_id) != "off":
                    log(f"DEACTIVATED {cls}", level="INFO")
                    set_ha_state(entity_id, "off")

    return handle_detections


# ---------------------------------------------------------------------------
# Camera initialisation (mirrors app.py initialize())
# ---------------------------------------------------------------------------


def init_monitors(config: dict) -> list[CameraMonitor]:
    trigger_classes = config.get("watch_for_class", [])
    poll_frequency = config["poll_frequency"]
    monitor_slots: list[CameraMonitor | None] = [None] * len(config["cameras"])

    def init_camera(i, cam_config):
        cam_name = cam_config.get("name", str(i))
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
                    region_of_interest=cv.imread(
                        cam_config["region_of_interest"], cv.IMREAD_GRAYSCALE
                    ),
                    night_mode_kwargs={
                        k[6:]: v
                        for k, v in cam_config.items()
                        if k.startswith("night_")
                    },
                ),
                save_blobs=cam_config.get("save_blobs", True),
                model_file=cam_config.get("model_file", None),
                output_dir=cam_config["output_dir"],
                log_lifespan=cam_config.get("log_lifespan", ONE_DAY_SECONDS / 2),
                log=log,
                on_state_transition=make_handle_state_transition(trigger_classes),
                on_detection=make_handle_detections(
                    trigger_classes, make_log(f"camera_monitor.{cam_name}")
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
        time.sleep(ONE_DAY_SECONDS / 6)
        for monitor in monitors:
            monitor.cleanup_files()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if not SUPERVISOR_TOKEN:
        logger.warning(
            "SUPERVISOR_TOKEN not set — HA state updates will fail. "
            "Ensure homeassistant_api: true in the add-on config.yaml."
        )

    monitors = init_monitors(config)
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
