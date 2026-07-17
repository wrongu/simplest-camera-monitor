import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from functools import partial
from pathlib import Path
from typing import Optional

try:
    import app
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import requests
import yaml


from app.image_loader import create_timestamped_filename
from app.detection import create_detector
from app.utils import BoundingBox, chain_callbacks, get_logger
from app.camera_monitor import (
    CameraMonitor,
    State,
    OnDetectionCallback,
    OnStateTransitionCallback,
    OnGetImageCallback,
)
from app.cameras import ONVIFCameraWrapper
from app.mqtt_client import MqttPublisher

ONE_DAY_SECONDS = 24 * 60 * 60


def slugify(name: str) -> str:
    """Camera name -> entity-id slug, matching the existing HA entity-id convention."""
    return name.lower().replace(" ", "_")


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/config/config.yaml"))

logger = get_logger("app", batching=30)


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
                f"{self.api_url}/states/binary_sensor.{entity_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"state": state},
                timeout=5,
            )
            resp.raise_for_status()
            self._binary_sensors[entity_id]["state"] = state
        except Exception as e:
            logger.error(f"Failed to set HA state {entity_id}={state}: {e}")


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
    client: HomeAssistantClient, trigger_classes: list[str]
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
# MQTT reporting (MQTT Discovery — replaces the REST path above)
# ---------------------------------------------------------------------------


def resolve_mqtt_config(config: dict) -> Optional[dict]:
    """Resolve broker connection settings.

    Order:
      1. Explicit ``mqtt:`` block in the config (needed for dev-machine runs).
      2. Otherwise, if running as an add-on, fetch host/port/credentials from the Supervisor
         Services API (the official Mosquitto add-on).

    ``base_topic`` / ``discovery_prefix`` from an ``mqtt:`` block always win, even when the
    broker itself is auto-discovered.
    """
    mqtt_cfg = dict(config.get("mqtt") or {})
    resolved = {"base_topic": "camera_monitor", "discovery_prefix": "homeassistant"}

    # Explicit block with a host: use it verbatim.
    if mqtt_cfg.get("host"):
        resolved.update(mqtt_cfg)
        return resolved

    # No explicit host — try the Supervisor Services API.
    token = os.environ.get("SUPERVISOR_TOKEN")
    if token:
        try:
            resp = requests.get(
                "http://supervisor/services/mqtt",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            resolved.update(
                {
                    "host": data["host"],
                    "port": data.get("port", 1883),
                    "username": data.get("username"),
                    "password": data.get("password"),
                }
            )
            # Let an mqtt: block still override topic/prefix (but not the broker).
            for key in ("base_topic", "discovery_prefix"):
                if key in mqtt_cfg:
                    resolved[key] = mqtt_cfg[key]
            return resolved
        except Exception as e:
            logger.error(f"Failed to fetch MQTT service from Supervisor: {e}")

    return None


def publish_all_discovery(
    publisher: MqttPublisher, config: dict, watch_for_class: list[str], *, publish_images: bool
) -> None:
    """Publish retained MQTT Discovery configs for every (camera, class): a confidence sensor
    plus an optional per-camera image entity.
    """
    for cam_cfg in config["cameras"]:
        monitor_name = cam_cfg.get("name", "")
        cam = slugify(monitor_name)
        node = f"camera_monitor_{cam}"
        device = {
            "identifiers": [f"camera_monitor_{cam}"],
            "name": monitor_name,
            "manufacturer": "camera-monitor",
        }
        avail = {
            "availability": [
                {"topic": publisher.bridge_availability_topic},
                {"topic": publisher.availability_topic(cam)},
            ],
            "availability_mode": "all",
        }

        for cls in watch_for_class:
            publisher.publish_discovery(
                "sensor",
                node,
                f"{cls}_confidence",
                {
                    "name": f"{monitor_name} {cls} confidence",
                    "unique_id": f"camera_monitor_{cam}_{cls}_confidence",
                    "state_topic": publisher.state_topic(cam),
                    "value_template": (
                        f"{{{{ (value_json.classes['{cls}'].conf * 100) | round(0) }}}}"
                    ),
                    "unit_of_measurement": "%",
                    "state_class": "measurement",
                    "device": device,
                    **avail,
                },
            )

        # One "last detection" image entity per camera.
        if publish_images:
            publisher.publish_discovery(
                "image",
                node,
                "last_detection",
                {
                    "name": f"{monitor_name} last detection",
                    "unique_id": f"camera_monitor_{cam}_last_detection",
                    "image_topic": publisher.image_topic(cam),
                    "image_encoding": "b64",
                    "device": device,
                    **avail,
                },
            )


def make_handle_state_transition_mqtt(publisher: MqttPublisher) -> OnStateTransitionCallback:
    def handle_state_transition(monitor: CameraMonitor, new_state: State):
        cam = slugify(monitor.name)
        if new_state == State.RUNNING:
            publisher.publish_availability(cam, "online")
        elif new_state in (State.CANT_CONNECT, State.CRASHED, State.REBOOT):
            publisher.publish_availability(cam, "offline")

    return handle_state_transition


def make_handle_detections_mqtt(
    publisher: MqttPublisher, watch_for_class: list[str]
) -> OnDetectionCallback:
    """Build per-frame state JSON (max conf + count per watched class) and publish it. Every
    frame is published — including empty ones — so a class that stops being seen drops to 0."""

    def handle_detections(monitor: CameraMonitor, detections: list[BoundingBox]):
        classes = {cls: {"conf": 0.0, "count": 0} for cls in watch_for_class}
        for det in detections:
            entry = classes.get(det.class_id)
            if entry is not None:
                entry["count"] += 1
                entry["conf"] = max(entry["conf"], float(det.confidence))
        payload = {"ts": monitor.last_timestamp, "classes": classes}
        publisher.publish_state(slugify(monitor.name), payload)

    return handle_detections


def make_handle_image_mqtt(
    publisher: MqttPublisher, min_interval: float = 1.0, max_size: Optional[int] = None
) -> OnDetectionCallback:
    """Publish an annotated, downscaled JPEG of the latest frame when a watched class is
    detected. Rate-limited to at most one image per ``min_interval`` seconds per camera."""
    last_pub: dict[str, float] = {}

    def handle_image(monitor: CameraMonitor, detections: list[BoundingBox]):
        cam = slugify(monitor.name)
        now = time.monotonic()
        if now - last_pub.get(cam, 0.0) < min_interval:
            return
        _, frame = monitor.camera.get_last_frame()
        if frame is None:
            return
        annotated = frame.copy()
        for det in detections:
            annotated = det.draw(annotated, color=(255, 150, 0))
        h, w = annotated.shape[:2]
        if max_size is not None and w > max_size:
            scale = max_size / w
            annotated = cv2.resize(
                annotated, (max_size, int(round(h * scale))), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            return
        publisher.publish_image(cam, buf.tobytes())
        last_pub[cam] = now

    return handle_image


def save_detection_snapshot(save_dir: Path, monitor: CameraMonitor, detections: list[BoundingBox]):
    if detections:
        ts, frame = monitor.camera.get_last_frame()
        if frame is not None:
            save_file = save_dir / create_timestamped_filename(ts, ".jpg")
            save_file.parent.mkdir(exist_ok=True, parents=True)
            annotated_frame = frame.copy()
            for det in detections:
                annotated_frame = det.draw(annotated_frame, color=(255, 150, 0))
            cv2.imwrite(str(save_file.resolve()), annotated_frame)


# ---------------------------------------------------------------------------
# Camera initialisation (mirrors app.py initialize())
# ---------------------------------------------------------------------------


def init_monitors(
    config: dict,
    *,
    on_get_image: Optional[OnGetImageCallback] = None,
    on_state_transition: Optional[OnStateTransitionCallback] = None,
    on_detection: Optional[OnDetectionCallback] = None,
) -> list[CameraMonitor]:
    monitor_slots: list[CameraMonitor | None] = [None] * len(config["cameras"])
    media_root = Path(config.get("media_root", "/media"))

    if config.get("save_detections", False):
        snapshot_callback = partial(save_detection_snapshot, media_root / "detections")
        if on_detection is not None:
            on_detection = chain_callbacks(on_detection, snapshot_callback)
        else:
            on_detection = snapshot_callback

    def init_camera(i, cam_config):
        cam_name = cam_config.get("name", str(i))
        media_dir = media_root / cam_name.lower().replace(" ", "_")

        if cam_config.get("detector_config", None) is not None:
            detector = create_detector(cam_config["detector_config"])
        else:
            detector = None

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
                detection_model=detector,
                output_dir=media_dir,
                log_lifespan=cam_config.get("log_lifespan", ONE_DAY_SECONDS / 2),
                on_get_image=on_get_image,
                on_state_transition=on_state_transition,
                on_detection=on_detection,
                confidence_threshold=cam_config.get("confidence_threshold", 0.5),
                roi=cam_config.get("roi"),
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
    with ThreadPoolExecutor(max_workers=len(monitors), thread_name_prefix="poll") as executor:
        next_poll = time.monotonic() + interval
        while True:
            futures = [executor.submit(m.poll) for m in monitors]
            futures_wait(futures, timeout=interval)
            for f in futures:
                if f.done() and not f.cancelled():
                    exc = f.exception()
                    if exc is not None:
                        logger.error(f"Unexpected error in poll: {exc}")
            sleep_time = next_poll - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1 / 30:
                logger.warning(f"Poll cycle overran by {-sleep_time:.2f}s")
            next_poll += interval


def cleanup_loop(monitors: list[CameraMonitor]) -> None:
    while True:
        for monitor in monitors:
            monitor.cleanup_files()
        time.sleep(ONE_DAY_SECONDS / 24)  # Run cleanup every hour


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_live():
    logger.info(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    cv2.namedWindow("Live Monitor", cv2.WINDOW_NORMAL)

    frames = defaultdict(lambda: np.zeros((360, 640, 3), dtype=np.uint8))

    def redraw():
        try:
            combo = np.concatenate(list(frames.values()), axis=1)
            cv2.imshow("Live Monitor", combo)
        except ValueError as e:
            print(e)

    def handle_frame(mon: CameraMonitor, frame: np.ndarray, timestamp: float):
        frames[mon.name] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def handle_state(mon: CameraMonitor, st: State):
        cv2.putText(
            frames[mon.name],
            f"STATE: {st}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    def handle_detect(mon: CameraMonitor, boxes: list[BoundingBox]):
        for box in boxes:
            box.draw(
                frames[mon.name],
                color=(
                    int(127 * (1 - box.confidence)),
                    int(255 * box.confidence),
                    int(127 * (1 - box.confidence)),
                ),
            )

    state_callbacks = [handle_state]
    detection_callbacks = [handle_detect]
    frame_callbacks = [handle_frame]

    if config.get("reporting", "none") == "mqtt":
        watch_for_class = ["person"]
        logger.info("Reporting via MQTT Discovery")
        mqtt_cfg = resolve_mqtt_config(config)
        if mqtt_cfg is None:
            logger.error(
                "reporting: mqtt selected but no broker could be resolved. Add an mqtt: block "
                "to the config, or run as an add-on with `services: [mqtt:need]`."
            )
            return
        publisher = MqttPublisher(**mqtt_cfg)
        publisher.connect()

        publish_images = config.get("mqtt_publish_images", False)
        # Discovery must be published before any state/availability so HA has the entities.
        publish_all_discovery(publisher, config, watch_for_class, publish_images=publish_images)
        state_callbacks.append(make_handle_state_transition_mqtt(publisher))
        detection_callbacks.append(make_handle_detections_mqtt(publisher, watch_for_class))
        if publish_images:
            detection_callbacks.append(make_handle_image_mqtt(publisher))

    monitors = init_monitors(
        config,
        on_get_image=chain_callbacks(*frame_callbacks),
        on_state_transition=chain_callbacks(*state_callbacks),
        on_detection=chain_callbacks(*detection_callbacks),
    )
    if not monitors:
        logger.error("No cameras initialized. Exiting.")
        return

    # Initial file cleanup
    for monitor in monitors:
        monitor.cleanup_files()

    def poll():
        for mon in monitors:
            mon.poll()
        redraw()

    poll()
    last_poll = time.time()
    while True:
        if time.time() - last_poll > config["poll_frequency"]:
            poll()
            last_poll = time.time()

        k = cv2.waitKey(1)
        if k == ord("q"):
            break


def main():
    logger.info(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    watch_for_class = config.get("watch_for_class", [])
    reporting = config.get("reporting", "mqtt")

    detection_callbacks: list[OnDetectionCallback] = []
    state_callbacks: list[OnStateTransitionCallback] = []
    publisher: Optional[MqttPublisher] = None

    if reporting == "rest":
        # Legacy path — Supervisor REST state-setting. TODO - remove/deprecate.
        logger.info("Reporting via HA Supervisor REST API")
        client = HomeAssistantClient(
            binary_sensors=[
                f"{cam['name']} {c} detector" for cam in config["cameras"] for c in watch_for_class
            ]
        )
        state_callbacks.append(make_handle_state_transition(client, watch_for_class))
        detection_callbacks.append(make_handle_detections(client, watch_for_class))
    elif reporting == "mqtt":
        logger.info("Reporting via MQTT Discovery")
        mqtt_cfg = resolve_mqtt_config(config)
        if mqtt_cfg is None:
            logger.error(
                "reporting: mqtt selected but no broker could be resolved. Add an mqtt: block "
                "to the config, or run as an add-on with `services: [mqtt:need]`."
            )
            return
        publisher = MqttPublisher(**mqtt_cfg)
        publisher.connect()

        publish_images = config.get("mqtt_publish_images", False)
        # Discovery must be published before any state/availability so HA has the entities.
        publish_all_discovery(publisher, config, watch_for_class, publish_images=publish_images)
        state_callbacks.append(make_handle_state_transition_mqtt(publisher))
        detection_callbacks.append(make_handle_detections_mqtt(publisher, watch_for_class))
        if publish_images:
            detection_callbacks.append(make_handle_image_mqtt(publisher))
    elif reporting == "none":
        pass
    else:
        logger.error(f"Unknown reporting mode: {reporting!r} (expected 'mqtt', 'rest', or 'none')")
        return

    on_state_transition = chain_callbacks(*state_callbacks) if state_callbacks else None
    on_detection = chain_callbacks(*detection_callbacks) if detection_callbacks else None

    monitors = init_monitors(
        config,
        on_state_transition=on_state_transition,
        on_detection=on_detection,
    )
    if not monitors:
        logger.error("No cameras initialized. Exiting.")
        if publisher is not None:
            publisher.disconnect()
        return

    poll_interval = config["poll_frequency"]
    threading.Thread(
        target=poll_loop, args=(monitors, poll_interval), daemon=True, name="poll"
    ).start()
    threading.Thread(target=cleanup_loop, args=(monitors,), daemon=True, name="cleanup").start()

    logger.info(
        f"Camera monitor running: {len(monitors)} camera(s), " f"poll_frequency={poll_interval}s"
    )
    # Block the main thread indefinitely
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        if publisher is not None:
            publisher.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live-debug", action="store_true")
    args = parser.parse_args()

    if args.live_debug:
        run_live()
    else:
        main()
