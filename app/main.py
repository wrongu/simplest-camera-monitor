import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from functools import partial
from pathlib import Path
from typing import Optional

from app.image_loader import create_timestamped_filename

try:
    import app
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import requests
import yaml

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

ONE_DAY_SECONDS = 24 * 60 * 60

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
            else:
                logger.warning(f"Poll cycle overran by {-sleep_time:.2f}s")
            next_poll += interval


def cleanup_loop(monitors: list[CameraMonitor]) -> None:
    while True:
        time.sleep(ONE_DAY_SECONDS / 24)  # Run cleanup every hour
        for monitor in monitors:
            monitor.cleanup_files()


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
        frames[mon.name] = frame

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
            cv2.rectangle(
                frames[mon.name],
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frames[mon.name],
                f"[{box.class_id}]",
                (box.x + box.width // 2, box.y + box.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    monitors = init_monitors(
        config,
        on_get_image=handle_frame,
        on_state_transition=handle_state,
        on_detection=handle_detect,
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

    # Connect a client
    client = HomeAssistantClient(
        binary_sensors=[
            f"{cam['name']} {c} detector"
            for cam in config["cameras"]
            for c in config.get("watch_for_class", [])
        ]
    )

    handle_state = make_handle_state_transition(client, config.get("watch_for_class", []))
    handle_detections = make_handle_detections(client, config.get("watch_for_class", []))

    monitors = init_monitors(
        config,
        on_state_transition=handle_state,
        on_detection=[handle_detections, partial(save_detection_snapshot, Path())],
    )
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
    threading.Thread(target=cleanup_loop, args=(monitors,), daemon=True, name="cleanup").start()

    logger.info(
        f"Camera monitor running: {len(monitors)} camera(s), " f"poll_frequency={poll_interval}s"
    )
    # Block the main thread indefinitely
    threading.Event().wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live-debug", action="store_true")
    args = parser.parse_args()

    if args.live_debug:
        run_live()
    else:
        main()
