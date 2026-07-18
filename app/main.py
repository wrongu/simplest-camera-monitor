import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

try:
    import app
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml


from app.detection import create_detector
from app.utils import BoundingBox, chain_callbacks, get_logger, slugify
from app.camera_monitor import (
    CameraMonitor,
    State,
    OnDetectionCallback,
    OnStateTransitionCallback,
    OnGetImageCallback,
)
from app.cameras import ONVIFCameraWrapper
from app.mqtt_client import MqttPublisher, resolve_mqtt_config, make_mqtt_callbacks, \
    publish_all_discovery

ONE_DAY_SECONDS = 24 * 60 * 60

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "/config/config.yaml"))

logger = get_logger("app", batching=30)


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

    def init_camera(i, cam_config):
        cam_name = cam_config.get("name", str(i))
        output_dir = media_root / slugify(cam_name)

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
                output_dir=output_dir,
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
    in_flight: set[CameraMonitor] = set()
    in_flight_lock = threading.Lock()

    def poll_once(monitor: CameraMonitor) -> None:
        try:
            monitor.poll()
        except Exception as e:
            logger.error(f"Unexpected error in poll: {e}")
        finally:
            with in_flight_lock:
                in_flight.discard(monitor)

    with ThreadPoolExecutor(max_workers=len(monitors), thread_name_prefix="poll") as executor:
        next_poll = time.monotonic() + interval
        while True:
            for m in monitors:
                with in_flight_lock:
                    if m in in_flight:
                        logger.warning(f"Skipping poll for {m}: previous poll still running")
                        continue
                    in_flight.add(m)
                executor.submit(poll_once, m)

            sleep_time = next_poll - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1 / 30:
                logger.warning(f"Poll cycle overran by {-sleep_time:.2f}s")
            next_poll += interval


def cleanup_loop(monitors: list[CameraMonitor]) -> None:
    while True:
        for monitor in monitors:
            try:
                monitor.cleanup_files()
            except Exception as e:
                logger.error(f"Unexpected error in cleanup: {e}")
        time.sleep(ONE_DAY_SECONDS / 24)  # Run cleanup every hour


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _init_live_callbacks():
    cv2.namedWindow("Live Monitor", cv2.WINDOW_NORMAL)
    live_display_frames = defaultdict(lambda: np.zeros((360, 640, 3), dtype=np.uint8))

    def redraw():
        if not live_display_frames:
            return
        combo = np.concatenate(list(live_display_frames.values()), axis=1)
        cv2.imshow("Live Monitor", combo)

    def live_get_image_update(mon: CameraMonitor, frame: np.ndarray, timestamp: float):
        live_display_frames[mon.name] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def live_state_update(mon: CameraMonitor, st: State):
        if mon.name not in live_display_frames:
            return
        cv2.putText(
            live_display_frames[mon.name],
            f"STATE: {st}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    def live_detection_update(mon: CameraMonitor, boxes: list[BoundingBox]):
        if mon.name not in live_display_frames:
            return
        for box in boxes:
            box.draw(
                live_display_frames[mon.name],
                color=(
                    int(127 * (1 - box.confidence)),
                    int(255 * box.confidence),
                    int(127 * (1 - box.confidence)),
                ),
            )

    return redraw, live_get_image_update, live_state_update, live_detection_update


def main(live: bool = False):
    logger.info(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    state_callbacks: list[OnStateTransitionCallback] = []
    detection_callbacks: list[OnDetectionCallback] = []
    get_image_callbacks: list[OnGetImageCallback] = []

    if live:
        redraw, live_get_image_update, live_state_update, live_detection_update = (
            _init_live_callbacks()
        )

        state_callbacks.append(live_state_update)
        detection_callbacks.append(live_detection_update)
        get_image_callbacks.append(live_get_image_update)

    watch_for_class = config.get("watch_for_class", [])
    reporting = config.get("reporting", "none")

    publisher: Optional[MqttPublisher] = None

    if reporting == "mqtt":
        logger.info("Reporting via MQTT Discovery")
        mqtt_cfg = resolve_mqtt_config(config)
        if mqtt_cfg is None:
            logger.error(
                "reporting: mqtt selected but no broker could be resolved. Add an mqtt: block "
                "to the config, or run as an add-on with `services: [mqtt:need]`."
            )
            return
        publish_images = mqtt_cfg.pop("publish_images", False)
        publisher = MqttPublisher(**mqtt_cfg)
        publisher.connect()
        publish_all_discovery(publisher, config, watch_for_class, publish_images=publish_images)

        logger.info(f"MQTT Configured; publish_images={publish_images}")

        mqtt_on_state, mqtt_on_detections = make_mqtt_callbacks(
            publisher, watch_for_class, publish_images=publish_images
        )
        state_callbacks.extend(mqtt_on_state)
        detection_callbacks.extend(mqtt_on_detections)

    elif reporting == "none":
        logger.info("Reporting disabled")
        pass
    else:
        logger.error(f"Unknown reporting mode: {reporting!r} (expected 'mqtt' or 'none')")
        return

    on_state_transition = chain_callbacks(*state_callbacks) if state_callbacks else None
    on_detection = chain_callbacks(*detection_callbacks) if detection_callbacks else None
    on_get_image = chain_callbacks(*get_image_callbacks) if get_image_callbacks else None

    monitors = init_monitors(
        config,
        on_state_transition=on_state_transition,
        on_detection=on_detection,
        on_get_image=on_get_image,
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
        if live:
            while True:
                redraw()
                k = cv2.waitKey(100)
                if k == ord("q"):
                    raise KeyboardInterrupt()
        else:
            threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        if publisher is not None:
            publisher.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--live-debug", action="store_true")
    args = parser.parse_args()

    main(args.live_debug)
