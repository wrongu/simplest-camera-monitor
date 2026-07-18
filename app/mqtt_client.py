"""MQTT publisher for Home Assistant MQTT Discovery.

This module provides an MQTT publisher that follows the Frigate pattern: the add-on keeps doing
all the CV work in its container and publishes results to MQTT; HA's built-in MQTT integration
turns them into entities via *MQTT Discovery*.

A note on availability / Last-Will: A single MQTT connection can only register **one** Last Will
& Testament. To mark *every* camera's entities unavailable on a hard add-on crash, we set the LWT
on a shared *bridge* availability topic (``<base_topic>/availability``) and give each discovery
entity two availability sources with ``availability_mode: all``:
    1. the bridge topic (offline via LWT if the whole add-on dies), and
    2. the per-camera topic ( ``<base_topic>/<cam>/availability``, driven by the state machine).
An entity is only "available" when both say ``online``.
"""

import base64
import json
import os
import threading
import time
from typing import Optional

import cv2
import paho.mqtt.client as mqtt
import requests

from app.camera_monitor import OnStateTransitionCallback, OnDetectionCallback, CameraMonitor, State
from app.utils import get_logger, slugify, BoundingBox

logger = get_logger("mqtt_client", batching=300)

# paho-mqtt v2 introduced an explicit callback-API version. We target v2 (see requirements.txt).
from paho.mqtt.enums import CallbackAPIVersion


class MqttPublisher(object):
    """Small MQTT publisher: connect, background loop, reconnect-with-backoff, LWT, discovery.

    Retained discovery configs are remembered and re-published on every (re)connect so HA
    re-creates entities after a broker restart.
    """

    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_topic: str = "camera_monitor",
        discovery_prefix: str = "homeassistant",
        client_id: str = "camera_monitor",
    ):
        self.host = host
        self.port = int(port)
        self.base_topic = base_topic.rstrip("/")
        self.discovery_prefix = discovery_prefix.rstrip("/")
        self.bridge_availability_topic = f"{self.base_topic}/availability"

        self._client = mqtt.Client(
            CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True
        )
        if username:
            self._client.username_pw_set(username, password)
        self._client.reconnect_delay_set(min_delay=1, max_delay=120)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Bridge-level LWT: if this process dies, HA marks every entity unavailable.
        self._client.will_set(self.bridge_availability_topic, "offline", qos=1, retain=True)

        # Retained messages we must re-send on every (re)connect: discovery configs keyed by
        # topic, and the most recent per-camera availability so a reconnect restores it.
        self._lock = threading.Lock()
        self._discovery: dict[str, str] = {}
        self._availability: dict[str, str] = {}
        self.connected = threading.Event()

    # -- lifecycle ----------------------------------------------------------------------------

    def connect(self) -> None:
        logger.info(f"Connecting to MQTT broker {self.host}:{self.port}")
        self._client.connect_async(self.host, self.port, keepalive=60)
        self._client.loop_start()

    def disconnect(self) -> None:
        """Graceful shutdown: mark the bridge offline, then stop the network loop."""
        try:
            self._client.publish(self.bridge_availability_topic, "offline", qos=1, retain=True)
        finally:
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            logger.error(f"MQTT connection failed: {reason_code}")
            return
        logger.info("MQTT connected")
        self.connected.set()
        # Bridge comes online, then re-assert retained discovery + per-camera availability.
        client.publish(self.bridge_availability_topic, "online", qos=1, retain=True)
        with self._lock:
            for topic, payload in self._discovery.items():
                client.publish(topic, payload, qos=1, retain=True)
            for cam, status in self._availability.items():
                client.publish(f"{self.base_topic}/{cam}/availability", status, qos=1, retain=True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        self.connected.clear()
        logger.warning(f"MQTT disconnected: {reason_code} (will auto-reconnect)")

    # -- topic helpers ------------------------------------------------------------------------

    def state_topic(self, cam: str) -> str:
        return f"{self.base_topic}/{cam}/state"

    def availability_topic(self, cam: str) -> str:
        return f"{self.base_topic}/{cam}/availability"

    def image_topic(self, cam: str) -> str:
        return f"{self.base_topic}/{cam}/image"

    # -- publishing ---------------------------------------------------------------------------

    def publish_discovery(self, component: str, node: str, object_id: str, payload: dict) -> None:
        """Publish a retained MQTT Discovery config to
        ``<discovery_prefix>/<component>/<node>/<object_id>/config`` and remember it for
        re-publishing on reconnect."""
        topic = f"{self.discovery_prefix}/{component}/{node}/{object_id}/config"
        data = json.dumps(payload)
        with self._lock:
            self._discovery[topic] = data
        self._client.publish(topic, data, qos=1, retain=True)

    def publish_state(self, cam: str, payload: dict) -> None:
        """Publish per-frame state JSON (not retained, QoS 0)."""
        self._client.publish(self.state_topic(cam), json.dumps(payload), qos=0, retain=False)

    def publish_availability(self, cam: str, status: str) -> None:
        """Publish per-camera availability (``online``/``offline``), retained, QoS 1."""
        with self._lock:
            self._availability[cam] = status
        self._client.publish(self.availability_topic(cam), status, qos=1, retain=True)

    def publish_image(self, cam: str, jpeg_bytes: bytes, encoding: str = "b64") -> None:
        """Publish an annotated JPEG, retained. ``encoding`` matches the discovery
        ``image_encoding`` field ('b64' or raw bytes)."""
        payload = base64.b64encode(jpeg_bytes) if encoding == "b64" else jpeg_bytes
        self._client.publish(self.image_topic(cam), payload, qos=0, retain=True)


def resolve_mqtt_config(main_config: dict) -> Optional[dict]:
    """Resolve broker connection settings.

    Order:
      1. Explicit ``mqtt:`` block in the config (needed for dev-machine runs).
      2. Otherwise, if running as an add-on, fetch host/port/credentials from the Supervisor
         Services API (the official Mosquitto add-on).

    ``base_topic`` / ``discovery_prefix`` from an ``mqtt:`` block always win, even when the
    broker itself is auto-discovered.
    """
    mqtt_cfg = dict(main_config.get("mqtt") or {})
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
        annotated = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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


def make_mqtt_callbacks(
    publisher: MqttPublisher,
    watch_for_class: list[str],
    publish_images: bool = False,
    min_interval: float = 1.0,
    max_size: Optional[int] = None,
) -> tuple[list[OnStateTransitionCallback], list[OnDetectionCallback]]:
    on_state = [make_handle_state_transition_mqtt(publisher)]
    on_detections = [make_handle_detections_mqtt(publisher, watch_for_class)]
    if publish_images:
        on_detections.append(make_handle_image_mqtt(publisher, min_interval, max_size))

    return on_state, on_detections
