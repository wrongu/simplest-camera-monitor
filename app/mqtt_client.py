"""MQTT publisher for Home Assistant MQTT Discovery.

This module replaces the Supervisor REST state-setting (``HomeAssistantClient`` in ``app/main.py``) with an MQTT
publisher that follows the Frigate pattern: the add-on keeps doing all the CV work in its container and publishes
results to MQTT; HA's built-in MQTT integration turns them into entities via *MQTT Discovery*.

A note on availability / Last-Will: A single MQTT connection can only register **one** Last Will & Testament. To mark
*every* camera's entities unavailable on a hard add-on crash, we set the LWT on a shared *bridge* availability topic
(``<base_topic>/availability``) and give each discovery entity two availability sources with ``availability_mode:
all``:
    1. the bridge topic (offline via LWT if the whole add-on dies), and
    2. the per-camera topic ( ``<base_topic>/<cam>/availability``, driven by the state machine).
An entity is only "available" when both say ``online``.
"""

import base64
import json
import threading
from typing import Optional

import paho.mqtt.client as mqtt

from app.utils import get_logger

logger = get_logger("mqtt_client", batching=300)

# paho-mqtt v2 introduced an explicit callback-API version. We target v2 (see requirements.txt).
from paho.mqtt.enums import CallbackAPIVersion


class MqttPublisher:
    """Small MQTT publisher: connect, background loop, reconnect-with-backoff, LWT, discovery.

    Retained discovery configs are remembered and re-published on every (re)connect so HA re-creates entities after a
    broker restart.
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
                client.publish(
                    f"{self.base_topic}/{cam}/availability", status, qos=1, retain=True
                )

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
