# CLAUDE.md — Agent Context for camera-monitor

## Project history

For main app logic and homeassistant (HA) integration:

- Version 1 used AppDaemon. This was an unnecessary layer and created extra complexity.
- Version 2 was a standalone homeassistant addon/app that talked directly to the HA supervisor REST API. This made for limited integration capabilities with HA.
- Version 3 (current) still uses a standalone addon/app, but communication back to HA is via MQTT.

For machine learning and computer vision:

- Version 1 used an ESPHome camera, opencv, and scikit-learn. Emphasis was on lightweight CV: background subtraction and feature-based classification.
  Major problems were lack of light at night and poor precision and recall. Poor performance may have been due primarily to sensor quality.
- Current ML pipeline is ultralytics/YOLO training on a dev machine (`requirements-dev.txt`), exporting to ONNX so 
  that edge deployment only needs onnx and not the entire yolo/pytorch suite of libraries.
- Current camera hardware is two Tapo C113 RSTP cameras. They switch to infrared night mode automatically.
  User also installed an infrared floodlight to improve nighttime image quality. Because imaging hardware
  improved slowly over time, older training data is less relevant.
- app/annotator.py is a standalone application unrelated to app/main.py; the user downloads and annotates and trains offline on a dev machine.

## Component map

| File                            | Role                                                                                                                |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `app/main.py`                   | HA App entry point — reads config, spins up monitors, posts to HA API                                               |
| `app/camera_monitor.py`         | Core state machine — polls frames, runs bg subtraction, fires callbacks                                             |
| `app/cameras.py`                | Camera backends: ONVIF, ESPHome HTTP, offline image replay                                                          |
| `app/detection.py`              | Defines a detector protocol and three implementatoins: a CV+scikit-learn feature-based detector, Yolo, and OnnxYolo |
| `app/background_model.py`       | MOG2 background subtraction, blob extraction, shadow filtering. No longer used in ML but used in annotation.        |
| `app/image_loader.py`           | Timestamped image file naming, sorting, migration, thinning                                                         |
| `app/annotator.py`              | Interactive GUI for labeling blobs (offline, dev only)                                                              |
| `app/classifier.py`             | Train decision-tree classifier on annotated blobs (offline, dev only)                                               |
| `app/utils.py`                  | Various data structures and shared processing utilities.                                                            |
| `app/mqtt_client.py`            | Defines a generic MqttPublisher class as well as all app-specific MQTT data structures and callbacks                |
| `models/train_yolo.py`          | Train a YOLO classifier and export to ONNX (offline, dev only)                                                      |
| `models/train_cfg.yaml`         | Configuring the training runs.                                                                                      |
| `models/export_yolo.py`         | Export from our custom annotation format to yolo format, once per camera/directory.                                 |
| `models/merge_yolo_datasets.py` | Merge yolo datasets across cameras/directories to train a shared model.                                             |
| `config.yaml`                   | HA Add-on manifest (name, slug, arch, volume maps)                                                                  |
| `example_config.yaml`           | An example config that a user would put in `/addon_configs/local_camera_monitor/config.yaml` on their HA instance   |
| `Dockerfile`                    | Alpine + OpenCV-from-source build                                                                                   |

## Key conventions

- Image files use the path format `YYYY/MM/DD/HHMMSS.jpg` (new format). The old flat `YYYYMMDD_HHMMSS.jpg` format is still supported for reading. See `image_loader.py`.
- Camera state transitions and detections go through callbacks (`on_state_transition`, `on_detection`). MQTT sets up callbacks.
- All callback invocations in `camera_monitor.py` are wrapped in `try/except` — a bad callback must not crash the monitor.
- Camera initialization, polling, and file cleanup all happen in threads.
- The cleanup loop in `main.py` runs every 4 hours and deletes images older than `log_lifespan` (default 12h).

## MQTT discovery and protocols

MQTT is used for communication between the camera monitor and Home Assistant. In `main`, a `MqttPublisher`
is initialized to establish a connection to the broker. `publish_all_discovery()` is used to initialize
things - it tells the broker what entities exist (one confidence score per class per camera, and optionally
one image per camera). Further updates happen in callbacks, which then call `publisher.publish_[thing]`

## Thread safety notes

- Each `ONVIFCameraWrapper` streams frames on a background thread; the poll loop reads the latest frame via `get_last_frame()`. RTSP streaming and `get_last_frame` are protected by a lock.
- `CameraMonitor.poll()` is called from the main poll loop thread; callbacks are invoked synchronously within `poll()`. Max of one simultaneous poll call per monitor is enforced with a lock.
- The cleanup loop runs on a separate `threading.Thread`; it only touches the filesystem, not shared state.
- All `CameraMonitor` file modifications are lock-protected. `cleanup` also invalidates the cached file search results to avoid re-deleting a file twice.
