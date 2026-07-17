# Camera Monitor

ONVIF/ESPHome camera surveillance system for Home Assistant. Detects motion via background subtraction, logs timestamped images, and optionally classifies detected objects with a trained decision-tree model.

## Architecture

The project has three logically distinct components (currently co-located; restructuring is a planned next step):

### 1. Camera Monitoring & Logging (`camera_monitor.py`, `cameras.py`, `background_model.py`, `image_loader.py`)

The core engine. Each camera runs in its own `CameraMonitor` instance:

- Polls frames at a configurable interval
- Applies MOG2 background subtraction (time-adaptive, night-mode-aware)
- Extracts foreground blobs and logs timestamped images to disk
- State machine handles reconnection, reboot, and crash recovery
- Optional object classification via a trained pickled model

Camera backends: ONVIF (`ONVIFCameraWrapper`), ESPHome HTTP (`ESPHomeCameraWrapper`), or offline image replay (`LoggedImagePseudoCamera`).

### 2. Annotation & Model Training (`annotator.py`, `classifier.py`)

Offline tooling to build labeled training datasets and train a decision-tree classifier:

- `annotator.py` — interactive OpenCV GUI for drawing bounding boxes and assigning class labels to detected blobs; saves JSON annotation files
- `classifier.py` — loads annotations, featurizes blobs (Hu moments, color, bbox geometry), does grid-search CV over `DecisionTree + SelectKBest + StandardScaler`, saves a pickle

### 3. Home Assistant App (`main.py`, `config.yaml`, `Dockerfile`)

Runs the monitor as a native HA App:

- Reads camera config from `/config/camera_monitor.yaml` on the HA host
- Reports detections to Home Assistant via **MQTT Discovery** (default) or the legacy
  Supervisor REST API (`reporting: rest`)
- Multi-threaded: poll loop + 4-hour cleanup loop
- Docker image compiled from Alpine with OpenCV built from source

## Home Assistant entities (MQTT Discovery)

The app does all the CV work and publishes results to MQTT; must have HA's built-in MQTT integration 
running. MQTT turns results into entities.

For every `(camera, class)` in `watch_for_class`, HA auto-creates, grouped under a per-camera
device:

- `sensor.<cam>_<class>_confidence` — detection confidence for that class in the latest
  frame, as a percentage. Max over all boxes. HA tracks statistics over time. Caveat:
  will be binary (0% or 100%) if using the `BackgroundModelWithMorphologyClassifier` detector
- `image.<cam>_last_detection` — the latest annotated frame with boxes drawn (one per camera).

Camera health drives availability: entities go **Unavailable** when the camera can't connect,
and a hard add-on crash marks everything unavailable via the MQTT Last Will.

### Broker setup

- **As an add-on:** install the official *Mosquitto broker* add-on and the *MQTT* integration.
  With `services: [mqtt:need]` in the manifest, the broker is injected automatically — no
  credential config needed.
- **On a dev machine:** point at any broker with an `mqtt:` block in the config (see
  `example_config.yaml`).

### Dashboard card

A Picture Entity card gives the "last detected objects" view:

```yaml
type: picture-entity
entity: image.frontdoor_frontdoor_last_detection
camera_view: auto
show_state: false
```

Replace `frontdoor` with your camera name slugified (lowercased, spaces → underscores).

## Running as a Home Assistant App

1. Clone this repo into `/addons/local/` on your HA host
2. Install from **Settings → Add-ons → Local add-ons** (expected to take a while.. the docker image is slow to build)
3. Start the add-on; logs appear in the add-on log panel

Image logs are written to `/media/` (mapped via `media:rw` in `config.yaml`).

## Configuration

The app expects a YAML file with a top-level `cameras` list. See `example_config.yaml`

Secrets can be stored in a separate `secrets.yaml` and referenced with `!secret key`.

## Development Setup

See [these HA docs](https://developers.home-assistant.io/docs/apps/testing).
