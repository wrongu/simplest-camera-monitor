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
- Authenticates to the HA Supervisor REST API via `SUPERVISOR_TOKEN`
- Updates HA binary sensor entities on detections and state transitions
- Multi-threaded: poll loop + 4-hour cleanup loop
- Docker image compiled from Alpine with OpenCV built from source

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
