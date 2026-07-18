# Camera Monitor

ONVIF/ESPHome camera surveillance system for Home Assistant. Detects motion via background subtraction, logs timestamped images, and optionally classifies detected objects with a trained decision-tree model.

## Architecture

The project has three logically distinct components (currently co-located; restructuring is a planned next step):

### 1. Camera Monitoring & Logging (`app/camera_monitor.py`, `app/cameras.py`, `app/image_loader.py`, `app/detection.py`)

The core engine. Each camera runs in its own `CameraMonitor` instance:

- Polls frames at a configurable interval
- Performs object detection with some detector imported from `detection.py`
- State machine handles reconnection, reboot, and crash recovery
- Callbacks for state transitions, new images, and detection results

Camera backends: ONVIF (`ONVIFCameraWrapper`), ESPHome HTTP (`ESPHomeCameraWrapper`), or offline 
image replay (`LoggedImagePseudoCamera`).

### 2. Annotation & Model Training (`app/annotator.py`, `app/classifier.py`, `models/train_yolo.py`)

Offline tooling to build labeled training datasets and train a decision-tree classifier:

- `app/annotator.py` — launches an in-browser app for drawing bounding boxes and assigning class labels to detected blobs; saves JSON annotation files. Annotation done separately for each camera.
- `app/classifier.py` — deprecated scikit-learn classification
- `models/export_yolo.py` — export from our custom annotation file format to ultralytics/YOLO format. Done once per camera.
- `models/merge_yolo_datasets.py` — merge YOLO-style datasets from multiple cameras into one big dataset.
- `models/train_yolo.py` — train and export a YOLO object detection model.

### 3. Home Assistant App (`main.py`, `config.yaml`, `Dockerfile`)

Runs the monitor as a native HA App:

- Reads camera config from `/config/camera_monitor.yaml` on the HA host (see `example_config.yaml`)
- Reports detections to Home Assistant via **MQTT Discovery** (or set 'reporting' to 'none')
- Multi-threaded: poll loop + cleanup loop
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

Image logs are written to `/media/<cam>` (mapped via `media:rw` in `config.yaml`) per camera.

Model training lifecycle: 

1. periodically download data from the HA instance to permanently store it on a dev machine
2. run the annotator script on that dev machine, once per camera
3. export the annotations to YOLO format
4. merge the YOLO datasets from all cameras into one big dataset
5. train a YOLO model on the merged dataset and export to onnx
6. look at run metrics... if good enough, upload new `best.onnx` back to homeassistant, wherever
   the `detector_config` yaml files point to.

__Example:__ if `best.onnx` is uploaded to `/addon_configs/local_camera_monitor/models/best.onnx` 
on the HA instance, and `/addon_configs/local_camera_monitor/config.yaml` contains

```yaml
---
cameras:
  - name: "MyCamera"
    # other config options here
    detector_config: "/config/models/mydetector.yaml"
    roi: "/media/mycamera/roi.png"
    confidence_threshold: 0.5
```

...then on the HA instance, `/addon_configs/local_camera_monitor/models/mydetector.yaml` should look like

```yaml
---
class: OnnxYoloDetectionModel
weights: /config/models/best.onnx
```

## Configuration

The app expects a YAML file with a top-level `cameras` list. See `example_config.yaml`

## Development Setup

See [these HA docs](https://developers.home-assistant.io/docs/apps/testing) to do a full local HA test.

Otherwise, run `main.py` with `--live-debug` and set the `CONFIG_PATH` environment variable. Example:

    CONFIG_PATH=local/live.yaml python3 -m app.main --live-debug

...and inside that local config file, set `reporting: none` or if using `reporting: mqtt`, be sure
to set up the `mqtt` config block to point to a dev/debug `mqtt` broker.
