# CLAUDE.md — Agent Context for camera-monitor

## Project history

This project was originally an AppDaemon app. AppDaemon was an unnecessary layer — it provided HA integration but added complexity and a custom Docker base image with OpenCV pre-built. That dependency is now removed. `main.py` is a native HA App that talks directly to the HA Supervisor REST API via `SUPERVISOR_TOKEN`.

The active development branch is `ha-app`. It is 3 commits ahead of `origin/ha-app` and not yet merged to `main`.

## Current state (as of 2026-04-02)

**Goal right now:** Get `main.py` running on the HA instance and logging images. Monitoring/alert logic is secondary; data collection for vision model training is the priority.

**Secondary goal:** Make `main.py` cleanly runnable on a dev machine without any HA-specific paths.

**Future goal:** Restructure the repo to cleanly separate the three components (see TODO.md). Don't restructure until the above goals are met.

## Component map

| File                     | Role                                                                    |
| ------------------------ | ----------------------------------------------------------------------- |
| `main.py`                | HA App entry point — reads config, spins up monitors, posts to HA API   |
| `camera_monitor.py`      | Core state machine — polls frames, runs bg subtraction, fires callbacks |
| `cameras.py`             | Camera backends: ONVIF, ESPHome HTTP, offline image replay              |
| `background_model.py`    | MOG2 background subtraction, blob extraction, shadow filtering          |
| `image_loader.py`        | Timestamped image file naming, sorting, migration, thinning             |
| `annotator.py`           | Interactive GUI for labeling blobs (offline, dev only)                  |
| `classifier.py`          | Train decision-tree classifier on annotated blobs (offline, dev only)   |
| `local_test_monitory.py` | Deprecated; will move to `main.py`                                      |
| `config.yaml`            | HA Add-on manifest (name, slug, arch, volume maps)                      |
| `Dockerfile`             | Alpine + OpenCV-from-source build                                       |

## Key conventions

- Image files use the path format `YYYY/MM/DD/HHMMSS.jpg` (new format). The old flat `YYYYMMDD_HHMMSS.jpg` format is still supported for reading. See `image_loader.py`.
- Camera state transitions and detections go through callbacks (`on_state_transition`, `on_detection`) — `main.py` implements these to post to HA; `local_test_monitory.py` implements them for display.
- All callback invocations in `camera_monitor.py` are wrapped in `try/except` — a bad callback must not crash the monitor.
- The cleanup loop in `main.py` runs every 4 hours and deletes images older than `log_lifespan` (default 12h).
- Secrets live in `secrets.yaml` (gitignored). Config references them with `!secret key`.

## HA Supervisor API

`main.py` authenticates via the `SUPERVISOR_TOKEN` environment variable (injected by HA automatically when running as an App). It posts to `http://supervisor/core/api/states/<entity_id>` to update binary sensor entities. Entity IDs are derived from camera `name` fields in the config.

## Thread safety notes

- Each `ONVIFCameraWrapper` streams frames on a background thread; the poll loop reads the latest frame via `get_last_frame()`.
- `CameraMonitor.poll()` is called from the main poll loop thread; callbacks are invoked synchronously within `poll()`.
- The cleanup loop runs on a separate `threading.Thread`; it only touches the filesystem, not shared state.

## Things to be careful about

- **Do not refactor prematurely.** The repo restructuring (separating the 3 components into subdirectories) should happen _after_ the HA App is confirmed working and local running is clean.
- **Do not add monitoring/alert logic yet.** Focus is on reliable image logging first.
- **OpenCV display calls (`cv2.imshow`, `cv2.waitKey`) must not run inside the HA container** — there is no display. These are only used in `annotator.py` and `local_test_monitory.py`.
- The `LoggedImagePseudoCamera` is for offline replay/testing only — it should never be instantiated in the HA entry point.

## Running locally vs. on HA

| Aspect       | Local (`local_test_monitory.py`)     | HA App (`main.py`)            |
| ------------ | ------------------------------------ | ----------------------------- |
| Config file  | `local_config.yaml` + `secrets.yaml` | `/config/camera_monitor.yaml` |
| Image output | Configurable local path              | `/media/<name>/`              |
| HA API       | None                                 | `SUPERVISOR_TOKEN` + REST     |
| Display      | OpenCV window                        | None                          |
| Entry point  | `python local_test_monitory.py`      | `python main.py` (Docker)     |
