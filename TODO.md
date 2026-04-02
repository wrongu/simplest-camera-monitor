# TODO

Priorities in order. Do not jump ahead — each step unblocks the next.

---

## 1. Get `main.py` running on HA (image logging only) [URGENT]

The Docker image builds (see recent commits fighting with OpenCV on Alpine). Need to confirm the add-on actually runs and logs images.

- [ ] Verify `config.yaml` (HA App manifest) is correct for current HA version
- [ ] Create `/config/camera_monitor.yaml` on the HA host with at least one camera entry
- [ ] Install and start the add-on; check logs for startup errors
- [ ] Confirm images are being written to `/media/<name>/`
- [ ] Confirm HA binary sensor entities are being created/updated

**Success criterion:** Images accumulating on disk from at least one camera.

Note: Monitoring/alert callback logic already exists but is not the priority. The image logging path (`on_detection` → save image) is what matters now.

---

## 2. Clean up local running (`local_test_monitory.py`)

Currently requires manually creating `local_config.yaml` and `secrets.yaml` with no documented schema.

- [ ] Create `local_config.yaml.example` with all supported fields and comments
- [ ] Create `secrets.yaml.example`
- [ ] Fix any hardcoded HA-specific paths in `local_test_monitory.py`
- [ ] Confirm `python local_test_monitory.py` works on a dev machine end-to-end (live camera or offline replay)
- [ ] Document the local setup steps in README.md

---

## 3. Repo restructure — separate the 3 components

Do this _after_ both HA and local running are confirmed stable.

Proposed layout:

```
camera-monitor/
├── monitor/              # component 1: camera monitoring + logging
│   ├── camera_monitor.py
│   ├── cameras.py
│   ├── background_model.py
│   └── image_loader.py
├── training/             # component 2: annotation + model training
│   ├── annotator.py
│   ├── classifier.py
│   └── ...
├── ha-app/               # component 3: HA App wrapper
│   ├── main.py
│   ├── config.yaml
│   └── Dockerfile
├── local_runner.py       # replaces local_test_monitory.py (name typo: "monitory")
└── ...
```

- [ ] Decide on directory names (proposal above)
- [ ] Move files, update imports
- [ ] Update Dockerfile paths
- [ ] Fix the typo: `local_test_monitory.py` → `local_runner.py`
- [ ] Update README.md

---

## 4. Light web UI (low priority)

A simple local web server for interacting with the running system.

Potential features (not yet decided):

- Live camera feed viewer
- Recent detection log with thumbnails
- Trigger annotation session from browser
- System status (camera states, disk usage)

- [ ] Choose framework (FastAPI + HTMX? Flask? something else?)
- [ ] Decide what HA integration looks like (standalone vs. HA panel integration)
- [ ] Implement

---

## Backlog / known issues

- `local_test_monitory.py` has a typo in the filename ("monitory")
- `.gitmodules` exists but is empty — check if a submodule was removed and clean up
- `data/` directory has real captured images from 2025-09-11; confirm gitignore covers it
- `requirements-dev.txt` still lists `appdaemon` — remove it
- The devcontainer (`.devcontainer/`) is untracked; decide whether to commit it
