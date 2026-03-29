import logging
from pathlib import Path
from pprint import pprint

import cv2 as cv

from background_model import TimestampAwareBackgroundSubtractor, BoundingBox
from camera_monitor import CameraMonitor
from cameras import ONVIFCameraWrapper, LoggedImagePseudoCamera


def run(*monitors: CameraMonitor, poll_interval: float):
    def on_detection_callback(mon: CameraMonitor, boxes: list[BoundingBox]):
        _, frame = mon.camera.get_last_frame()
        if frame is not None:
            for b in boxes:
                x, y, w, h = b.bbox
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(
                    frame,
                    str(b.class_id),
                    (x - 10, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
            cv.imshow("Frame" + str(mon), frame)

    for mon in monitors:
        mon.on_detection = on_detection_callback

    while True:
        for mon in monitors:
            mon.poll()
            _, frame = mon.camera.get_last_frame()
            if frame is not None:
                cv.imshow("Frame" + str(mon), frame)
        k = cv.waitKey(int(poll_interval * 1000)) & 0xFF
        if k == ord("q"):
            break
    cv.destroyAllWindows()


def main(app_config, secrets, use_log):
    monitors = []
    for config in app_config["cameras"]:
        mon = CameraMonitor(
            camera=(
                LoggedImagePseudoCamera(Path(config["output_dir"]))
                if use_log
                else ONVIFCameraWrapper(**secrets[config["name"]])
            ),
            name=config["name"],
            brightness_threshold=config["brightness_threshold"],
            history_seconds=config["history_seconds"],
            bg_model=TimestampAwareBackgroundSubtractor(
                history_seconds=config["history_seconds"],
                var_threshold=config["var_threshold"],
                detect_shadows=config["detect_shadows"],
                area_threshold=config["area_threshold"],
                shadow_correlation_threshold=config["shadow_correlation_threshold"],
                morph_radius=config["morph_radius"],
                morph_thresh=config["morph_thresh"],
                morph_iters=config["morph_iters"],
                default_fps=1 / app_config["poll_frequency"],
                region_of_interest=cv.imread(
                    config["region_of_interest"], cv.IMREAD_GRAYSCALE
                ),
                night_mode_kwargs={
                    k[6:]: v for k, v in config.items() if k.startswith("night_")
                },
            ),
            save_blobs=config.get("save_blobs", True),
            model_file=config.get("model_file", None),
            output_dir=config["output_dir"],
            log=lambda msg, level="INFO": logging.log(
                level=logging.INFO, msg=f"[{level}]: {msg}"
            ),
            on_state_transition=print,
            on_detection=print,
        )
        monitors.append(mon)

    run(*monitors, poll_interval=app_config["poll_frequency"])


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    with open("secrets.yaml", "r") as f:
        secrets = yaml.safe_load(f)

    with open("local_config.yaml", "r") as f:
        app_config = yaml.safe_load(f)

    pprint(app_config)

    main(app_config, secrets, use_log=args.log)
