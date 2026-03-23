from pathlib import Path

import cv2 as cv

from background_model import TimestampAwareBackgroundSubtractor
from camera_monitor import CameraMonitor
from cameras import ONVIFCameraWrapper, LoggedImagePseudoCamera


def run(monitor: CameraMonitor, poll_interval: float):
    while True:
        monitor.poll()
        _, frame = monitor.camera.get_last_frame()
        for b in monitor.last_blobs:
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
        cv.imshow("Frame", frame)
        k = cv.waitKey(int(poll_interval * 1000)) & 0xFF
        if k == ord("q"):
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    with open("secrets.yaml", "r") as f:
        secrets = yaml.safe_load(f)

    with open("local_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.live:
        camera = ONVIFCameraWrapper(
            host=secrets["host"],
            port=secrets["port"],
            username=secrets["username"],
            password=secrets["password"],
            resolution=tuple(secrets["resolution"]),
        )
    elif args.log:
        camera = LoggedImagePseudoCamera(args.data_dir)
    else:
        raise ValueError("Must specify either --live or --log")

    bg_model = TimestampAwareBackgroundSubtractor(
        history_seconds=config["history_seconds"],
        var_threshold=config["var_threshold"],
        detect_shadows=config["detect_shadows"],
        area_threshold=config["area_threshold"],
        shadow_correlation_threshold=config["shadow_correlation_threshold"],
        morph_radius=config["morph_radius"],
        morph_thresh=config["morph_thresh"],
        morph_iters=config["morph_iters"],
        default_fps=config["default_fps"],
        night_mode_kwargs={
            k[6:]: v for k, v in config.items() if k.startswith("night_")
        },
        region_of_interest=cv.imread(config["region_of_interest"], cv.IMREAD_GRAYSCALE),
        debug_dir=Path("debug"),
    )
    monitor = CameraMonitor(
        camera=camera,
        name="DebugMonitor",
        brightness_threshold=config["brightness_threshold"],
        history_seconds=config["history_seconds"],
        bg_model=bg_model,
        output_dir=args.data_dir if args.live else None,
    )
    run(monitor, config["poll_frequency"])
