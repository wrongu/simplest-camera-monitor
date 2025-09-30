import time
from pathlib import Path

import cv2 as cv

from background_model import TimestampAwareBackgroundSubtractor
from camera_monitor import CameraMonitor
from cameras import Camera, ONVIFCameraWrapper
from image_loader import get_all_timestamped_files_sorted


def run_with_live_camera(camera: Camera, monitor: CameraMonitor, poll_interval: float):
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("No frame ready, waiting...")
            time.sleep(poll_interval / 2)
            continue

        t = time.time()
        blobs = monitor.process_frame(frame, t)
        for b in blobs:
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


def rerun_with_logged_images(
    monitor: CameraMonitor,
    data_dir: Path,
    start_timestamp: float = 0.0,
    end_timestamp: float = float("inf"),
):
    monitor.bg_model.reset()
    for t, f in get_all_timestamped_files_sorted(data_dir):
        human_readable_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
        print(human_readable_t, "\t", f)
        if t < start_timestamp:
            continue
        if t > end_timestamp:
            break
        frame = cv.imread(str(f))
        if frame is None:
            raise FileNotFoundError(f"Failed to load image {f}")
        blobs = monitor.process_frame(frame, t)

        for b in blobs:
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
        k = cv.waitKey(0 if blobs else 1000 // 30)
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

    bg_model = TimestampAwareBackgroundSubtractor()
    monitor = CameraMonitor(
        brightness_threshold=40,
        history_seconds=300,
        bg_model=bg_model,
        output_dir=args.data_dir,
        model_file=None,
    )

    if args.live:
        camera = ONVIFCameraWrapper(
            host=secrets["host"],
            port=secrets["port"],
            username=secrets["username"],
            password=secrets["password"],
            resolution=tuple(secrets["resolution"]),
        )
        run_with_live_camera(camera, monitor, 2)

    if args.log:
        rerun_with_logged_images(monitor, args.data_dir)
