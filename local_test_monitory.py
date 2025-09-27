import time
from pathlib import Path

from camera_monitor import CameraMonitor
from background_model import TimestampAwareBackgroundSubtractor
from image_loader import get_all_timestamped_files_sorted
import cv2 as cv


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

        cv.putText(
            frame,
            human_readable_t,
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
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
    start_timestamp = time.mktime(
        time.strptime("2025-08-16 06:00:00", "%Y-%m-%d %H:%M:%S")
    )
    end_timestamp = time.mktime(
        time.strptime("2025-08-16 20:00:00", "%Y-%m-%d %H:%M:%S")
    )

    data_dir = Path("data")
    bg_model = TimestampAwareBackgroundSubtractor()
    monitor = CameraMonitor(
        brightness_threshold=40,
        history_seconds=300,
        bg_model=bg_model,
        output_dir=None,
        model_file=None,
    )
    rerun_with_logged_images(monitor, data_dir, start_timestamp, end_timestamp)
