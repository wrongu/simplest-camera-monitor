import time
from tqdm import tqdm
import numpy as np

import cv2 as cv
import argparse
from pathlib import Path

from app.detection import create_detector
from app.image_loader import get_all_timestamped_files_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector-config",
        type=Path,
        required=True,
        help="Path to the detector configuration file",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to the images directory",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum number of frames to process",
    )
    args = parser.parse_args()

    start = time.time()
    detector = create_detector(args.detector_config)
    elapsed = time.time() - start
    print(f"Instantiated detector in {elapsed:.2f} seconds")

    images = get_all_timestamped_files_sorted(args.image_dir)

    times = []
    for i, (ts, fil) in tqdm(zip(range(args.max_frames), images), total=args.max_frames):
        the_image = cv.imread(str(fil))

        start = time.time()
        dets = detector.process_frame(the_image, ts)
        elapsed = time.time() - start

        if dets:
            print(fil, len(dets), "detections")

        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print("Average time: ", avg_time)
    print("Standard deviation: ", std_time)


if __name__ == "__main__":
    main()
