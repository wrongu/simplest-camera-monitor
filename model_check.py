import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import jsonargparse
from tqdm import tqdm

from app.detection import YoloDetectionModel, create_detector
from app.image_loader import get_all_timestamped_files_sorted
from app.utils import BoundingBox


@dataclass(frozen=True)
class DetectionResult(object):
    model_uid: str
    image: Path
    detections: list[BoundingBox] = field(default_factory=list)

    def to_dict(self):
        return {
            "model_uid": self.model_uid,
            "image": str(self.image),
            "detections": [bb.to_dict() for bb in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            model_uid=d["model_uid"],
            image=Path(d["image"]),
            detections=[BoundingBox.from_dict(bb) for bb in d["detections"]],
        )

    def __eq__(self, other):
        return (self.model_uid, str(self.image)) == (other.model_uid, str(other.image))

    def __hash__(self):
        return hash((self.model_uid, self.image))


class JSONListStreamer:
    def __init__(self, out_file: Path):
        self.filename = out_file
        self.progress_filename = out_file.with_name(out_file.name + ".progress")
        self.f = open(self.progress_filename, "w")
        self.count = 0

    def __enter__(self):
        self.f.seek(0)
        self.f.write("[\n")
        self.count = 0
        return self

    def append(self, obj):
        if self.count > 0:
            self.f.write(",\n")
        json.dump(obj, self.f)
        self.count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.write("\n]")
        self.f.close()
        if self.filename.exists():
            self.filename.unlink()
        self.progress_filename.rename(self.filename)


def main(
    model_cfg: Path,
    image_dir: Path,
    memo_file: Path = Path("tmp") / "memo.json",
    threshold: float = 0.5,
):
    if memo_file.is_file():
        with open(memo_file) as f:
            results = set(map(DetectionResult.from_dict, json.load(f)))
    else:
        results = set()

    detection_model = create_detector(model_cfg)

    # Precompute everything
    model_uid = hash(detection_model._uid)
    with JSONListStreamer(memo_file) as record:
        for det in results:
            record.append(det.to_dict())
        for ts, image_file in tqdm(get_all_timestamped_files_sorted(image_dir)):
            # if DetectionResult(model_uid, image_file) not in results:
                bboxes = detection_model.process_frame(cv2.imread(str(image_file), cv2.IMREAD_COLOR_RGB), ts)
                bboxes = [box for box in bboxes if box.confidence > threshold]
                result = DetectionResult(model_uid, image_file, bboxes)
                results.add(result)
                # record.append(result.to_dict())
#
    # Display results in reverse chronological order
    results_sorted = list(sorted(results, key=lambda det: det.image))
    latest_detection = next(det for det in reversed(results_sorted) if det.detections)
    index = results_sorted.index(latest_detection)

    def display(idx):
        im = cv2.imread(str(results_sorted[idx].image))
        for bbox in results_sorted[idx].detections:
            bbox.draw(im, color=(255, 0, 0))
        cv2.putText(
            im,
            str(results_sorted[idx].image),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.imshow("DETECTIONS", im)

    quit = False
    while not quit:
        display(index)
        k = cv2.waitKey(0)
        if k in {ord("q"), 27}:
            quit = True
        elif k == ord(","):
            index = (index - 1) % len(results_sorted)
        elif k == ord("."):
            index = (index + 1) % len(results_sorted)
        elif k == ord("["):
            index = (index - 1) % len(results_sorted)
            while not results_sorted[index].detections:
                index = (index - 1) % len(results_sorted)
        elif k == ord("]"):
            index = (index + 1) % len(results_sorted)
            while not results_sorted[index].detections:
                index = (index + 1) % len(results_sorted)
        else:
            print("Key: ", k)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(main)
    args = parser.parse_args()

    main(**args.as_dict())
