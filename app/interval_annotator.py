import os
from datetime import datetime
from pathlib import Path

import portion

from annotator import main, get_parser, AnnotationFile


def interval_covered(str_interval, exist_intervals):
    lower, upper = str_interval.split(":")
    check_interval = portion.closed(
        datetime.strptime(lower.replace("\\","/"), "%Y/%m/%d/%H%M%S.jpg"),
        datetime.strptime(upper.replace("\\","/"), "%Y/%m/%d/%H%M%S.jpg"),
    )
    return exist_intervals.contains(check_interval)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--intervals", type=Path)
    args = parser.parse_args()

    with open(args.intervals, "r") as f:
        intervals = [
            line.strip().replace("/", os.path.sep).replace("\\", os.path.sep)
            for line in f.readlines()
        ]

    exist_intervals = portion.Interval()
    for exist_annot in Path(args.image_dir).glob("*.json"):
        annot = AnnotationFile.load(exist_annot)
        interval = annot.get_interval(fudge=True)
        if interval is not None:
            exist_intervals |= interval

    for interval in reversed(intervals):
        print(interval)
        if interval_covered(interval, exist_intervals):
            print("-skip-")
            continue
        args.interval = interval.split(":")
        main(args)
