import sys
from pathlib import Path

import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.datasets) < 2:
        sys.exit(1)

    assert args.output.is_absolute()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset_specs = []
    all_labels = set()
    for dataset in args.datasets:
        with open(dataset, "r") as f:
            dataset_specs.append(yaml.safe_load(f))
            all_labels |= set(enumerate(dataset_specs[-1]["names"]))

    # Ultralytics implicitly defines the id:name mapping by storing 'names' as a list in the yaml file. The *index* of
    # the name in that list is the numeric id in the label files. When merging datasets, we will *assert* that the same
    # names appear in the same order, lest a numeric label like '0' mean one class in one subset of images and another
    # class in a different subset.
    definitive_set_of_labels = {}
    for num, name in sorted(all_labels):
        if num in definitive_set_of_labels:
            assert definitive_set_of_labels[num] == name, (
                f"Two different names for the same label id {num}: "
                f"{definitive_set_of_labels[num]} != {name}"
            )
        definitive_set_of_labels[num] = name

    # ...furthermore, since list indices are the numeric IDs, we'd better ensure that the ids are precisely the
    # numbers 0...N-1, in order. Given that each of the datasets coming into this merge operation already have this
    # property and we did 'sorted' earlier, we expect this assertion never to fail. But still good to state the
    # requirement explicitly.
    assert all(
        i == j
        for i, j in zip(range(len(definitive_set_of_labels)), definitive_set_of_labels.keys())
    ), "Expected keys to be precisely 0..N-1 in order... something has gone wrong."

    # This is not as general as it could be. Ultralytics has different ways of specifying datasets. Here we assume (
    # because that's how our export_yolo.py fn works) that the dataset is formatted like train: path_to_file.txt and
    # where that txt file contains newline-separated absolute paths to .jpg files. Each .jpg file then has a
    # corresponding .txt file adjacent to it on disk.
    merged_spec = {
        "path": str(args.output),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(definitive_set_of_labels),
        "names": list(definitive_set_of_labels.values())
    }
    all_train_images, all_val_images = [], []
    for path, spec in zip(args.datasets, dataset_specs):
        with open(Path(spec["path"]) / spec["train"], "r") as f:
            all_train_images.extend([l.strip() for l in f.readlines() if l.strip()])
        with open(Path(spec["path"]) / spec["val"], "r") as f:
            all_val_images.extend([l.strip() for l in f.readlines() if l.strip()])

    # Write out merged list of train and val images
    with open(args.output / "train.txt", "w") as f:
        f.write("\n".join(all_train_images))
    with open(args.output / "val.txt", "w") as f:
        f.write("\n".join(all_val_images))
    with open(args.output / "dataset.yaml", "w") as f:
        yaml.dump(merged_spec, f)
