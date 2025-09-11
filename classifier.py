from pathlib import Path
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Optional
import cv2 as cv
import json
import numpy as np


def bgr_to_xy(b, g, r):
    """Convert float BGR color to CIE 1931 xy chromaticity."""

    # Convert RGB to XYZ
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Convert XYZ to xy
    if (X + Y + Z) == 0:
        return (0.0, 0.0)

    return X / (X + Y + Z), Y / (X + Y + Z)


def featurize(blob):
    fg_xy = np.array(bgr_to_xy(*blob["fg_bgr"]))
    bg_xy = np.array(bgr_to_xy(*blob["bg_bgr"]))
    color_features = [*fg_xy, *bg_xy]
    # return blob["moments"] + [blob["area"]] + color_features
    return [np.log1p(blob["area"])] + color_features


def _sanity_check_labels(annotations: dict):
    unique_labels = set()
    for key, blobs in annotations.items():
        if key == "labels":
            continue
        for blob in blobs:
            if "label" not in blob:
                raise ValueError(f"Blob in {key} missing 'label' field.")
            if blob["label"] not in annotations["labels"]:
                raise ValueError(
                    f"Blob in {key} has unknown label '{blob['label']}'. Known labels: {annotations['labels']}"
                )
            unique_labels.add(blob["label"])

    # It's expected that labels are strings corresponding to integers {0, 1, 2, ..., K-1} for some K
    unique_labels = {int(label) for label in unique_labels}
    k = len(annotations["labels"])
    if unique_labels != set(range(k)):
        raise ValueError(
            f"Labels must be integers in the range [0, {k-1}]. Found: {unique_labels}"
        )


def load_annotations_as_data(
    annotations_file: Path,
    binary_detection: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[int]], dict[int, str]]:
    """Convert JSON dict of annotations to (X, y, files, bboxes)"""
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    with open(annotations_file) as f:
        annotations = json.load(f)

    _sanity_check_labels(annotations)

    if binary_detection is not None:
        label_handler = lambda lbl: 1 if annotations["labels"][lbl] == binary_detection else 0
        label_lookup = {0: f"not {binary_detection}", 1: binary_detection}
    else:
        label_handler = int
        label_lookup = {int(lbl): annotations["labels"][lbl] for lbl in annotations["labels"]}

    features, labels, files, bboxes = [], [], [], []
    n_loaded, n_skipped = 0, 0
    for key, blobs in annotations.items():
        if key == "labels":
            continue
        for blob in blobs:
            try:
                fg_xy = np.array(bgr_to_xy(*blob["fg_bgr"]))
                bg_xy = np.array(bgr_to_xy(*blob["bg_bgr"]))
                color_features = [*fg_xy, *bg_xy]
                # feat = blob["moments"] + [blob["area"]] + color_features
                feat = [np.log(blob["area"])] + color_features
                features.append(feat)
                labels.append(label_handler(blob["label"]))
                files.append(key)
                bboxes.append(blob["bounding_box"])
                n_loaded += 1
            except KeyError:
                n_skipped += 1
                continue
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, files, bboxes, label_lookup


def train(X_train, y_train, rebalance: bool) -> GaussianNB:
    clf = GaussianNB()
    if rebalance:
        class_counts = np.bincount(y_train)
        n_0 = class_counts[0]
        weight = np.array([n_0 / class_counts[y] for y in y_train], dtype=float)
    else:
        weight = np.ones((len(y_train),), dtype=float)

    clf.fit(X_train, y_train, sample_weight=weight)
    return clf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Gaussian Naive Bayes classifier to detect deer in images based on blob features."
    )
    parser.add_argument(
        "--annot_file",
        type=Path,
        required=True,
        help="Path to the JSON file containing blob annotations.",
    )
    parser.add_argument(
        "--binary-detection",
        type=str,
        default=None,
        help="Optional label to focus on, making a binary classifier (e.g. 'person' vs 'not person')"
    )
    parser.add_argument(
        "--model_file",
        type=Path,
        default="deer_classifier.pkl",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing.",
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Rebalance classes during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize features and misclassifications.",
    )
    args = parser.parse_args()

    features, labels, files, bboxes, label_lookup = load_annotations_as_data(
        args.annot_file, args.binary_detection
    )

    X_train, X_test, y_train, y_test, files_train, files_test, bb_train, bb_test = (
        train_test_split(
            features,
            labels,
            files,
            bboxes,
            test_size=args.test_fraction,
            random_state=args.seed,
        )
    )

    print("Training balance:", sum(y_train == 1), "/", len(y_train))
    print("Testing balance:", sum(y_test == 1), "/", len(y_test))

    clf = train(X_train, y_train, args.rebalance)

    if args.model_file is not None:
        with open(args.model_file, "wb") as f:
            pickle.dump({"model": clf, "label_lookup": label_lookup}, f)

    ## Feature visualization with classes ##

    if args.visualize:
        feature_names = ["area", "fg_x", "fg_y", "bg_x", "bg_y"]

        import matplotlib.pyplot as plt

        n_features = features.shape[1]

        # Plotgrid of feature distributions pairwise
        fig, axes = plt.subplots(n_features, n_features, figsize=(10, 10))
        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]
                if i == j:
                    vmin = np.min(features[:, i])
                    vmax = np.max(features[:, j])
                    x = np.linspace(vmin, vmax, 20)
                    for k in range(len(label_lookup)):
                        # Data hists
                        h = ax.hist(
                            features[labels == k, i],
                            bins=x,
                            alpha=0.5,
                            label=label_lookup[k],
                            density=True,
                        )
                        # Fitted Gaussians
                        mu = clf.theta_[k, i]
                        sigma = np.sqrt(clf.var_[k, i])
                        p = np.exp(
                            -0.5 * ((x - mu) / sigma) ** 2
                            - 0.5 * np.log(2 * np.pi * sigma**2)
                        )
                        ax.plot(x, p, color=h[2][0].get_facecolor(), lw=1)
                    if i == 0:
                        ax.legend()
                else:
                    for k in range(len(label_lookup)):
                        ax.scatter(
                            features[labels == k, j],
                            features[labels == k, i],
                            alpha=0.5,
                            label=label_lookup[k],
                            s=1,
                        )
                if i == n_features - 1:
                    ax.set_xlabel(feature_names[j])
                if j == 0:
                    ax.set_ylabel(feature_names[i])
        plt.tight_layout()
        plt.show()

    # Print classification report
    y_pred = clf.predict(X_test)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[label_lookup[i] for i in range(len(label_lookup)) if i in set(y_test)],
        )
    )
