import json
import pickle
from pathlib import Path
from typing import Optional, override

import cv2 as cv
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection._univariate_selection import _BaseFilter
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from background_model import ForegroundBlob, _is_night_mode_image

FEATURE_NAMES = [
    "day_night",
    "m00",
    "m10",
    "m01",
    "m20",
    "m11",
    "m02",
    "m30",
    "m21",
    "m12",
    "m03",
    "mu20",
    "mu11",
    "mu02",
    "mu30",
    "mu21",
    "mu12",
    "mu03",
    "nu20",
    "nu11",
    "nu02",
    "nu30",
    "nu21",
    "nu12",
    "nu03",
    "x",
    "y",
    "w",
    "h",
    "shadow_correlation",
    "b",
    "g",
    "r",
]


def featurize(blob: ForegroundBlob) -> np.ndarray:
    moments = cv.moments(blob.mask)
    bbox = blob.bbox
    is_night = 1 if _is_night_mode_image(blob.image) else 0
    return np.array(
        [
            is_night,
            *moments.values(),
            *bbox,
            np.mean(blob.shadow_correlation()),
            *np.median(blob.image[blob.mask > 0], axis=0),
        ]
    )


class DropNamedFeatures(_BaseFilter):
    def __init__(self, names: list[str]):
        self.names = names
        self.indices = [FEATURE_NAMES.index(name) for name in names]
        super().__init__(score_func=DropNamedFeatures.dummy_score)

    @staticmethod
    def dummy_score(X, y):
        return np.zeros(X.shape[1]), np.ones(X.shape[1])

    @override
    def _get_support_mask(self) -> np.ndarray:
        mask = np.ones(len(FEATURE_NAMES), dtype=bool)
        mask[self.indices] = False
        return mask


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
    annotations_file: Path, binary_detection: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[int]], dict[int, str]]:
    """Convert JSON dict of annotations to (X, y, files, bboxes)"""
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    with open(annotations_file) as f:
        annotations = json.load(f)

    _sanity_check_labels(annotations)

    if binary_detection is not None:
        label_handler = lambda lbl: (
            1 if annotations["labels"][lbl] == binary_detection else 0
        )
        label_lookup = {0: f"not {binary_detection}", 1: binary_detection}
    else:
        label_handler = int
        label_lookup = {
            int(lbl): annotations["labels"][lbl] for lbl in annotations["labels"]
        }

    features, labels, files, bboxes = [], [], [], []
    n_loaded, n_skipped = 0, 0
    for key, annots in annotations.items():
        if key == "labels":
            continue
        for ann in annots:
            if ann.get("features", []):
                features.append(ann["features"])
                labels.append(label_handler(ann["label"]))
                files.append(key)
                bboxes.append(ann["bbox"])
                n_loaded += 1
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, files, bboxes, label_lookup


def new_estimator(drop_features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("drop", DropNamedFeatures(drop_features)),
            ("zscore", StandardScaler()),
            ("features", SelectKBest()),
            ("classifier", DecisionTreeClassifier()),
        ]
    )


def get_sample_weights(classes, rebalance):
    class_counts = np.bincount(classes)
    n_0 = class_counts[0]
    equal_weight = np.ones((len(classes),), dtype=float)
    rebalance_weight = np.array([n_0 / class_counts[y] for y in classes], dtype=float)
    if isinstance(rebalance, bool) and rebalance:
        weight = rebalance_weight
    elif isinstance(rebalance, float) and rebalance > 0:
        weight = np.exp(
            (1 - rebalance) * np.log(equal_weight)
            + rebalance * np.log(rebalance_weight)
        )
    else:
        weight = equal_weight
    return weight


def model_selection(x, y, rebalance: bool | float, max_k: int, cv: int = 5) -> Pipeline:
    # Drop raw moments
    features_to_drop = [
        "m00",
        "m10",
        "m01",
        "m20",
        "m11",
        "m02",
        "m30",
        "m21",
        "m12",
        "m03",
    ]
    cv_model = new_estimator(features_to_drop)
    weight = get_sample_weights(y, rebalance)

    searcher = GridSearchCV(
        estimator=cv_model,
        param_grid={
            "features__k": np.arange(1, max_k + 1),
            "classifier__max_depth": [3, 5, 10, 15, 20, None],
        },
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    searcher.fit(x, y, classifier__sample_weight=weight)
    print("Score per parameter set:")
    for params, mean_score, scores in zip(
        searcher.cv_results_["params"],
        searcher.cv_results_["mean_test_score"],
        searcher.cv_results_["std_test_score"],
    ):
        print(f"  {params}: {mean_score:.3f} (+/-{scores * 2:.3f})")
    print("Best parameters from CV:", searcher.best_params_)

    # Update classifier with best params
    cv_model.set_params(**searcher.best_params_)
    cv_model.fit(x, y, classifier__sample_weight=weight)
    return cv_model


def main(
    annot_file: Path,
    binary_detection: Optional[str],
    test_fraction: float,
    seed: int,
    rebalance: bool | float,
    k_features: int,
    model_file: Path,
    visualize: bool = False,
):
    global FEATURE_NAMES
    features, labels, files, bboxes, label_lookup = load_annotations_as_data(
        annot_file, binary_detection
    )

    X_train, X_test, y_train, y_test, files_train, files_test, bb_train, bb_test = (
        train_test_split(
            features,
            labels,
            files,
            bboxes,
            test_size=test_fraction,
            random_state=seed,
        )
    )

    # Cross-validation hyperparameter search
    model = model_selection(X_train, y_train, rebalance, k_features)

    # Save trained model to disk
    if model_file is not None:
        with open(model_file, "wb") as f:
            pickle.dump({"model": model, "label_lookup": label_lookup}, f)

    ## Feature visualization with classes ##
    if visualize:
        import matplotlib.pyplot as plt

        dropper : DropNamedFeatures = model[0]
        which_features = np.arange(len(FEATURE_NAMES))
        which_features = which_features[dropper.get_support()]
        selector: SelectKBest = model[2]
        which_features = which_features[selector.get_support()]
        n_features = len(which_features)
        sub_features = features[:, which_features]

        # Plotgrid of feature distributions pairwise
        fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))
        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]
                if i == j:
                    vmin = np.min(sub_features[:, i])
                    vmax = np.max(sub_features[:, j])
                    x = np.linspace(vmin, vmax, 20)
                    for k_features in range(len(label_lookup)):
                        # Data hists
                        h = ax.hist(
                            sub_features[labels == k_features, i],
                            bins=x,
                            alpha=0.5,
                            label=label_lookup[k_features],
                            density=True,
                        )
                    if i == 0:
                        ax.legend()
                else:
                    for k_features in range(len(label_lookup)):
                        ax.scatter(
                            sub_features[labels == k_features, j],
                            sub_features[labels == k_features, i],
                            alpha=0.5,
                            label=label_lookup[k_features],
                            s=1,
                        )
                if i == n_features - 1:
                    ax.set_xlabel(FEATURE_NAMES[j])
                if j == 0:
                    ax.set_ylabel(FEATURE_NAMES[i])
        plt.tight_layout()
        plt.show()

    # Print classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    if visualize:
        # Confusion matrix
        disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true")
        disp.figure_.suptitle("Confusion Matrix")
        disp.figure_.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Decision Tree classifier to classify based on blob features."
    )
    parser.add_argument(
        "--annot_file",
        type=Path,
        required=True,
        help="Path to the JSON file containing blob annotations.",
    )
    parser.add_argument(
        "--binary_detection",
        type=str,
        default=None,
        help="Optional label to focus on, making a binary classifier (e.g. 'person' vs 'not person')",
    )
    parser.add_argument(
        "--model_file",
        type=Path,
        default="classifier.pkl",
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
        type=float,
        default=0.0,
        help="Amount to rebalance classes during training.",
    )
    parser.add_argument(
        "--k_features",
        type=int,
        default=8,
        help="Number of features to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4796487,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize features and misclassifications.",
    )
    args = parser.parse_args()

    main(**(vars(args)))
