"""Evaluate held-out predictions at an operating point taken from cross-validation.

The decision threshold is selected with Youden's J statistic on the pooled
out-of-fold predictions of the cross-validation run (`fold_*/val_pred.csv`) and
is then applied unchanged to the held-out predictions.
AUC, accuracy, sensitivity, specificity,PPV and NPV are reported with percentile
bootstrap confidence intervals.

Usage:
    python test_evaluation.py --cv-dir path/to/cv_results \
        --predictions path/to/inference/predictions.csv
"""

import argparse
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

METRICS = ("AUC", "ACC", "Sensitivity", "Specificity", "PPV", "NPV")

PRED_COLUMN_PATTERN = re.compile(r"^epoch(\d+)_pred_prob$")
FOLD_DIR_PATTERN = re.compile(r"fold_(\d+)")


# --------------------------------------------------------------------------- #
# Prediction files
# --------------------------------------------------------------------------- #
def find_fold_dirs(cv_dir):
    """Return the fold directories in cv_dir, ordered by fold index."""
    fold_dirs = sorted(
        glob.glob(os.path.join(cv_dir, "fold_*")),
        key=lambda path: int(FOLD_DIR_PATTERN.search(os.path.basename(path)).group(1)),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"no fold_* directory found in {cv_dir}")
    return fold_dirs


def last_epoch(frame):
    """Return the highest epoch index for which predictions were stored."""
    epochs = []
    for column in frame.columns:
        match = PRED_COLUMN_PATTERN.match(column)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise ValueError("no 'epoch<N>_pred_prob' column found")
    return max(epochs)


def require_columns(frame, csv_path, *columns):
    """Raise if any of the requested columns is absent from frame."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing column(s) {missing}; "
            f"available columns: {list(frame.columns)}"
        )


def load_predictions(csv_path, label_column, score_column):
    """Return (labels, scores) from a prediction file."""
    frame = pd.read_csv(csv_path)
    require_columns(frame, csv_path, label_column, score_column)
    return (
        frame[label_column].to_numpy().astype(int),
        frame[score_column].to_numpy().astype(float),
    )


def load_oof(fold_dirs, epoch, label_column, val_file):
    """Concatenate the validation predictions of every fold."""
    labels, scores = [], []
    for fold_dir in fold_dirs:
        csv_path = os.path.join(fold_dir, val_file)
        frame = pd.read_csv(csv_path)
        selected = last_epoch(frame) if epoch is None else epoch
        score_column = f"epoch{selected}_pred_prob"
        require_columns(frame, csv_path, label_column, score_column)
        labels.append(frame[label_column].to_numpy().astype(int))
        scores.append(frame[score_column].to_numpy().astype(float))
    return np.concatenate(labels), np.concatenate(scores)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def youden_threshold(y_true, y_score):
    """Return the threshold maximising Youden's J statistic, and J itself."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best]), float(tpr[best] - fpr[best])


def compute_metrics(y_true, y_score, threshold):
    """AUC plus the confusion-matrix metrics at the given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else float("nan")

    return {
        "AUC": roc_auc_score(y_true, y_score),
        "ACC": ratio(tp + tn, tp + tn + fp + fn),
        "Sensitivity": ratio(tp, tp + fn),
        "Specificity": ratio(tn, tn + fp),
        "PPV": ratio(tp, tp + fp),
        "NPV": ratio(tn, tn + fn),
    }


def bootstrap_ci(y_true, y_score, threshold, n_bootstrap, seed, ci_level):
    """Percentile confidence intervals obtained by resampling the cohort.

    The threshold is held fixed, so the intervals are conditional on the
    operating point selected on the out-of-fold predictions. Replicates that
    contain a single class are discarded because AUC is undefined for them.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    replicates = []

    for _ in range(n_bootstrap):
        index = rng.integers(0, n_samples, size=n_samples)
        y_true_boot = y_true[index]
        if len(np.unique(y_true_boot)) < 2:
            continue
        replicates.append(compute_metrics(y_true_boot, y_score[index], threshold))

    if not replicates:
        raise ValueError("no bootstrap replicate contained both classes")

    frame = pd.DataFrame(replicates)
    alpha = 1.0 - ci_level
    lower = frame.quantile(alpha / 2)
    upper = frame.quantile(1.0 - alpha / 2)
    return lower, upper, len(replicates)


def build_table(point, lower, upper):
    """One row per metric, formatted as 'estimate (lower-upper)'."""
    return pd.DataFrame(
        [
            {
                "Metric": name,
                "Value": f"{point[name]:.3f} "
                         f"({lower[name]:.3f}-{upper[name]:.3f})",
            }
            for name in METRICS
        ]
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate held-out predictions."
    )
    parser.add_argument(
        "--cv-dir",
        required=True,
        help="Directory holding the fold_* subdirectories of the cross-validation run.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV file with the predictions to evaluate.",
    )
    parser.add_argument(
        "--val-file",
        default="val_pred.csv",
        help="Name of the validation-split prediction file inside each fold directory.",
    )
    parser.add_argument(
        "--oof-epoch",
        type=int,
        default=None,
        help="Epoch of the out-of-fold predictions. Defaults to the last epoch present.",
    )
    parser.add_argument(
        "--label-column",
        default="true_label",
        help="Name of the ground-truth column.",
    )
    parser.add_argument(
        "--score-column",
        default="pred_prob",
        help="Name of the predicted-probability column in the evaluated file.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level of the intervals.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Defaults to <cv-dir>/test_metrics_with_ci.csv.",
    )
    args = parser.parse_args()

    fold_dirs = find_fold_dirs(args.cv_dir)
    y_true_oof, y_score_oof = load_oof(
        fold_dirs, args.oof_epoch, args.label_column, args.val_file
    )
    threshold, youden = youden_threshold(y_true_oof, y_score_oof)
    logger.info(
        "out-of-fold predictions | %d folds | n = %d | positive %d | negative %d",
        len(fold_dirs), len(y_true_oof), int(y_true_oof.sum()),
        int((y_true_oof == 0).sum()),
    )
    logger.info("threshold %.6f | Youden's J %.4f", threshold, youden)

    y_true, y_score = load_predictions(
        args.predictions, args.label_column, args.score_column
    )
    logger.info(
        "evaluated cohort | n = %d | positive %d | negative %d",
        len(y_true), int(y_true.sum()), int((y_true == 0).sum()),
    )

    point = compute_metrics(y_true, y_score, threshold)
    lower, upper, n_replicates = bootstrap_ci(
        y_true, y_score, threshold, args.n_bootstrap, args.seed, args.ci_level
    )
    logger.info(
        "bootstrap | %d/%d replicates retained | seed %d | %.0f%% intervals",
        n_replicates, args.n_bootstrap, args.seed, args.ci_level * 100,
    )

    table = build_table(point, lower, upper)
    output_path = args.output or os.path.join(args.cv_dir, "test_metrics_with_ci.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    table.to_csv(output_path, index=False)

    logger.info("results\n%s", table.to_string(index=False))
    logger.info("done. results in %s", output_path)


if __name__ == "__main__":
    main()
