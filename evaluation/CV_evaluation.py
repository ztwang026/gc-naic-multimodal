"""Evaluate the per-fold predictions produced by a cross-validation run.

Reads the prediction files written for every fold (`fold_*/train_pred.csv` and
`fold_*/val_pred.csv`) and reports AUC, accuracy, sensitivity, specificity, PPV
and NPV for each fold together with their mean and standard deviation.

The operating point of a fold is selected on its training split with Youden's J
statistic and then applied unchanged to the matching validation split.

Usage:
    python CV_evaluation.py --results-dir path/to/results
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

# Metrics reported for every fold, in the order they appear in the tables.
METRICS = ("AUC", "ACC", "Sensitivity", "Specificity", "PPV", "NPV")

# Predicted probability of the positive class stored for a given epoch, and the
# directory naming scheme used for the folds.
PRED_COLUMN_PATTERN = re.compile(r"^epoch(\d+)_pred_prob$")
FOLD_DIR_PATTERN = re.compile(r"fold_(\d+)")


# --------------------------------------------------------------------------- #
# Prediction files
# --------------------------------------------------------------------------- #
def find_fold_dirs(results_dir):
    """Return the fold directories in results_dir, ordered by fold index."""
    fold_dirs = sorted(
        glob.glob(os.path.join(results_dir, "fold_*")),
        key=lambda path: int(FOLD_DIR_PATTERN.search(os.path.basename(path)).group(1)),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"no fold_* directory found in {results_dir}")
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


def load_scores(csv_path, epoch=None):
    """Return (labels, scores) from a prediction file.

    If epoch is None, the last epoch present in the file is used.
    """
    frame = pd.read_csv(csv_path)
    selected = last_epoch(frame) if epoch is None else epoch
    column = f"epoch{selected}_pred_prob"
    if column not in frame.columns:
        raise ValueError(f"{csv_path} has no column '{column}'")
    y_true = frame["true_label"].to_numpy().astype(int)
    y_score = frame[column].to_numpy().astype(float)
    return y_true, y_score


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def youden_threshold(y_true, y_score):
    """Return the threshold maximising Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def metrics_at_threshold(y_true, y_score, threshold):
    """AUC plus the confusion-matrix metrics at the given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    return {
        "AUC": roc_auc_score(y_true, y_score),
        "ACC": ratio(tp + tn, tp + tn + fp + fn),
        "Sensitivity": ratio(tp, tp + fn),
        "Specificity": ratio(tn, tn + fp),
        "PPV": ratio(tp, tp + fp),
        "NPV": ratio(tn, tn + fn),
    }


def evaluate_folds(fold_dirs, epoch, train_file, val_file):
    """Evaluate every fold with a threshold fitted on its own training split."""
    train_metrics = {name: [] for name in METRICS}
    val_metrics = {name: [] for name in METRICS}
    thresholds = []

    for fold, fold_dir in enumerate(fold_dirs):
        y_true_train, y_score_train = load_scores(
            os.path.join(fold_dir, train_file), epoch
        )
        y_true_val, y_score_val = load_scores(os.path.join(fold_dir, val_file), epoch)

        threshold = youden_threshold(y_true_train, y_score_train)
        thresholds.append(threshold)

        train_row = metrics_at_threshold(y_true_train, y_score_train, threshold)
        val_row = metrics_at_threshold(y_true_val, y_score_val, threshold)
        for name in METRICS:
            train_metrics[name].append(train_row[name])
            val_metrics[name].append(val_row[name])

        logger.info(
            "fold %d/%d | threshold %.4f | train_auc %.4f | val_auc %.4f",
            fold + 1, len(fold_dirs), threshold, train_row["AUC"], val_row["AUC"],
        )

    return train_metrics, val_metrics, thresholds


# --------------------------------------------------------------------------- #
# Result tables
# --------------------------------------------------------------------------- #
def build_table(per_fold, thresholds, ddof):
    """One row per metric, one column per fold, plus a mean +/- std column."""
    rows = []
    for name in list(METRICS) + ["Youden_Threshold"]:
        values = thresholds if name == "Youden_Threshold" else per_fold[name]
        row = {"Metric": name}
        for fold, value in enumerate(values):
            row[f"Fold{fold + 1}"] = round(float(value), 4)
        row["Mean +/- Std"] = (
            f"{np.mean(values):.4f} +/- {np.std(values, ddof=ddof):.4f}"
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the per-fold predictions of a cross-validation run."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory holding the fold_* subdirectories.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to evaluate. Defaults to the last epoch present in the files.",
    )
    parser.add_argument(
        "--train-file",
        default="train_pred.csv",
        help="Name of the training-split prediction file inside each fold directory.",
    )
    parser.add_argument(
        "--val-file",
        default="val_pred.csv",
        help="Name of the validation-split prediction file inside each fold directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output workbook. Defaults to <results-dir>/cv_evaluation.xlsx.",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        choices=(0, 1),
        help="Delta degrees of freedom of the standard deviation "
             "(0: population, 1: sample).",
    )
    args = parser.parse_args()

    fold_dirs = find_fold_dirs(args.results_dir)
    logger.info("found %d folds in %s", len(fold_dirs), args.results_dir)

    train_metrics, val_metrics, thresholds = evaluate_folds(
        fold_dirs, args.epoch, args.train_file, args.val_file
    )
    train_table = build_table(train_metrics, thresholds, args.ddof)
    val_table = build_table(val_metrics, thresholds, args.ddof)

    output_path = args.output or os.path.join(args.results_dir, "cv_evaluation.xlsx")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        train_table.to_excel(writer, sheet_name="train", index=False)
        val_table.to_excel(writer, sheet_name="val", index=False)

    logger.info(
        "training splits, thresholds fitted in-sample\n%s",
        train_table.to_string(index=False),
    )
    logger.info(
        "validation splits, thresholds carried over from the training splits\n%s",
        val_table.to_string(index=False),
    )
    logger.info("done. results in %s", output_path)


if __name__ == "__main__":
    main()
