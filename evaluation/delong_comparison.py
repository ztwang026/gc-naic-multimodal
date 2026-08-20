"""Pairwise DeLong tests between the ROC curves of the three models.

Reads the predictions of the baseline (BL), transfer-learning (TL) and
multimodal (ML) models on the same samples and compares their AUCs with the
DeLong test for correlated ROC curves, using the fast implementation of Sun and
Xu (2014). All pairwise comparisons are reported with the raw p-value and with
Holm and Benjamini-Hochberg adjusted p-values. The prediction file is expected
next to this script and the results are written beside it.


Usage:
    python delong_comparison.py
"""

import itertools
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "BL_TL_ML_predictions.xlsx"
OUTPUT_FILE = "delong_comparison.csv"

LABEL_COLUMN = "true_label"
MODEL_COLUMNS = {
    "BL": "pred_prob_BL",
    "TL": "pred_prob_TL",
    "ML": "pred_prob_ML",
}


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def load_predictions(path, label_column, model_columns):
    """Return the labels and a (n_models, n_samples) array of scores."""
    reader = pd.read_csv if path.lower().endswith(".csv") else pd.read_excel
    frame = reader(path)
    required = [label_column] + list(model_columns.values())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {missing}; "
            f"available columns: {list(frame.columns)}"
        )
    y_true = frame[label_column].to_numpy().astype(int)
    scores = np.vstack(
        [frame[column].to_numpy(dtype=float) for column in model_columns.values()]
    )
    return y_true, scores


# --------------------------------------------------------------------------- #
# DeLong test
# --------------------------------------------------------------------------- #
def midrank(values):
    """Return the ranks of values, averaged within groups of ties."""
    order = np.argsort(values)
    ordered = values[order]
    n_samples = len(values)
    ranks = np.zeros(n_samples)
    i = 0
    while i < n_samples:
        j = i
        while j < n_samples and ordered[j] == ordered[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n_samples)
    out[order] = ranks
    return out


def delong_auc_cov(y_true, scores):
    """Return the AUC of every model and the covariance matrix of those AUCs.

    scores is a (n_models, n_samples) array of predicted probabilities of the
    positive class, evaluated on the same samples for every model.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    scores = np.asarray(scores, dtype=float)[:, order]

    n_pos = int(y_true.sum())
    n_neg = scores.shape[1] - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both classes must be present in the labels")

    n_models = scores.shape[0]
    tx = np.vstack([midrank(scores[row, :n_pos]) for row in range(n_models)])
    ty = np.vstack([midrank(scores[row, n_pos:]) for row in range(n_models)])
    tz = np.vstack([midrank(scores[row]) for row in range(n_models)])

    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.cov(v01) / n_pos + np.cov(v10) / n_neg
    return aucs, np.atleast_2d(cov)


def delong_pvalue(aucs, cov, i, j):
    """Two-sided DeLong test of AUC_i against AUC_j."""
    variance = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    if variance <= 0:
        raise ValueError(
            "non-positive variance of the AUC difference; the two score columns "
            "are probably identical"
        )
    z = (aucs[i] - aucs[j]) / np.sqrt(variance)
    return float(z), float(2 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #
def holm(p_values):
    """Holm step-down adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n_tests)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n_tests - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg adjusted p-values controlling the false discovery rate."""
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)
    order = np.argsort(p_values)
    scaled = p_values[order] * n_tests / (np.arange(n_tests) + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n_tests)
    adjusted[order] = np.minimum(scaled, 1.0)
    return adjusted


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    data_path = os.path.join(SCRIPT_DIR, DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"prediction file not found: {data_path}")

    y_true, scores = load_predictions(data_path, LABEL_COLUMN, MODEL_COLUMNS)
    names = list(MODEL_COLUMNS)
    aucs, cov = delong_auc_cov(y_true, scores)

    logger.info(
        "n = %d | positive %d | negative %d",
        len(y_true), int(y_true.sum()), int((y_true == 0).sum()),
    )
    for name, auc in zip(names, aucs):
        logger.info("%s | AUC %.4f", name, auc)

    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        z, p_value = delong_pvalue(aucs, cov, i, j)
        rows.append(
            {
                "comparison": f"{names[i]} vs {names[j]}",
                "auc_difference": aucs[i] - aucs[j],
                "z": z,
                "p_raw": p_value,
            }
        )

    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p_raw"].to_numpy())
    table["p_bh"] = benjamini_hochberg(table["p_raw"].to_numpy())

    output_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    table.to_csv(output_path, index=False)

    formatted = table.copy()
    for column in ("auc_difference", "z"):
        formatted[column] = formatted[column].map("{:.4f}".format)
    for column in ("p_raw", "p_holm", "p_bh"):
        formatted[column] = formatted[column].map("{:.4g}".format)
    logger.info("pairwise comparisons\n%s", formatted.to_string(index=False))
    logger.info("done. results in %s", output_path)


if __name__ == "__main__":
    main()
