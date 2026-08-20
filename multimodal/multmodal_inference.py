"""Evaluate the multimodal fusion variants on the NAIC test cohort.

The three variants fitted by multimodal_fusion_xgb.py are applied unchanged to
a held-out cohort:

    ct_only         the CT score used directly as the predicted probability
    clinical_only   the refitted gradient boosting model with a zero offset
    multimodal      the refitted gradient boosting model with the CT score
                    supplied as a fixed offset

Usage:
    python multimodal_fusion_test.py --model-dir results/multimodal_fusion
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DATA_FILE = "data/multimodal_features.xlsx"
SHEET_NAME = "NAIC-test"
MODEL_DIR = "outputs/multimodal_fusion"

ID_COL = "ID"
LABEL_COL = "true_label"
CT_SCORE_COL = "CT_score"

FEATURE_COLS = [
    "CA199",
    "CEA",
    "tumor_necros_density",
    "tumor_inflam_density",
    "RBC",
]
CATEGORICAL_COLS = []

VARIANTS = ("ct_only", "clinical_only", "multimodal")

METRICS = ("AUC", "ACC", "SEN", "SPE", "PPV", "NPV")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 97.5)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_features(df):
    X = df[FEATURE_COLS].reset_index(drop=True)
    for col in X.columns:
        if col not in CATEGORICAL_COLS:
            numeric = pd.to_numeric(X[col], errors="coerce")
            n_lost = int((numeric.isna() & X[col].notna()).sum())
            if n_lost:
                logger.warning("%s | %d non-numeric values set to missing",
                               col, n_lost)
            X[col] = numeric
    return X


def base_margin(variant, ct_scores, eps=1e-3):
    """Offset added to the raw model score, on the log-odds scale."""
    if variant != "multimodal":
        return pd.Series(np.zeros(len(ct_scores)), index=ct_scores.index)
    p = np.clip(ct_scores.to_numpy(dtype=float), eps, 1 - eps)
    return pd.Series(np.log(p / (1 - p)), index=ct_scores.index)


# --------------------------------------------------------------------------- #
# Loading the fitted model
# --------------------------------------------------------------------------- #
def load_threshold(refit_dir):
    """Read the operating threshold fixed on the development cohort."""
    path = os.path.join(refit_dir, "refit_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no refit summary at {path}")
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if "threshold" not in summary:
        raise KeyError(f"no threshold recorded in {path}")
    return float(summary["threshold"])


def load_pipeline(refit_dir):
    """Load the refitted pipeline after checking it was fitted on these features."""
    params_path = os.path.join(refit_dir, "refit_pipeline_params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"no pipeline parameters at {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if params.get("feature_cols") != FEATURE_COLS:
        raise ValueError(
            f"feature mismatch with {params_path}: "
            f"fitted on {params.get('feature_cols')}, requested {FEATURE_COLS}"
        )
    if params.get("categorical_cols") != CATEGORICAL_COLS:
        raise ValueError(
            f"categorical feature mismatch with {params_path}: "
            f"fitted on {params.get('categorical_cols')}, requested {CATEGORICAL_COLS}"
        )
    with open(os.path.join(refit_dir, "refit_pipeline.pkl"), "rb") as f:
        return pickle.load(f)


def predict_proba(pipeline, X, margin):
    """Predict with an offset, which Pipeline.predict_proba cannot forward."""
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    return pipeline.named_steps["classifier"].predict_proba(
        transformed, base_margin=np.asarray(margin, dtype=float)
    )[:, 1]


def variant_probabilities(variant, X, ct_scores, refit_dir):
    if variant == "ct_only":
        return ct_scores.to_numpy(dtype=float)
    pipeline = load_pipeline(refit_dir)
    return predict_proba(pipeline, X, base_margin(variant, ct_scores))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion_counts(y_true, y_pred):
    return {
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


def classification_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    counts = confusion_counts(y_true, (y_prob >= threshold).astype(int))
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    nan = float("nan")
    return {
        "AUC": roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else nan,
        "ACC": (tp + tn) / len(y_true) if len(y_true) else nan,
        "SEN": tp / (tp + fn) if tp + fn else nan,
        "SPE": tn / (tn + fp) if tn + fp else nan,
        "PPV": tp / (tp + fp) if tp + fp else nan,
        "NPV": tn / (tn + fn) if tn + fn else nan,
    }


def bootstrap_ci(y_true, y_prob, threshold, n_resamples, seed):
    """Percentile confidence intervals from resampling with replacement.

    Resamples drawing a single class are discarded, as the AUC is then
    undefined. The point estimates come from the full cohort, not from the
    resamples.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = y_true.size

    samples = {metric: [] for metric in METRICS}
    n_used = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if np.unique(y_true[idx]).size < 2:
            continue
        metrics = classification_metrics(y_true[idx], y_prob[idx], threshold)
        for metric in METRICS:
            samples[metric].append(metrics[metric])
        n_used += 1

    point = classification_metrics(y_true, y_prob, threshold)
    nan = float("nan")
    intervals = {}
    for metric in METRICS:
        arr = np.asarray(samples[metric], dtype=float)
        arr = arr[~np.isnan(arr)]
        lower, upper = (np.percentile(arr, CI_PERCENTILES) if arr.size
                        else (nan, nan))
        intervals[metric] = {
            "point": round(float(point[metric]), 3),
            "ci_lower": round(float(lower), 3),
            "ci_upper": round(float(upper), 3),
            "formatted": f"{point[metric]:.3f} ({lower:.3f}-{upper:.3f})",
        }
    return intervals, point, n_used


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions(ids, probs, y_true, threshold, path):
    probs = np.asarray(probs, dtype=float)
    pd.DataFrame({
        "ID": np.asarray(ids),
        "prob": probs,
        "predicted_label": (probs >= threshold).astype(int),
        "true_label": np.asarray(y_true).astype(int),
    }).to_csv(path, index=False)


def save_bootstrap_ci(intervals, path):
    pd.DataFrame([
        {
            "metric": metric,
            "point": intervals[metric]["point"],
            "ci_lower": intervals[metric]["ci_lower"],
            "ci_upper": intervals[metric]["ci_upper"],
            "formatted": intervals[metric]["formatted"],
        }
        for metric in METRICS
    ]).to_csv(path, index=False)


def save_variant_comparison(summaries, path):
    rows = []
    for variant, summary in summaries.items():
        row = {"variant": variant, "threshold": summary["threshold"]}
        row.update({metric: summary["bootstrap"]["results"][metric]
                    for metric in METRICS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(variant, X, y, ids, ct_scores, refit_dir, out_dir,
             n_bootstrap, seed):
    os.makedirs(out_dir, exist_ok=True)

    threshold = load_threshold(refit_dir)
    prob = variant_probabilities(variant, X, ct_scores, refit_dir)
    intervals, point, n_used = bootstrap_ci(y, prob, threshold,
                                            n_bootstrap, seed)
    counts = confusion_counts(np.asarray(y).astype(int),
                              (prob >= threshold).astype(int))

    save_predictions(ids, prob, y, threshold,
                     os.path.join(out_dir, "test_predictions.csv"))
    save_bootstrap_ci(intervals, os.path.join(out_dir, "test_bootstrap_ci.csv"))

    summary = {
        "variant": variant,
        "n_samples": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "feature_cols": [CT_SCORE_COL] if variant == "ct_only" else FEATURE_COLS,
        "threshold": round(threshold, 6),
        "threshold_source": "refit summary of the development cohort",
        "metrics_at_threshold": {k: round(float(v), 6) for k, v in point.items()},
        "confusion_counts": counts,
        "bootstrap": {
            "n_resamples": int(n_bootstrap),
            "n_resamples_used": int(n_used),
            "seed": int(seed),
            "percentiles": list(CI_PERCENTILES),
            "results": {metric: intervals[metric]["formatted"]
                        for metric in METRICS},
        },
    }
    save_json(summary, os.path.join(out_dir, "test_summary.json"))

    logger.info("%s | threshold %.4f | test AUC %s",
                variant, threshold, intervals["AUC"]["formatted"])
    return summary


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the multimodal fusion variants on the NAIC test cohort."
    )
    parser.add_argument("--model-dir", default=MODEL_DIR,
                        help="Output directory written by the training script.")
    parser.add_argument("--data-file", default=DATA_FILE)
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    df = pd.read_excel(args.data_file, sheet_name=args.sheet_name,
                       na_values=["NA", "na", "N/A", ""])
    required = [ID_COL, LABEL_COL, CT_SCORE_COL] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {args.data_file}: {missing}")

    ids = df[ID_COL].reset_index(drop=True)
    y = df[LABEL_COL].reset_index(drop=True).astype(int)
    ct_scores = df[CT_SCORE_COL].reset_index(drop=True).astype(float)
    if ct_scores.isna().any():
        raise ValueError(
            f"{CT_SCORE_COL} has {int(ct_scores.isna().sum())} missing values"
        )
    X = prepare_features(df)

    logger.info("loaded %d samples | positive rate %.1f%%",
                len(df), 100 * y.mean())
    for col, count in X.isna().sum().items():
        if count:
            logger.info("%s | %d missing values (%.1f%%)",
                        col, int(count), 100 * count / len(X))

    summaries = {}
    for variant in VARIANTS:
        variant_dir = os.path.join(args.model_dir, variant)
        refit_dir = os.path.join(variant_dir, "refit")
        if not os.path.isdir(refit_dir):
            raise FileNotFoundError(f"no refit directory at {refit_dir}")
        summaries[variant] = evaluate(
            variant, X, y, ids, ct_scores, refit_dir,
            os.path.join(variant_dir, "test"),
            args.n_bootstrap, args.seed,
        )

    save_variant_comparison(
        summaries, os.path.join(args.model_dir, "test_variant_comparison.csv")
    )
    logger.info("done. results in %s", args.model_dir)


if __name__ == "__main__":
    main()