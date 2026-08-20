"""Fit the multimodal fusion model on the NAIC development cohort.

Three variants are evaluated from a single config:

    ct_only         the CT score used directly as the predicted probability
    clinical_only   gradient boosting on the tabular features alone
    multimodal      gradient boosting on the tabular features with the CT score
                    supplied as a fixed offset, so the trees fit a residual
                    correction on top of the CT model

Each variant is assessed by stratified k-fold cross-validation and then
refitted on the full development cohort. The number of boosting rounds is
chosen by an inner cross-validation nested inside every outer fold. The
operating threshold is taken from the pooled out-of-fold predictions and is
reused unchanged for the refitted model, so it is fixed before any external
evaluation.

Missing values in the numerical features are left to the native handling of
XGBoost and are not imputed.

Usage:
    python multimodal_fusion_xgb.py --config configs/multimodal_fusion.yaml
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

METRICS = ("AUC", "ACC", "SEN", "SPE", "PPV", "NPV")
SPLITS = ("train", "val")
MIN_ROUNDS = 1


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_features(df, config):
    X = df[config["feature_cols"]].reset_index(drop=True)
    for col in X.columns:
        if col not in config["categorical_cols"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def base_margin(variant, ct_scores, eps=1e-3):
    """Offset added to the raw model score, on the log-odds scale."""
    if variant != "multimodal":
        return pd.Series(np.zeros(len(ct_scores)), index=ct_scores.index)
    p = np.clip(ct_scores.to_numpy(dtype=float), eps, 1 - eps)
    return pd.Series(np.log(p / (1 - p)), index=ct_scores.index)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_preprocessor(config):
    transformers = []
    if config["categorical_cols"]:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)),
            ]),
            config["categorical_cols"],
        ))
    numerical_cols = [c for c in config["feature_cols"]
                      if c not in config["categorical_cols"]]
    if numerical_cols:
        transformers.append(("num", "passthrough", numerical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def scale_pos_weight(y):
    positives = int((y == 1).sum())
    return float((y == 0).sum() / positives) if positives else 1.0


def fit_pipeline(X, y, margin, config, n_rounds, spw, seed):
    classifier = XGBClassifier(
        n_estimators=int(n_rounds),
        scale_pos_weight=float(spw),
        random_state=int(seed),
        n_jobs=-1,
        eval_metric="auc",
        tree_method="hist",
        **config["xgb_params"],
    )
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(config)),
        ("classifier", classifier),
    ])
    pipeline.fit(X, y, classifier__base_margin=np.asarray(margin, dtype=float))
    return pipeline


def predict_proba(pipeline, X, margin):
    """Predict with an offset, which Pipeline.predict_proba cannot forward."""
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    return pipeline.named_steps["classifier"].predict_proba(
        transformed, base_margin=np.asarray(margin, dtype=float)
    )[:, 1]


def tune_n_rounds(X, y, margin, config, spw, seed):
    """Choose the number of boosting rounds by inner cross-validation.

    The selected count is scaled by n / (n - 1) because each inner fold is
    fitted on that fraction of the data the returned model will see.
    """
    preprocessor = build_preprocessor(config)
    dtrain = xgb.DMatrix(
        preprocessor.fit_transform(X, y),
        label=np.asarray(y, dtype=int),
        base_margin=np.asarray(margin, dtype=float),
    )
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": float(spw),
        "seed": int(seed),
        "nthread": -1,
        "verbosity": 0,
        **config["xgb_params"],
    }
    n_inner = config["n_inner_folds"]
    results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=config["n_rounds_max"],
        nfold=n_inner,
        stratified=True,
        early_stopping_rounds=config["early_stopping_rounds"],
        seed=int(seed) + config["inner_cv_seed_offset"],
        verbose_eval=False,
    )
    best = int(results["test-auc-mean"].idxmax())
    n_rounds_raw = best + 1
    n_rounds = max(int(round(n_rounds_raw * n_inner / (n_inner - 1))), MIN_ROUNDS)
    inner = {
        "inner_cv_test_auc": float(results["test-auc-mean"].iloc[best]),
        "inner_cv_test_auc_std": float(results["test-auc-std"].iloc[best]),
        "inner_cv_train_auc": float(results["train-auc-mean"].iloc[best]),
        "inner_cv_rounds_explored": int(len(results)),
    }
    return n_rounds, n_rounds_raw, inner


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def youden_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, np.asarray(y_prob, dtype=float))
    return float(thresholds[int(np.argmax(tpr - fpr))])


def classification_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    nan = float("nan")
    return {
        "AUC": roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else nan,
        "ACC": (tp + tn) / len(y_true) if len(y_true) else nan,
        "SEN": tp / (tp + fn) if tp + fn else nan,
        "SPE": tn / (tn + fp) if tn + fp else nan,
        "PPV": tp / (tp + fp) if tp + fp else nan,
        "NPV": tn / (tn + fn) if tn + fn else nan,
    }


def mean_std(values, decimals=4):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "nan +/- nan"
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    return f"{arr.mean():.{decimals}f} +/- {std:.{decimals}f}"


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_predictions(ids, probs, y_true, path):
    pd.DataFrame({
        "ID": np.asarray(ids),
        "prob": np.asarray(probs, dtype=float),
        "true_label": np.asarray(y_true).astype(int),
    }).to_csv(path, index=False)


def save_fold_metrics(records, path):
    rows = []
    for record in records:
        row = {
            "fold": record["fold"],
            "n_rounds": record["n_rounds"] or "",
            "n_rounds_raw": record["n_rounds_raw"] or "",
            "threshold": round(record["threshold"], 6),
        }
        for split in SPLITS:
            for metric in METRICS:
                row[f"{split}_{metric}"] = round(record[f"{split}_{metric}"], 6)
        rows.append(row)

    summary = {"fold": "mean +/- std", "n_rounds": "", "n_rounds_raw": "",
               "threshold": ""}
    for split in SPLITS:
        for metric in METRICS:
            summary[f"{split}_{metric}"] = mean_std(
                [r[f"{split}_{metric}"] for r in records]
            )
    rows.append(summary)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_pipeline(pipeline, config, n_rounds, out_dir, name):
    with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    categories = {}
    if config["categorical_cols"]:
        encoder = (pipeline.named_steps["preprocessor"]
                   .named_transformers_["cat"].named_steps["encoder"])
        categories = {col: encoder.categories_[i].tolist()
                      for i, col in enumerate(config["categorical_cols"])}
    save_json({
        "feature_cols": config["feature_cols"],
        "categorical_cols": config["categorical_cols"],
        "ordinal_encoding": categories,
        "n_rounds": int(n_rounds),
        "uses_base_margin": True,
    }, os.path.join(out_dir, f"{name}_params.json"))


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(variant, X, y, ids, ct_scores, margin, config, out_dir):
    """Run outer k-fold CV and return the fold records and the OOF threshold."""
    os.makedirs(out_dir, exist_ok=True)
    seed = config["cv_seed"]
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True,
                          random_state=seed)

    records = []
    oof_ids, oof_probs, oof_true = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        ids_train = ids.iloc[train_idx].reset_index(drop=True)
        ids_val = ids.iloc[val_idx].reset_index(drop=True)

        if variant == "ct_only":
            train_prob = ct_scores.iloc[train_idx].to_numpy(dtype=float)
            val_prob = ct_scores.iloc[val_idx].to_numpy(dtype=float)
            n_rounds = n_rounds_raw = None
        else:
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            margin_train = margin.iloc[train_idx].reset_index(drop=True)
            margin_val = margin.iloc[val_idx].reset_index(drop=True)

            spw = scale_pos_weight(y_train)
            n_rounds, n_rounds_raw, _ = tune_n_rounds(
                X_train, y_train, margin_train, config, spw, seed
            )
            pipeline = fit_pipeline(
                X_train, y_train, margin_train, config, n_rounds, spw, seed
            )
            train_prob = predict_proba(pipeline, X_train, margin_train)
            val_prob = predict_proba(pipeline, X_val, margin_val)
            save_pipeline(pipeline, config, n_rounds, fold_dir, "pipeline")

        threshold = youden_threshold(y_train, train_prob)
        train_metrics = classification_metrics(y_train, train_prob, threshold)
        val_metrics = classification_metrics(y_val, val_prob, threshold)

        save_predictions(ids_train, train_prob, y_train,
                         os.path.join(fold_dir, "train_predictions.csv"))
        save_predictions(ids_val, val_prob, y_val,
                         os.path.join(fold_dir, "val_predictions.csv"))

        oof_ids.extend(ids_val.tolist())
        oof_probs.extend(val_prob.tolist())
        oof_true.extend(y_val.tolist())

        records.append({
            "fold": fold,
            "threshold": threshold,
            "n_rounds": n_rounds,
            "n_rounds_raw": n_rounds_raw,
            **{f"train_{m}": train_metrics[m] for m in METRICS},
            **{f"val_{m}": val_metrics[m] for m in METRICS},
        })

    save_fold_metrics(records, os.path.join(out_dir, "fold_metrics.csv"))

    oof_prob = np.asarray(oof_probs, dtype=float)
    oof_label = np.asarray(oof_true).astype(int)
    oof_threshold = youden_threshold(oof_label, oof_prob)
    save_predictions(oof_ids, oof_prob, oof_label,
                     os.path.join(out_dir, "oof_predictions.csv"))
    save_json({
        "threshold": round(oof_threshold, 6),
        "method": "maximum Youden J on pooled out-of-fold predictions",
        "n_samples": int(oof_label.size),
        "metrics_at_threshold": {
            k: round(float(v), 6) for k, v in
            classification_metrics(oof_label, oof_prob, oof_threshold).items()
        },
    }, os.path.join(out_dir, "threshold.json"))

    return records, oof_threshold


# --------------------------------------------------------------------------- #
# Refit on the full development cohort
# --------------------------------------------------------------------------- #
def refit(variant, X, y, ids, ct_scores, margin, config, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)
    summary = {"variant": variant, "n_samples": int(len(y))}

    if variant == "ct_only":
        prob = ct_scores.to_numpy(dtype=float)
    else:
        seed = config["refit_seed"]
        spw = scale_pos_weight(y)
        n_rounds, n_rounds_raw, inner = tune_n_rounds(
            X, y, margin, config, spw, seed
        )
        pipeline = fit_pipeline(X, y, margin, config, n_rounds, spw, seed)
        prob = predict_proba(pipeline, X, margin)
        save_pipeline(pipeline, config, n_rounds, out_dir, "refit_pipeline")
        summary.update({
            "seed": int(seed),
            "scale_pos_weight": round(spw, 4),
            "n_rounds": n_rounds,
            "n_rounds_raw": n_rounds_raw,
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in inner.items()},
        })

    summary["threshold"] = round(threshold, 6)
    summary["threshold_source"] = "pooled out-of-fold predictions of the CV stage"
    summary["in_sample_metrics"] = {
        k: round(float(v), 6)
        for k, v in classification_metrics(y, prob, threshold).items()
    }

    save_predictions(ids, prob, y,
                     os.path.join(out_dir, "refit_predictions.csv"))
    save_json(summary, os.path.join(out_dir, "refit_summary.json"))
    return summary


# --------------------------------------------------------------------------- #
# Variant comparison
# --------------------------------------------------------------------------- #
def save_variant_comparison(cv_records, refit_summaries, out_dir):
    rows = []
    for variant, records in cv_records.items():
        val_auc = [r["val_AUC"] for r in records]
        rows.append({
            "variant": variant,
            "cv_mean_val_AUC": round(float(np.nanmean(val_auc)), 6),
            "cv_std_val_AUC": round(float(np.nanstd(val_auc, ddof=1)), 6),
            "cv_min_val_AUC": round(float(np.nanmin(val_auc)), 6),
            "cv_max_val_AUC": round(float(np.nanmax(val_auc)), 6),
            "refit_in_sample_AUC":
                refit_summaries[variant]["in_sample_metrics"]["AUC"],
        })
    pd.DataFrame(rows).sort_values("cv_mean_val_AUC", ascending=False).to_csv(
        os.path.join(out_dir, "variant_comparison.csv"), index=False
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Fit the multimodal fusion model on the NAIC cohort."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output_dir"], exist_ok=True)

    df = pd.read_excel(config["data_file"], sheet_name=config["sheet_name"],
                       na_values=["NA", "na", "N/A", ""])
    required = ([config["id_col"], config["label_col"], config["ct_score_col"]]
                + config["feature_cols"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {config['data_file']}: {missing}")

    ids = df[config["id_col"]].reset_index(drop=True)
    y = df[config["label_col"]].reset_index(drop=True).astype(int)
    ct_scores = df[config["ct_score_col"]].reset_index(drop=True).astype(float)
    X = prepare_features(df, config)

    logger.info("loaded %d samples | positive rate %.1f%%",
                len(df), 100 * y.mean())

    cv_records, refit_summaries = {}, {}
    for variant in config["variants"]:
        variant_dir = os.path.join(config["output_dir"], variant)
        margin = base_margin(variant, ct_scores)

        records, threshold = cross_validate(
            variant, X, y, ids, ct_scores, margin, config,
            os.path.join(variant_dir, "cv"),
        )
        summary = refit(
            variant, X, y, ids, ct_scores, margin, config,
            os.path.join(variant_dir, "refit"), threshold,
        )

        cv_records[variant] = records
        refit_summaries[variant] = summary
        val_auc = [r["val_AUC"] for r in records]
        logger.info(
            "%s | cv val AUC %.4f +/- %.4f | threshold %.4f",
            variant, float(np.nanmean(val_auc)),
            float(np.nanstd(val_auc, ddof=1)), threshold,
        )

    save_variant_comparison(cv_records, refit_summaries, config["output_dir"])
    logger.info("done. results in %s", config["output_dir"])


if __name__ == "__main__":
    main()