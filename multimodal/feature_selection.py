"""Select features for the multimodal model by SHAP importance.

The procedure has three stages:

    1. Fit one model per seed and fold over repeated stratified k-fold
       cross-validation using every candidate feature, and record the mean
       absolute SHAP value of each feature on the held-out fold.
    2. Aggregate those values across all folds into an importance score and a
       measure of how stable each feature's rank is.
    3. Take the top K features for several values of K and re-evaluate each
       subset by repeated cross-validation, alongside the full-feature model.

Usage:
    python feature_selection.py --config configs/feature_selection.yaml
"""

import argparse
import json
import logging
import os
import pickle
from collections import namedtuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

MARGIN_EPS = 1e-3
MIN_ROUNDS = 1
RANK_FREQUENCY_CUTOFFS = (5, 10, 15)

Columns = namedtuple("Columns", ["categorical", "numerical"])


def split_columns(feature_cols, categorical_cols):
    """Order the features as the preprocessor emits them: categorical first."""
    return Columns(
        categorical=[c for c in feature_cols if c in categorical_cols],
        numerical=[c for c in feature_cols if c not in categorical_cols],
    )


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare_features(df, config):
    X = df[config["feature_cols"]].reset_index(drop=True)
    for col in X.columns:
        if col not in config["categorical_cols"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def add_missing_indicators(X, indicator_groups):
    """Add a 0/1 column recording whether a block of features was measured."""
    X = X.copy()
    for name, cols in indicator_groups.items():
        present = [c for c in cols if c in X.columns]
        if not present:
            raise ValueError(f"indicator {name} references no known column")
        X[name] = X[present].notna().any(axis=1).astype(int)
        logger.info("%s | %d of %d samples have the data",
                    name, int(X[name].sum()), len(X))
    return X


def logit_margin(ct_scores):
    """Convert CT probabilities to the log-odds offset added to the raw score."""
    p = np.clip(ct_scores.to_numpy(dtype=float), MARGIN_EPS, 1 - MARGIN_EPS)
    return pd.Series(np.log(p / (1 - p)), index=ct_scores.index)


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def build_preprocessor(columns):
    transformers = []
    if columns.categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1)),
            ]),
            columns.categorical,
        ))
    if columns.numerical:
        transformers.append(("num", "passthrough", columns.numerical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def scale_pos_weight(y):
    positives = int((y == 1).sum())
    return float((y == 0).sum() / positives) if positives else 1.0


def fit_pipeline(X, y, margin, columns, config, n_rounds, spw, seed):
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
        ("preprocessor", build_preprocessor(columns)),
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


def tune_n_rounds(X, y, margin, columns, config, spw, seed):
    """Choose the number of boosting rounds by inner cross-validation.

    The selected count is scaled by n / (n - 1) because each inner fold is
    fitted on that fraction of the data the returned model will see.
    """
    preprocessor = build_preprocessor(columns)
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
    return max(int(round((best + 1) * n_inner / (n_inner - 1))), MIN_ROUNDS)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def mean_abs_shap(pipeline, X, columns):
    """Mean absolute SHAP value per feature over the given rows."""
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    values = shap.TreeExplainer(
        pipeline.named_steps["classifier"]
    ).shap_values(transformed)

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]

    order = list(columns.categorical) + list(columns.numerical)
    if values.shape[1] != len(order):
        raise ValueError(
            f"expected {len(order)} SHAP columns, got {values.shape[1]}"
        )
    return dict(zip(order, np.abs(values).mean(axis=0)))


def shap_stability(shap_records, feature_cols, force_include):
    """Summarise the per-fold SHAP values into a score and a rank stability."""
    values = np.array(
        [[record[feature] for feature in feature_cols] for record in shap_records],
        dtype=float,
    )
    ranks = np.empty_like(values, dtype=int)
    for i, row in enumerate(values):
        order = np.argsort(-row)
        ranks[i, order] = np.arange(1, len(order) + 1)

    rows = []
    for j, feature in enumerate(feature_cols):
        column, rank = values[:, j], ranks[:, j]
        row = {
            "feature": feature,
            "mean_abs_shap": round(float(column.mean()), 6),
            "std_abs_shap": round(float(column.std(ddof=1)), 6),
            "median_abs_shap": round(float(np.median(column)), 6),
            "mean_rank": round(float(rank.mean()), 2),
            "median_rank": int(np.median(rank)),
            "force_included": feature in force_include,
        }
        for cutoff in RANK_FREQUENCY_CUTOFFS:
            row[f"top{cutoff}_freq"] = round(float((rank <= cutoff).mean()), 3)
        rows.append(row)

    return (pd.DataFrame(rows)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def select_top_k(stability, k, force_include):
    selected = stability["feature"].head(k).tolist()
    known = set(stability["feature"])
    for feature in force_include:
        if feature in known and feature not in selected:
            selected.append(feature)
    return selected


# --------------------------------------------------------------------------- #
# Repeated cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(X, y, ids, margin, columns, config, n_folds,
                   collect_shap=False, out_dir=None):
    """Repeat stratified k-fold cross-validation over every configured seed.

    Returns one AUC record per fold and, when requested, one record of mean
    absolute SHAP values per fold.
    """
    auc_records, shap_records = [], []

    for seed in config["seeds"]:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=int(seed))
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)
            margin_train = margin.iloc[train_idx].reset_index(drop=True)
            margin_val = margin.iloc[val_idx].reset_index(drop=True)

            spw = scale_pos_weight(y_train)
            n_rounds = tune_n_rounds(X_train, y_train, margin_train, columns,
                                     config, spw, seed)
            pipeline = fit_pipeline(X_train, y_train, margin_train, columns,
                                    config, n_rounds, spw, seed)
            train_prob = predict_proba(pipeline, X_train, margin_train)
            val_prob = predict_proba(pipeline, X_val, margin_val)

            auc_records.append({
                "seed": seed,
                "fold": fold,
                "n_rounds": n_rounds,
                "train_auc": round(float(roc_auc_score(y_train, train_prob)), 6),
                "val_auc": round(float(roc_auc_score(y_val, val_prob)), 6),
            })

            if collect_shap:
                shap_records.append(mean_abs_shap(pipeline, X_val, columns))

            if out_dir is not None:
                fold_dir = os.path.join(out_dir, f"seed_{seed}", f"fold_{fold}")
                os.makedirs(fold_dir, exist_ok=True)
                save_predictions(
                    ids.iloc[val_idx], val_prob, y_val,
                    os.path.join(fold_dir, "val_predictions.csv"),
                )
                with open(os.path.join(fold_dir, "pipeline.pkl"), "wb") as f:
                    pickle.dump(pipeline, f)

    return auc_records, shap_records


def summarize_by_seed(auc_records):
    frame = pd.DataFrame(auc_records)
    rows = []
    for seed, group in frame.groupby("seed", sort=True):
        rows.append({
            "seed": int(seed),
            "mean_train_auc": round(float(group["train_auc"].mean()), 6),
            "mean_val_auc": round(float(group["val_auc"].mean()), 6),
            "std_val_auc": round(float(group["val_auc"].std(ddof=1)), 6),
            "min_val_auc": round(float(group["val_auc"].min()), 6),
            "max_val_auc": round(float(group["val_auc"].max()), 6),
            "mean_n_rounds": round(float(group["n_rounds"].mean()), 1),
        })
    return pd.DataFrame(rows)


def summarize_subset(name, features, n_folds, seed_summary):
    return {
        "config": name,
        "n_features": len(features),
        "n_folds": n_folds,
        "n_seeds": len(seed_summary),
        "mean_val_auc": round(float(seed_summary["mean_val_auc"].mean()), 6),
        "std_val_auc": round(float(seed_summary["mean_val_auc"].std(ddof=1)), 6),
        "min_val_auc": round(float(seed_summary["mean_val_auc"].min()), 6),
        "max_val_auc": round(float(seed_summary["mean_val_auc"].max()), 6),
        "mean_train_auc": round(float(seed_summary["mean_train_auc"].mean()), 6),
        "worst_fold_val_auc": round(float(seed_summary["min_val_auc"].min()), 6),
        "features": ",".join(features),
    }


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


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Select features for the multimodal model by SHAP importance."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

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
    margin = logit_margin(ct_scores)

    X = add_missing_indicators(prepare_features(df, config),
                               config["missing_indicators"])
    feature_cols = list(X.columns)
    columns = split_columns(feature_cols, config["categorical_cols"])

    logger.info("loaded %d samples | %d features | positive rate %.1f%%",
                len(df), len(feature_cols), 100 * y.mean())

    fold_dir = (os.path.join(out_dir, "folds")
                if config["save_fold_outputs"] else None)
    auc_records, shap_records = cross_validate(
        X, y, ids, margin, columns, config,
        n_folds=config["n_folds_selection"], collect_shap=True, out_dir=fold_dir,
    )
    pd.DataFrame(auc_records).to_csv(
        os.path.join(out_dir, "full_model_fold_auc.csv"), index=False)
    baseline = summarize_by_seed(auc_records)
    baseline.to_csv(os.path.join(out_dir, "full_model_seed_summary.csv"),
                    index=False)
    logger.info("full model | mean val AUC %.4f over %d folds",
                baseline["mean_val_auc"].mean(), len(auc_records))

    stability = shap_stability(shap_records, feature_cols,
                               config["force_include"])
    stability.to_csv(os.path.join(out_dir, "shap_stability.csv"), index=False)
    logger.info("highest mean |SHAP|: %s",
                ", ".join(stability["feature"].head(10)))

    comparison = [summarize_subset(
        f"all_{len(feature_cols)}_features", feature_cols,
        config["n_folds_selection"], baseline,
    )]
    subsets = {}

    for k in config["top_k"]:
        subset = select_top_k(stability, k, config["force_include"])
        subsets[f"top{k}"] = subset
        records, _ = cross_validate(
            X[subset], y, ids, margin,
            split_columns(subset, config["categorical_cols"]),
            config, n_folds=config["n_folds_subset"],
        )
        summary = summarize_subset(
            f"top{k}", subset, config["n_folds_subset"],
            summarize_by_seed(records),
        )
        comparison.append(summary)
        logger.info("top%d | %d features | mean val AUC %.4f",
                    k, len(subset), summary["mean_val_auc"])

    save_json(subsets, os.path.join(out_dir, "candidate_subsets.json"))
    (pd.DataFrame(comparison)
     .sort_values("mean_val_auc", ascending=False)
     .to_csv(os.path.join(out_dir, "subset_comparison.csv"), index=False))
    logger.info("done. results in %s", out_dir)


if __name__ == "__main__":
    main()