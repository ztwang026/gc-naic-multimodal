"""Train the dual-branch CT model.

A single trainer used for two settings, selected by the YAML config:

    configs/pretrain_nac.yaml    NAC pretraining (NAC cohort)
    configs/naic_baseline.yaml   NAIC de-novo baseline (without NAC pretraining)

Both settings run stratified k-fold cross-validation for evaluation and then
fit a final model on the full dataset. Per-fold splits, epoch metrics,
per-epoch predictions and the final weights are written to the output
directory given in the config.

Usage:
    python train.py --config configs/naic_baseline.yaml
"""

import argparse
import importlib
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StomachCancerDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

MIXUP_ALPHA = 0.2

# Supported backbones, selected by the `backbone` key of the config file.
BACKBONES = {
    "resnet18": ("model.dual_resnet18_model", "DualResnet18Classifier"),
    "resnet34": ("model.dual_resnet34_model", "DualResnet34Classifier"),
    "resnet50": ("model.dual_resnet50_model", "DualResnet50Classifier"),
    "resnet101": ("model.dual_resnet101_model", "DualResnet101Classifier"),
}
# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Mixup
# --------------------------------------------------------------------------- #
def mixup_data(primary_img, lymph_img, label, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(primary_img.size(0)).to(primary_img.device)
    mixed_primary = lam * primary_img + (1 - lam) * primary_img[index]
    mixed_lymph = lam * lymph_img + (1 - lam) * lymph_img[index]
    return mixed_primary, mixed_lymph, label, label[index], lam


def mixup_criterion(criterion, output, label_a, label_b, lam):
    return lam * criterion(output, label_a) + (1 - lam) * criterion(output, label_b)


def safe_auc(y_true, y_score):
    """ROC-AUC that returns NaN when a batch/fold contains a single class."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# Train / evaluate one epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    y_true, y_score, ids = [], [], []

    for primary_img, lymph_img, label, ctid in tqdm(
        loader, desc=f"train epoch {epoch + 1}", leave=False
    ):
        primary_img = primary_img.to(device, dtype=torch.float)
        lymph_img = lymph_img.to(device, dtype=torch.float)
        label = label.to(device)

        mixed_primary, mixed_lymph, label_a, label_b, lam = mixup_data(
            primary_img, lymph_img, label
        )
        optimizer.zero_grad()
        output = model(mixed_primary, mixed_lymph)
        loss = mixup_criterion(criterion, output, label_a, label_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)

        # Score on the un-mixed images so the training AUC is interpretable.
        with torch.no_grad():
            prob = torch.softmax(model(primary_img, lymph_img), dim=1)[:, 1]
        y_true.extend(label.cpu().numpy())
        y_score.extend(prob.cpu().numpy())
        ids.extend(ctid)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, safe_auc(y_true, y_score), ids, y_true, y_score


@torch.no_grad()
def evaluate(model, loader, device, epoch):
    model.eval()
    y_true, y_score, ids = [], [], []

    for primary_img, lymph_img, label, ctid in tqdm(
        loader, desc=f"eval epoch {epoch + 1}", leave=False
    ):
        primary_img = primary_img.to(device, dtype=torch.float)
        lymph_img = lymph_img.to(device, dtype=torch.float)
        prob = torch.softmax(model(primary_img, lymph_img), dim=1)[:, 1]
        y_true.extend(label.numpy())
        y_score.extend(prob.cpu().numpy())
        ids.extend(ctid)

    return safe_auc(y_true, y_score), ids, y_true, y_score


# --------------------------------------------------------------------------- #
# Prediction bookkeeping
# --------------------------------------------------------------------------- #
def collect_preds(store, ids, y_true, y_score, epoch):
    for i, ctid in enumerate(ids):
        store.setdefault(ctid, {"ctid": ctid, "true_label": y_true[i]})
        store[ctid][f"epoch{epoch + 1}_pred_prob"] = y_score[i]


def preds_to_frame(store, n_epochs):
    cols = ["ctid", "true_label"] + [
        f"epoch{e}_pred_prob" for e in range(1, n_epochs + 1)
    ]
    return pd.DataFrame(list(store.values()))[cols]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_model(config, device):
    module_name, class_name = BACKBONES[config["backbone"]]
    classifier = getattr(importlib.import_module(module_name), class_name)
    return classifier(
        pretrained=True, num_classes=2, dropout_rate=config["dropout_rate"]
    ).to(device)


def build_optimizer(model, config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )
    return optimizer, scheduler


def make_dataset(df, out_dir, name, config, test):
    """Persist a split to disk and build the dataset it is read from."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.xlsx")
    df.to_excel(path, index=False)
    return StomachCancerDataset(
        csv_file=path, dataset_root=config["dataset_root"], test=test
    )


def make_loader(dataset, config, shuffle, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


# --------------------------------------------------------------------------- #
# Core training loop (shared by CV folds and the full-dataset run)
# --------------------------------------------------------------------------- #
def run_training(train_loader, val_loader, config, out_dir, device, n_epochs):
    """Train for n_epochs, writing metrics/predictions/weights to out_dir.

    If val_loader is given, the model is evaluated every epoch and the
    final-epoch validation AUC is returned; otherwise NaN is returned.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = build_optimizer(model, config)

    metrics, train_store, val_store = [], {}, {}
    final_val_auc = float("nan")

    for epoch in range(n_epochs):
        train_loss, train_auc, tr_ids, tr_true, tr_score = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        collect_preds(train_store, tr_ids, tr_true, tr_score, epoch)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "lr": scheduler.get_last_lr()[0],
        }

        if val_loader is not None:
            val_auc, va_ids, va_true, va_score = evaluate(
                model, val_loader, device, epoch
            )
            collect_preds(val_store, va_ids, va_true, va_score, epoch)
            row["val_auc"] = val_auc
            final_val_auc = val_auc
            logger.info(
                "epoch %d/%d | loss %.4f | train_auc %.4f | val_auc %.4f",
                epoch + 1, n_epochs, train_loss, train_auc, val_auc,
            )
        else:
            logger.info(
                "epoch %d/%d | loss %.4f | train_auc %.4f",
                epoch + 1, n_epochs, train_loss, train_auc,
            )

        metrics.append(row)
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(out_dir, f"epoch{n_epochs}.pth"))
    pd.DataFrame(metrics).to_csv(
        os.path.join(out_dir, "training_metrics.csv"), index=False
    )
    preds_to_frame(train_store, n_epochs).to_csv(
        os.path.join(out_dir, "train_pred.csv"), index=False
    )
    if val_loader is not None:
        preds_to_frame(val_store, n_epochs).to_csv(
            os.path.join(out_dir, "val_pred.csv"), index=False
        )
    return final_val_auc


# --------------------------------------------------------------------------- #
# Cross-validation and full-dataset training
# --------------------------------------------------------------------------- #
def cross_validate(full_df, config, device):
    skf = StratifiedKFold(
        n_splits=config["n_folds"], shuffle=True, random_state=config["seed"]
    )
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["label"])):
        setup_seed(config["seed"] + fold)
        fold_dir = os.path.join(config["output_dir"], f"fold_{fold}")

        train_ds = make_dataset(
            full_df.iloc[train_idx].reset_index(drop=True),
            fold_dir, f"train_fold_{fold}", config, test=False,
        )
        val_ds = make_dataset(
            full_df.iloc[val_idx].reset_index(drop=True),
            fold_dir, f"val_fold_{fold}", config, test=True,
        )
        train_loader = make_loader(
            train_ds, config, shuffle=True, seed=config["seed"] + fold
        )
        val_loader = make_loader(val_ds, config, shuffle=False)

        logger.info(
            "fold %d/%d | train %d | val %d",
            fold + 1, config["n_folds"], len(train_ds), len(val_ds),
        )
        fold_aucs.append(
            run_training(train_loader, val_loader, config, fold_dir, device,
                         config["epochs"])
        )
        torch.cuda.empty_cache()

    return fold_aucs


def train_full(full_df, config, device):
    setup_seed(config["seed"])
    full_dir = os.path.join(config["output_dir"], "full_dataset")
    full_ds = make_dataset(full_df, full_dir, "full_dataset", config, test=False)
    full_loader = make_loader(full_ds, config, shuffle=True, seed=config["seed"])
    logger.info("full dataset | n = %d", len(full_ds))
    run_training(full_loader, None, config, full_dir, device,
                 config["full_dataset_epochs"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Train the dual-branch CT model."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config.get("backbone") not in BACKBONES:
        raise ValueError(
            f"config key 'backbone' must be one of {sorted(BACKBONES)}; "
            f"got {config.get('backbone')!r}"
        )

    setup_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    full_df = pd.read_excel(config["data_file"], sheet_name=config["sheet_name"])
    full_df = full_df.rename(columns={"CT_id": "ctid", "Label": "label"})
    full_df = full_df[["ctid", "label"]]
    logger.info(
        "backbone %s | loaded %d samples | class counts %s",
        config["backbone"], len(full_df),
        dict(full_df["label"].value_counts().sort_index()),
    )

    fold_aucs = cross_validate(full_df, config, device)
    mean_auc, std_auc = float(np.nanmean(fold_aucs)), float(np.nanstd(fold_aucs))

    summary = pd.DataFrame(
        [{"fold": i + 1, "val_auc": a} for i, a in enumerate(fold_aucs)]
        + [{"fold": "mean", "val_auc": mean_auc},
           {"fold": "std", "val_auc": std_auc}]
    )
    summary.to_csv(os.path.join(config["output_dir"], "cv_summary.csv"), index=False)
    logger.info("cross-validation AUC: %.4f +/- %.4f", mean_auc, std_auc)

    train_full(full_df, config, device)
    logger.info("done. results in %s", config["output_dir"])


if __name__ == "__main__":
    main()